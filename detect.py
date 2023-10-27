import pyttsx4
import threading
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from stereo.dianyuntu_yolo import preprocess, undistortion, getRectifyTransform, draw_line, rectifyImage, \
    stereoMatchSGBM
from stereo import stereoconfig


def detect(save_img=False):
    global num
    global dis
    global label

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        x = (xyxy[0] + xyxy[2]) / 2
                        y = (xyxy[1] + xyxy[3]) / 2
                        if x <= 1280:
                            t3 = time_synchronized()
                            p = num

                            height_0, width_0 = im0.shape[0:2]
                            iml = im0[0:int(height_0), 0:int(width_0 / 2)]
                            imr = im0[0:int(height_0), int(width_0 / 2):int(width_0)]

                            # camera calibration
                            height, width = iml.shape[0:2]
                            config = stereoconfig.stereoCamera()
                            map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
                            iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

                            line = draw_line(iml_rectified, imr_rectified)
                            iml = undistortion(iml, config.cam_matrix_left, config.distortion_l)
                            imr = undistortion(imr, config.cam_matrix_right, config.distortion_r)
                            iml_, imr_ = preprocess(iml, imr)
                            iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)

                            disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, True)
                            points_3d = cv2.reprojectImageTo3D(disp, Q)

                            text_cxy = "*"
                            cv2.putText(im0, text_cxy, (int(x), int(y)), cv2.FONT_ITALIC, 1.2, (0, 0, 255), 3)

                            print('点 (%d, %d) 的三维坐标 (x:%.1fcm, y:%.1fcm, z:%.1fcm)' % (
                                int(x), int(y), points_3d[int(y), int(x), 0] / 10, points_3d[int(y), int(x), 1] / 10,
                                points_3d[int(y), int(x), 2] / 10))

                            dis = ((points_3d[int(y), int(x), 0] ** 2 + points_3d[int(y), int(x), 1] ** 2 + points_3d[
                                int(y), int(x), 2] ** 2) ** 0.5) / 10
                            print('点 (%d, %d) 的 %s 距离左摄像头的相对距离为 %0.1f cm' % (x, y, label, dis))

                            text_x = "x:%.1fcm" % (points_3d[int(y), int(x), 0] / 10)
                            text_y = "y:%.1fcm" % (points_3d[int(y), int(x), 1] / 10)
                            text_z = "z:%.1fcm" % (points_3d[int(y), int(x), 2] / 10)
                            text_dis = "dis:%.1fcm" % dis

                            cv2.rectangle(im0, (int(xyxy[0] + (xyxy[2] - xyxy[0])), int(xyxy[1])),
                                          (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5 + 220), int(xyxy[1] + 150)),
                                          colors[int(cls)], -1)
                            cv2.putText(im0, text_x, (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5), int(xyxy[1] + 30)),
                                        cv2.FONT_ITALIC, 1.5, (255, 255, 255), 3)
                            cv2.putText(im0, text_y, (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5), int(xyxy[1] + 65)),
                                        cv2.FONT_ITALIC, 1.5, (255, 255, 255), 3)
                            cv2.putText(im0, text_z, (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5), int(xyxy[1] + 100)),
                                        cv2.FONT_ITALIC, 1.5, (255, 255, 255), 3)
                            cv2.putText(im0, text_dis, (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5), int(xyxy[1] + 145)),
                                        cv2.FONT_ITALIC, 1.5, (255, 255, 255), 3)

                            t4 = time_synchronized()
                            print(f'Done. ({t4 - t3:.3f}s)')

                    # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if save_img:
                result_save(dataset, save_path, im0, vid_path, vid_writer, vid_cap)

        if exit_flag.is_set():
            break

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    time.sleep(1)


def voice_broadcast():
    """
    when something in front of the camera just tell you what and distance
    """
    while True:
        dis_voice = dis
        label_voice = label
        engine = pyttsx4.init()
        engine.setProperty('rate', 500)
        engine.setProperty('volume', 1)
        if (dis_voice > 100 and dis_voice <= 1000 and label_voice != 0):
            engine.say('请注意，距离您%0.1f米处有%s' % (dis_voice / 100, label_voice))
            engine.runAndWait()
            engine.stop()

        # close the thread
        if exit_flag.is_set():
            break
        time.sleep(1)


def result_save(dataset, save_path, image, vid_path, vid_writer, vid_cap):
    if dataset.mode == 'image':
        cv2.imwrite(save_path, image)
        print(f" The image with the result is saved in: {save_path}")
    else:  # 'video' or 'stream'
        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, image.shape[1], image.shape[0]
                save_path += '.mp4'
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(image)
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", 2560, 720)
        cv2.moveWindow("Video", 0, 0)
        cv2.imshow("Video", image)
        cv2.waitKey(1)


def main():
    task1 = threading.Thread(target=detect)
    task2 = threading.Thread(target=voice_broadcast)
    task1.start()
    task2.start()
    # recognize the task is alive or not
    while task1.is_alive() and task2.is_alive():
        time.sleep(1)
    exit_flag.set()
    task1.join()
    task2.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    # init the config
    exit_flag = threading.Event()
    num = 200
    dis = 0
    label = 0

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                main()
                strip_optimizer(opt.weights)
        else:
            main()
