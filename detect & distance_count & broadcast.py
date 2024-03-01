 # -*- coding: utf-8 -*-
import argparse
import time
from pathlib import Path

import threading
from threading import Thread
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import numpy as np

import voice_
import judge_priority

from PIL import Image, ImageDraw, ImageFont

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from stereo.dianyuntu_yolo import preprocess, undistortion, getRectifyTransform, draw_line, rectifyImage,\
     stereoMatchSGBM#, hw3ToN3, view_cloud ,DepthColor2Cloud

from stereo import stereoconfig_040_2


from stereo.stereo import stereo_40

from stereo.stereo import get_median,stereo_threading,MyThread
#import numba
#from numba import cuda, vectorize
#from numba import njit, jit
#@cuda.jit
#@jit
#@vectorize(["float32 (float32 , float32 )"], target='cuda')



def detect(save_img=False): 
    ####
    dis_avg = 0
    x = 0
    label = 0 
    x_end = 0
    y_end = 0
    ####
    accel_frame = 0  
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://') )

    
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
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
        #make streams dir
        save_stream_dir = Path( increment_path(Path("./runs/streams") / opt.name, exist_ok=opt.exist_ok) )  # increment run
        #print("save_stream_dir is %s"%save_stream_dir)
        (save_stream_dir / 'labels' if save_txt else save_stream_dir).mkdir(parents=True, exist_ok=True)  # make stream dir

    
        flag = 0
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        # Directories
        save_dir = Path( increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok) )  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        print("img_size:")
        print(imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


    ###############################
    #stereo code

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    # 读取相机内参和外参
    config = stereoconfig_040_2.stereoCamera()

    
    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(720, 1280, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        
 
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            ###############################
            #stereo code
            fps_set = 60 #setting the frame
            if(accel_frame % fps_set == 0):
                #t3 = time.time() # stereo time start 0.510s
                #string = ''
                
                #thread = threading.Thread(target = mythread,args = ((config,im0,map1x, map1y, map2x, map2y,Q)) )
                
                thread= MyThread(stereo_threading,args = (config,im0,map1x, map1y, map2x, map2y,Q))
                thread.start()
                
                #if(accel_frame % fps_set == 0):
                #    thread.join()
                #points_3d = thread.get_result()
                print()
                print(threading.active_count())                              #获取已激活的线程数
                print(threading.enumerate())    # see the thread list，查看所有线程信息，一个<_MainThread(...)> 带多个 <Thread(...)>
                print()
                print("############## Frame is %d !##################" %accel_frame)

            

            p = Path(p)  # to Path
            if webcam:
                save_stream_path = str(save_stream_dir / "stream0.mp4")  # save streams path
            else:   
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            
           
            
            
            s += '%gx%g ' % img.shape[2:]  # print string
            #print("txt_path is %s"%txt_path)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results，打印识别目标的数量和结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]} {'s' * (n > 1)} , "  # add to string

                #thread.join()
                #if(accel_frame % fps_set == 0):
                # Write results on every frame
                for *xyxy, conf, cls in reversed(det):
                
                    if (0< xyxy[2] < 1280):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                            print("xywh  x : %d, y : %d"%(xywh[0],xywh[1]) )
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                        
                            ##print label x,y zuobiao 
                            
                            ##center of the bbox 
                            x_center = (xyxy[0] + xyxy[2]) / 2
                            y_center = (xyxy[1] + xyxy[3]) / 2
                            x_0 = int(x_center)
                            y_0 = int(y_center)

                            #print(" %s is  x: %d y: %d " %(label,x,y) )
                            if (0< x_0 < 1280):
                                ########################################################### #
                                #print 3d  x y z dis to images
                                '''             x1 = xyxy[0]        x2 = xyxy[2]
                                             -------------------------------------
                                y1 = xyxy[1] |     |-------------------|        
                                             |     |   person          |
                                             |     |                   |
                                y2 = xyxy[3] |     |-------------------|
                                             |     
                                '''
                                
                                #if(accel_frame % fps_set == 0):
                                x1 = xyxy[0]
                                x2 = xyxy[2]
                                y1 = xyxy[1]
                                y2 = xyxy[3]
                            
                                
                                bbox_h = y2-y1
                                bbox_w = x2-x1
                                
                                precision = 20 #precision of ceju 
                                set_step = 30 #(precision/8/0.1) #set range of detect
                                rad =  bbox_h/bbox_w
                                y_step = int(bbox_h / set_step)
                                x_step = int(y_step / rad)
                                
                                #precision = 50 #precision of ceju 
                                points_x      = np.zeros(precision) # x
                                points_y      = np.zeros(precision) # y
                                points_detect = [0] * precision
                                
                                count_x = 0
                                count_y = 0
                                index = 0
                                
                                if(True):
                                    t3 = time.time() # stereo time end
                                    thread.join()
                                    points_3d = thread.get_result()
                                    t4 = time.time() # stereo time end
                                    print(f'{s}Stereo Done. ({t4 - t3:.3f}s)')
                                
                                #迭代计算3D点云，直到精度符合
                                while( bool((0<x_0<1280) & (0<y_0<720))  ): #get dis point  "*" sreach
                                     count_x += 0.5
                                     x_0 += count_x*x_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ): # z < 12m (x_0 + 1, y_0)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break
                                     
                                     count_x += 0.5
                                     x_0 += count_x*x_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ): # z < 12m (x_0 + 1, y_0)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break

                                     count_y += 0.5
                                     y_0 += count_y*y_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ): # z < 12m  (x_0 + 1, y_0 + 1)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break
                                        
                                     count_y += 0.5
                                     y_0 += count_y*y_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ): # z < 12m  (x_0 + 1, y_0 + 1)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break
                                        
                                     count_x += 0.5
                                     x_0 -= count_x*x_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ): # z < 12m (x_0 - 1, y_0 + 1)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break
                                     
                                     count_x += 0.5
                                     x_0 -= count_x*x_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ): # z < 12m (x_0 - 1, y_0 + 1)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break
                                     
                                     count_y += 0.5
                                     y_0 -= count_y*y_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ):# z < 12m (x_0 - 1, y_0 - 1)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break
                                     
                                     count_y += 0.5
                                     y_0 -= count_y*y_step
                                     if((x1<x_0<x2)&(y1<y_0<y2)):
                                         if( 0 < points_3d[int(y_0), int(x_0), 2] < 12000 ):# z < 12m (x_0 - 1, y_0 - 1)
                                            dis = ( (points_3d[int(y_0), int(x_0), 0] ** 2 + points_3d[int(y_0), int(x_0), 1] ** 2 + points_3d[int(y_0), int(x_0), 2] **2) ** 0.5 ) / 1000
                                            points_x[index]      = x_0
                                            points_y[index]      = y_0
                                            points_detect[index] = dis
                                            #text_xy_00 = "."
                                            #cv2.putText(im0, text_xy_00, (int(x_0), int(y_0)) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                            index += 1
                                     if(index>=precision):
                                        break
                                
                                #求平均距离
                                dis_avg_new = []
                                #这一段是为了去除离群点
                                num = 0
                                for i in range(0,precision):
                                    min_detect = min(points_detect)
                                    max_detect = max(points_detect)
                                    level_detect = min_detect + (max_detect - min_detect)/4 #deep set 
                                    if ((points_detect[i]!=0)&(min_detect < points_detect[i] < level_detect) ):
                                        dis_avg_new.append(points_detect[i])
                                        num += 1

                                        #print()        
                                #print("num is %d"%len(dis_avg_new) )

                                # global dis_avg
                                if(num!=0):  
                                    index_end = 0      
                                    dis_avg = get_median(dis_avg_new) #取中位数
                                    index_end = points_detect.index(dis_avg)#中位数索引
                                else:
                                    dis_avg = 0
                                    #print("match error!!!")     
                                 


                                count2 = 0
                                # global x
                                x = (xyxy[0] + xyxy[2]) / 2
                                y = (xyxy[1] + xyxy[3]) / 2

                                while( bool((x<1280) & (y<720)) ):#out of index
                                    count2 += 1

                                    x += count2
                                    if(x >= 1280):
                                       x = 1278
                                       break                                      

                                    if( 0 < points_3d[int(y), int(x), 2] < 5000 ):
                                       break
                                   
                                    y += count2
                                    if(y >= 720):
                                       y = 718
                                       break

                                    if(0 < points_3d[int(y), int(x), 2] < 5000 ):
                                       break
                 

                                    count2 += 1
                                    x -= count2
                                    if( 0 < points_3d[int(y), int(x), 2] < 5000 ):
                                       break
                                    y -= count2
                                    if( 0 < points_3d[int(y), int(x), 2] < 5000 ):
                                        break


                        
                                if(x>=1280):
                                   x = 1278
                                   print("x is out of index!")
                                if(y>=720):
                                   y = 718
                                   print("y is out of index!")
                                                                   
                               
                               
                                '''             x1 = xyxy[0]        x2 = xyxy[2]
                                             -------------------------------------
                                y1 = xyxy[1] |     |-------------------|        
                                             |     |   person          |
                                             |     |                   |
                                y2 = xyxy[3] |     |-------------------|
                                             |     
                                '''
                                if (dis_avg!=0):## Add bbox to image
                                    # global label
                                    label = f'{names[int(cls)]}'
                                    im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                                    #print center x y
                                    x_end = points_x[index_end]
                                    y_end = points_y[index_end]
                                    #print("x_end = points_x[%f] = %f, y_end = points_y[%f] = %f"%(index_end,x_end,index_end,y_end))
                                    #x_center = (xyxy[0] + xyxy[2]) / 2
                                    #y_center = (xyxy[1] + xyxy[3]) / 2
                                    text_xy_0 = "*"
                                    cv2.putText(im0, text_xy_0, (int(x_end), int(y_end) ) ,  cv2.FONT_ITALIC, 1.2, (0,0,255), 3)
                                    
                                    print()
                                    print('点 (%d, %d) 的 %s 距离左摄像头的相对距离为 %0.2f m' %(x_center, y_center,label, dis_avg) )
                                    
                                    print('点 (%d, %d) 的三维坐标 (x:%.1fcm, y:%.1fcm, z:%.1fcm)' % (int(x), int(y), 
                                        points_3d[int(y), int(x), 0]/10, 
                                        points_3d[int(y), int(x), 1]/10, 
                                        points_3d[int(y), int(x), 2]/10) )

                                    text_dis_avg = "dis:%0.2fm" %dis_avg

                                    #only put dis on frame
                                    cv2.rectangle(im0,(int(x1+(x2-x1)),int(y1)),(int(x1+(x2-x1)+5+210),int(y1+40)),colors[int(cls)],-1);    
                                    cv2.putText(im0, text_dis_avg, (int(x1+(x2-x1)+5), int(y1+30)), cv2.FONT_ITALIC, 1.2, (255, 255, 255), 3)
                    
                    #将计算结果加载至判优函数
                    judge_priority.set_avg(round(dis_avg,3),int(x),float(x_end)/1000,float(y_end)/1000,label)
                voice_.set_avg()
                judge_priority.clear()
                    

            t5 = time_synchronized() # stereo time end
            # Print time (inference + NMS)
            print(f'{s}yolov5 Done. ({t2 - t1:.3f}s)')
            #print(threading.active_count())                              #获取已激活的线程数
            #print(threading.enumerate())    # see the thread list，查看所有线程信息，一个<_MainThread(...)> 带多个 <Thread(...)>


            if(accel_frame % fps_set  == 0):
                # Print time (inference + NMS +yolov5 + stereo)
                print(f'{s}yolov5+stereo Done. ({t5 - t1:.3f}s)')
            #print(threading.active_count())                              #获取已激活的线程数
            #print(threading.enumerate())    # see the thread list，查看所有线程信息，一个<_MainThread(...)> 带多个 <Thread(...)>
            
            #press "s" to talk for sttings
            #press "q" to stop the program.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                voice_.allow_voice_setting()
            elif key == ord("q"):
                 if save_txt or save_img:
                      s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                 
                 if view_img:
                      s = save_stream_path
                 print(f"Results saved to {s}")
                 print(f'All Done. ({time.time() - t0:.3f}s)')
                 vid_writer.release()  # release previous video writer
                 voice_.stop() # stop
                 exit()
                 #quit()

            # Stream results
            if view_img:
                if (dataset.mode == 'stream' ) & (flag == 0):
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                  
                    fourcc = 'mp4v'  # output video codec
                    fps = 24#vid_cap.get(cv2.CAP_PROP_FPS)
                    w = 2560#int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = 720#int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print("save_stream_dir is %s"%save_stream_dir)
                    print("save_stream_path is %s"%save_stream_path)
                    vid_writer = cv2.VideoWriter(save_stream_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    flag = 1

                vid_writer.write(im0)
                cv2.namedWindow("Webcam",cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Webcam",2560,720) ##创建一个名为Video01的窗口，设置窗口大小为 1280 * 480 与上一个设置的 0 有冲突
                cv2.moveWindow("Webcam",0,100) # left top
                cv2.imshow("Webcam", im0)
                #cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img :
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = 24#vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
                    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Video",1280,480) ##创建一个名为Video01的窗口，设置窗口大小为 1920 * 1080 与上一个设置的 0 有冲突
                    cv2.moveWindow("Video",0,0)# left top
                    cv2.imshow("Video", im0)
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                    
        print("frame %d is done!"%accel_frame)
        accel_frame += 1
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'All Done. ({time.time() - t0:.3f}s)')
    time.sleep(1)



def main():
    t1 = threading.Thread(target=detect)
    t2 = threading.Thread(target=voice_.voice_broadcast)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.55, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                main()
                strip_optimizer(opt.weights)              
        else:
            main()