#语音指令识别程序
#一级目录：1，设置；2，导航；3，关机
import pyttsx4
import speech_recognition as sr 
import time
import judge_priority
import re
from numpy import random

#中文语音合成包目录,后期可加入英语等其他语言
zh_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0"
#树结构设置选单
options_tree = {
    "音量": {
        "低": 0.7,"中": 1,"高": 1.3
    },
    "语速": {
        "慢": 210,"中": 230,"快": 250
    },
    "播报距离": {
        "近": 0.6,"中": 1,"远": 3
    },
    "播报间隔":{
        "短": 0,"中": 1,"长": 2
    },
    "退出": {}
}
#这是一个存储默认设置选项的字典
setting_options = {
    "音量": 1,
    "语速": 230,
    "播报距离": 1,
    "播报间隔": 1,
}
#这个字典存储个性化设置的默认设置，由于个性化设置是直接对数据进行修改，因此不需要设置树结构选单
personalise_setting = {
    "身高": 170,
    "体重": 55,
    "年龄": 33, #加入年龄选项是为了通过用户年龄判断遇到各种障碍可能发生的风险
}

################语音播报##################
def init_data():
    return {
        "weight" : 0,
        "dis_avg": 0,
        "x": 0,
        "label": 0
    }

data1 = init_data()
data2 = init_data()

def set_avg():
    global data1, data2
    data1, data2 = judge_priority.send_avgs()

#停止语音播报
stop_signal = 1
def stop():
    global stop_signal
    stop_signal = 0  
#允许语音设置
voice_setting = 0
def allow_voice_setting():
    global voice_setting
    voice_setting = 1

def voice_broadcast():
    def get_azimuth(x):
        if x < -600:
            return "左前方"
        elif x >= -600 and x <= 600:
            return "前方"
        else:
            return "右前方"
    engine_broadcast = pyttsx4.init()
    engine_broadcast.setProperty('voice', zh_voice_id)
    engine_broadcast.setProperty('rate', setting_options["语速"])
    engine_broadcast.setProperty('volume', setting_options["音量"])
    while stop_signal:
        global voice_setting 
        if voice_setting:
            settings()
            voice_setting = 0
            #我实在想不到还有什么别的办法能重启这个engine
            engine_broadcast = pyttsx4.init()
            engine_broadcast.setProperty('voice', zh_voice_id)
            engine_broadcast.setProperty('rate', setting_options["语速"])
            engine_broadcast.setProperty('volume', setting_options["音量"])
        x1, x2 = data1['x'], data2['x']
        dis_1, dis_2 = data1['dis_avg'], data2['dis_avg']
        label_1, label_2 = data1['label'], data2['label']
        azimuth1, azimuth2 = get_azimuth(x1), get_azimuth(x2)

        broadcast_sentence = [('请注意,%s距离您%0.1f米处有%s,%0.1f米处有%s','请注意,%s距离您%0.1f米处有%s,%s距离您%0.1f米处有%s','请注意,%s距离您%0.1f米处有%s'),
                              ('请留意，%s距离您%0.1f米处有%s，%0.1f米处有%s', '请留意，%s距离您%0.1f米处有%s，%s距离您%0.1f米处有%s', '请留意，%s距离您%0.1f米处有%s'),
                              ('请小心，%s距离您%0.1f米处有%s，%0.1f米处有%s', '请小心，%s距离您%0.1f米处有%s，%s距离您%0.1f米处有%s', '请小心，%s距离您%0.1f米处有%s'), 
                              ('警告，%s距离您%0.1f米处有%s，%0.1f米处有%s', '警告，%s距离您%0.1f米处有%s，%s距离您%0.1f米处有%s', '警告，%s距离您%0.1f米处有%s')
        ]
        sen = random.randint(0,3)
        if (dis_1 > 0.1 and dis_1 <= 10 and label_1 != 0):
            if (dis_2 > 0.1 and dis_2 <= 10 and label_2 != 0 and azimuth1 == azimuth2):
                engine_broadcast.say(broadcast_sentence[sen][0] % (azimuth1, dis_1, label_1, dis_2, label_2))
            elif (dis_2 > 0.1 and dis_2 <= 10 and label_2 != 0):
                engine_broadcast.say(broadcast_sentence[sen][1] % (azimuth1, dis_1, label_1, azimuth2, dis_2, label_2))
            else:
                engine_broadcast.say(broadcast_sentence[sen][2] % (azimuth1, dis_1, label_1))
            engine_broadcast.runAndWait()
            engine_broadcast.stop()
        
        time.sleep(1)

##############语音命令################## 

#这个函数用于返回语音指令或者错误信息
def takeCommand():
    recognizer = sr.Recognizer()   
    with sr.Microphone() as source:
        print("请说话...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) 
    try:
        text = recognizer.recognize_google(audio, language='zh-CN')
        print(f"指令为: {text}")
        return text
    except sr.UnknownValueError:
        print("无法识别语音，请重试。")
        return "无法识别语音，请重试。"
    except sr.RequestError as e:
        print(f"请求出错;{e}")
        return f"请求出错;{e}"
    
#这个函数用于将口语化表达转化为数字
def get_value(command):
    pattern_height_1 = r"(\d+)米(\d+)"
    pattern_weight_1 = r"(\d+)斤" 
    pattern_height_2 = r"(\d+)厘米"
    pattern_weight_2 = r"(\d+)公斤"
    pattern_age = r"(\d+)岁"

    matche_1 = re.findall(pattern_height_1, command)
    if matche_1 and int(matche_1[0][1])<10:
        return int(matche_1[0][0])*100 + int(matche_1[0][1])*10
    elif matche_1 and int(matche_1[0][1])>=10:
        return int(matche_1[0][0])*100 + int(matche_1[0][1])
    
    matche_2 = re.findall(pattern_weight_1, command)
    if matche_2:
        return int(matche_2[0])/2
    
    matche_3 = re.findall(pattern_height_2, command)
    if matche_3:
        return int(matche_3[0])
    
    matche_4 = re.findall(pattern_weight_2, command)
    if matche_4:
        return int(matche_4[0])
    
    matche_5 = re.findall(pattern_age, command)
    if matche_5:
        return int(matche_5[0])
    
    return int(command)
    

#这个函数用于设置
def settings():
    engine = pyttsx4.init()
    engine.setProperty('rate', setting_options["语速"])
    engine.setProperty('volume', setting_options["音量"])
    engine.setProperty('voice', zh_voice_id)
    engine.say("正在接收指令")
    engine.runAndWait()
    command = takeCommand()
    if "设置" in command:
        while True:
            engine.say("正在设置列表中,您可以选择")    
            for _ in options_tree.keys():
                engine.say(f"{_},")
            print("正在设置列表中")
            engine.runAndWait()
            # 设置内容
            setting_command = takeCommand()
            print(setting_command)
            if "退出" in setting_command:
                engine.say("是否确定退出？")    
                print("是否确定退出？")
                engine.runAndWait()
                command_ = takeCommand()
                if "确定" in command_ or "是" in command_:
                    engine.say("已退出")
                    print("已退出")
                    engine.runAndWait()
                    break
                else:
                    continue
            key_words = options_tree.items()
            for key, value in key_words:
                while key in setting_command:
                    engine.say(f"请说出您要设置的{key}级别,")
                    for _ in options_tree[key].keys():
                        engine.say(f"{_},")
                    print(f"请说出您要设置的{key}级别")
                    engine.runAndWait()
                    option = takeCommand()
                    print(option)
                    if option in value:
                        engine.say(f"已将{key}设置为{option}")
                        print(f"已将{key}设置为{option}")
                        engine.runAndWait()
                        #改变字典中的值，重启播报函数中的语音引擎
                        setting_options[key] = options_tree[value][option]
                        break
                    else:
                        engine.say(f"对不起，我不明白您的意思")
                        engine.runAndWait()
                        continue
            break


    elif "个性化" in command:
        while True:
            engine.say("正在个性化设置列表中,您可以设置您的")  
            for _ in personalise_setting.keys():
                engine.say(f"{_},")
            print("正在个性化设置列表中")
            engine.runAndWait()
            personalise_command = takeCommand()
            print(personalise_command)
            if "退出" in personalise_command:
                engine.say("是否确定退出？")    
                print("是否确定退出？")
                engine.runAndWait()
                command_ = takeCommand()
                if "确定" in command_ or "是" in command_:
                    engine.say("已退出")
                    print("已退出")
                    engine.runAndWait()
                    break
                else:
                    continue
            key_words = personalise_setting.items()
            for key, value in key_words:
                while key in personalise_command:
                    engine.say(f"请设置您的{key},")
                    print(f"请设置您的{key}")
                    engine.runAndWait()
                    option = get_value(takeCommand())
                    print(option)
                    if option:
                        engine.say(f"已将{key}设置为{option}")
                        print(f"已将{key}设置为{option}")
                        engine.runAndWait()
                        #改变字典中的值,并且将其传入判优函数
                        personalise_setting[key] = option
                        judge_priority.set_personalise(personalise_setting)
                        break
                    else:
                        engine.say(f"对不起，我不明白您的意思")
                        engine.runAndWait()
                        continue
            break


    elif "导航" in command:
        engine.say("请问您要去哪里?")
        destination = takeCommand()
        print(f"导航至{destination}")
        #navigation()


    elif "关机" in command:
        engine.say("是否确定关机?")
        command_ = takeCommand()
        while True:
            if "确定" in command_ or "是的" in command_:
                engine.say("正在关机...")
                #shutdown()
            else:
                break
    elif "无法识别语音，请重试。" == command or "请求出错;" in command:
        engine.say(command)
        engine.runAndWait()

    

    engine.stop()
    print ("done")

