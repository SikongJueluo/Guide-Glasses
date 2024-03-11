import heapq
import numpy as np
#传入参数改为坐标而不是距离
#加入碰撞体积，当左侧和右侧物体进入碰撞体积后才进行播报
#元素的第一个值为权重，后两个值分别为播报阈值的x、y坐标
object_weight = {
    #"person": 10, "cell phone": 0,  # for only testing
    "行人": (10, 1, 3), "汽车": (10, 2, 10), "公交车": (10, 3, 10), "摩托车": (10, 1, 3), "自行车": (10, 1, 3), #多数情况可能发生移动
    "交通灯": (10, 1, 3), "斑马线": (10, 1, 3), "树": (5, 2, 5), "楼梯": (8, 1, 3), "路灯": (5, 1, 3), "指示牌": (7, 1, 3), #几乎不会发生移动
    "垃圾桶": (4, 1, 2), "电线杆": (4, 1, 3), "躺椅": (4, 1, 2), "灌木": (4, 2, 4), "消防桩": (6, 1, 2),
    "狗": (6, 1, 2), "猫": (6, 1, 2)
}

#通过相对坐标判断是否进入碰撞体积,行人、动物等目标的y轴更大，因为其更有可能发生移动
#个性化使用者身高体重判定碰撞箱,老年人提高车辆等目标的敏感度
weight_addition = 0
def set_personalise(user_info):
    global weight_addition
    BMI = user_info["体重"] / (user_info["身高"] / 100) ** 2
    if BMI < 18.5:
        weight_addition = 0
    elif BMI < 24:
        weight_addition = 0.3
    elif BMI < 28: 
        weight_addition = 0.5
    else:
        weight_addition = 1
    #处理年龄,对老年人提高移动目标的敏感度
    if user_info["年龄"] > 60:
        object_weight.update({
            "行人": (10, 1, 5),"狗": (6, 1, 5),"猫": (6, 1, 5),
            "自行车": (10, 1, 5),"摩托车": (10, 1, 5),"汽车": (10, 2, 15),
            "公交车": (10, 3, 15)
        })

#优先队列，用于储存对象判断优先级
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def clear(self):
        self._queue = []
        self._index = 0

pq = PriorityQueue()

def clear():
    pq.clear()

#输入至判优函数,弹出一个或两个对象
def set_avg(dis_avg,x,x_avg,y_avg,label,indexID): 
    '''
    输入参数分别为平均距离，输出画面中心点x坐标，空间坐标系x、y坐标，对象标签，跟踪器ID 
    无return，函数处理结果为将对象信息加入优先队列
    '''
    try:
        if label in object_weight:
            label_weight = object_weight[label][0]
            necessity = 1 if x_avg<object_weight[label][1] or y_avg<object_weight[label][2] else 0
        else:
            label_weight = 0 
            necessity = 0
    except TypeError:
        print("label 或 object_weight 的类型不正确")
    except Exception as e:
        print(f"发生了未知错误: {e}")
    position_weight = 10 - dis_avg if dis_avg != 0 else 0
    direction_weight = 5  if abs(x)<600 else 0
    weight = position_weight + label_weight + direction_weight
    object = {
        "weight" : weight,
        "dis_avg": dis_avg,
        "x": x,
        "label": label,
    }
    if necessity:
        pq.push(object, object['weight'])


#输出至语音函数
def send_avgs():
    empty = {
            "weight" : 0,
            "dis_avg": 0,
            "x": 0,
            "label": 0
        }
    if len(pq._queue) == 0:
        return empty,empty
    elif len(pq._queue) >= 2:
        object1 = pq.pop()
        object2 = pq.pop()
        return object1,object2     
    else:
        object = pq.pop()
        return object,empty 
    
#下面是一些想法
  
#这一段用来拟合对象移动路线，根据对象的移动路径调整权重
#因为汽车等目标的播报距离较长，通过这个函数可以判断是否可以调整权重减少播报频率
# 
    '''
    这一段分为当用户静止时，当用户移动时
    拟合线性函数(当用户前行，且目标向左右行驶)
    '''
    #当目标相对静止
    #当目标相对向前移动
    #当目标相对向左、右移动
    #通过静止目标，如消防栓等判断用户是否移动
'''
载入对象位置参数： 只需要载入可能发生移动的对象
使用一个列表存储所有的xy坐标当样本足够时，使用线性回归拟合出移动路线
当预测到 如人，车，自行车向用户前方移动时，增加权重
'''
#这一段用来判断是否可能发生碰撞

#这个列表用于存储当前帧出现的所有对象
frame_objects = []
class object_tracking:
    def __init__(self):#创建对象
        self.id_coordinates = {}
    def set_route(x,y,id):#更新目标状态，发送是否需要broadcast
        '''
        三个参数分别为x，y坐标，跟踪器目标id
        ''' 
        pass
    def if_moving(self):#判断是否为移动目标，判断是否正在移动
        '''if id == "行人" or "狗" or "猫" or "自行车" or "摩托车" or "汽车" or "公交车":
            if 
        return 0'''
        pass
    def if_collision():#判断是否发生碰撞
        pass 