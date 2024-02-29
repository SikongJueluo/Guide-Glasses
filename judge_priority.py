import heapq
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
#个性化使用者身高体重判定碰撞箱
def set_personalise(UserProfile):
    '''
    这里将用户的身高、体重、年龄传入，用于判断碰撞体积
    主要判断依据为体重和身高
    年龄用于判断用户可能发生的风险
    '''
    return


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
def set_avg(dis_avg,x,x_avg,y_avg,label):
    print(x_avg,y_avg)
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
  

