"""
时间机器：根据目标线速度，计算出完整运动周期所需要的时间，作为时间机器的主周期。

"""
import time

class TimeMachine:
    def __init__(self):
        self.period = 1000 #ms 
        self.__tick = 0
        self.__pstart = time.time()

    def get_leg_phase(self, period):
        return 