class Joint:
    def __init__(self, init_angle,  uplimit: float, downlimit: float):
        self.limit_up = uplimit
        self.limit_down = downlimit
        self.angle = init_angle
        self.tau = 0

    @property
    def Angle(self):
        return self.angle
    
    @Angle.setter
    def Angle(self, value):
        self.angle = value