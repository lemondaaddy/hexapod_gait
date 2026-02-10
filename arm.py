import numpy as np
from joint import Joint
from FKIK import fk, ik, pos_err

class Arm:
    def __init__(self, id, base_coor: tuple[float, float, float], joints_limit: tuple[any, any, any], length: tuple[float, float, float], yDir = 1):
        self.id = id
        self.base_coor = np.array(base_coor)
        self.L0 = length[0]
        self.L1 = length[1]
        self.L2 = length[2]
        self._pos = 0
        self._v = 0
        self.joints:list[Joint] = []
        
        for i in range(3):
            self.joints.append(Joint(0, joints_limit[i][0], joints_limit[i][1]))

        self.yDir = yDir
        self.xFactor = -1 if yDir == 1 else 1

    @property
    def Position(self):
        return self._pos
    
    @Position.setter
    def Position(self, value):
        self._pos = value 

    @property
    def Vel(self):
        return self._v
    
    @Vel.setter
    def Vel(self, value):
        self._v = value 

    @property
    def A0(self):
        return self.joints[0].Angle 
    
    @property
    def A1(self):
        return self.joints[1].Angle
    
    @property
    def A2(self):
        return self.joints[2].Angle 
        
    @A0.setter
    def A0(self, value):
        self.joints[0].Angle = value

    @A1.setter
    def A1(self, value):
        self.joints[1].Angle = value

    @A2.setter
    def A2(self, value):
        self.joints[2].Angle = value

    def FK(self, A0, A1, A2, L0, L1, L2):
        c1, s1   = np.cos(A1), np.sin(A1)
        c12, s12 = np.cos(A1 + A2), np.sin(A1 + A2)
        S = L0 + L1*c1 + L2*c12
        x = np.sin(A0) * S
        y = np.cos(A0) * S * self.yDir
        z =  L1*s1 + L2*s12
        return np.array([x, y, z])
   
    def set_foot_coor(self, coor_body:np.array):
        coor = coor_body - self.base_coor
        q_J0, q_J1, q_J2 = ik(coor[0], coor[1]*self.yDir, coor[2], self.L0, self.L1, self.L2)
        
        A = np.array([q_J0, q_J1, q_J2])
        self.A0 = A[0] * self. yDir
        self.A1 = A[1] 
        self.A2 = A[2] 

    def get_foot_coor(self):
        _, _, _, foot_coor = self.get_joints_coor()
        return foot_coor
    def get_joints_coor(self):
        theta1, theta2, theta3 = self.A0, self.A1, self.A2

        theta1 *= self.xFactor

        # Coxa关节末端
        x1 = self.base_coor[0] + self.L0 * np.sin(theta1)

        y1 = self.base_coor[1] + self.L0 * np.cos(theta1) * self.yDir
        z1 = self.base_coor[2]

        # Femur关节末端
        x2 = x1 + self.L1 * np.cos(theta2) * np.sin(theta1)
        y2 = y1 + self.L1 * np.cos(theta2) * np.cos(theta1) * self.yDir
        z2 = z1 + self.L1 * np.sin(theta2)

        # Tibia关节末端
        x3 = x2 + self.L2 * np.cos(theta2 + theta3) * np.sin(theta1)
        y3 = y2 + self.L2 * np.cos(theta2 + theta3) * np.cos(theta1) * self.yDir
        z3 = z2 + self.L2 * np.sin(theta2 + theta3)



        return self.base_coor, (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)