import numpy as np
from arm import Arm
import hexa_define as Hexa
from planner import HexapodGaitPlanner

class HexaRobot:

    def __init__(self):
        #self.body_state = {"pos_world": np.zeros(2), "vel_world": np.zeros(2), "acc_world": np.zeros(2)}

        self.arms: dict[int, Arm] = {}
        for k in Hexa.ARM_LENGTH.keys():
            self.arms[k] = Arm(
                id=k,
                base_coor=Hexa.ARM_BASE_COOR.get(k),
                joints_limit=Hexa.ARM_JOINT_LIMIT.get(k),
                length=Hexa.ARM_LENGTH.get(k),
                yDir=1 if k in [Hexa.ARM_LF, Hexa.ARM_LM, Hexa.ARM_LL] else -1,
            )

    def get_plot_links(self):
        lines = []
        body_line = []

        for key, arm in self.arms.items():
            body_line.append(arm.base_coor)
            arm_line = arm.get_joints_coor()
            lines.append(arm_line)

        body_line.append(self.arms[0].base_coor)

        lines.append(body_line)
        return lines



