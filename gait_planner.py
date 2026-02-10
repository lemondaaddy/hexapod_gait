from matplotlib.axes import Axes
import numpy as np
from enum import Enum
from arm import Arm
import time
import hexa_define as Hexa
from typing import Tuple
from robot import HexaRobot
class GAIT:
    TRIPLE = "tiple"
    WAVE = "wave"
    RIPLE = "riple"

    RIPLE_SQUENCE = {
        0: 0,
        1: 4,
        2: 2,
        3: 5,
        4: 1,
        5: 3
    }

    WAVE_SQUENCE = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }

    TRIPLE_SQUENCE = {
        0: 0,
        1: 1,
        2: 0,
        3: 1,
        4: 0,
        5: 1
    }

    def __init__(self, gait_type: str):
        self.gait_type = gait_type
        if gait_type == self.WAVE:
            self.squence = self.WAVE_SQUENCE
        elif gait_type == self.RIPLE:
            self.squence = self.RIPLE_SQUENCE
        else:
            self.squence = self.TRIPLE_SQUENCE
        #self.squence = self.WAVE_SQUENCE if gait_type == self.WAVE else self.TRIPLE_SQUENCE if gait_type == self.TRIPLE else self.RIPLE_SQUENCE
        self.phase_count = 6 if (gait_type == self.WAVE or gait_type == self.RIPLE) else 2
    @property
    def Squence(self):
        return self.squence
    
    @property
    def PhaseCount(self):
        return self.phase_count

def solve_quintic_for_T(q0, qf, v0, vf, a0, af, T):
    """返回五次多项式系数"""
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3]
    ])
    b = np.array([q0, v0, a0, qf, vf, af])
    return np.linalg.solve(A, b)




class StepPhase:
    def __init__(self, id, init_speed: Tuple[float, float], acc: Tuple[float, float], stride: Tuple[float, float], period_stride: Tuple[float, float]):
        self.id = id
        self.init_speed = np.array(init_speed, dtype=float) 
        self.acc = np.array(acc, dtype=float)  # 加速度
        self.stride = np.array(stride, dtype=float)  # 步幅
        self.final_speed = np.zeros(2, dtype=float)
        self.time_span = 0.0
        self.time_elapsed = 0.0

        # 如果提供了运动参数，则解算阶段时间与末速度
     
        t_axis = np.zeros(2, dtype=float)
        for i in range(2):
            v0 = self.init_speed[i]
            a = self.acc[i]
            s = self.stride[i]

            if np.isclose(a, 0.0):
                if np.isclose(v0, 0.0) or np.isclose(s, 0.0):
                    t_axis[i] = 0.0
                else:
                    t_axis[i] = s / v0
            else:
                A = 0.5 * a
                B = v0
                C = -s
                disc = B * B - 4 * A * C
                disc = max(disc, 0.0)
                sqrt_disc = np.sqrt(disc)
                t0 = (-B + sqrt_disc) / (2 * A)
                t1 = (-B - sqrt_disc) / (2 * A)
                candidates = [t for t in (t0, t1) if t > 0]
                t_axis[i] = min(candidates) if candidates else 0.0

            self.time_span = max(t_axis)
            self.final_speed = self.init_speed + self.acc * self.time_span
            self.swing_quintic = solve_quintic_for_T(-period_stride[0]/2, period_stride[0]/2, -self.init_speed[0], -self.final_speed[0], 0, 0, self.time_span)

    def step(self, dt:float):
        if self.time_elapsed + dt > self.time_span:
            self.time_elapsed = self.time_span
            remain = self.time_elapsed + dt - self.time_span
        else:
            self.time_elapsed += dt
            remain = 0.0
        t = self.time_elapsed
        self.swing_pos = self.swing_quintic[0] + self.swing_quintic[1] * t + self.swing_quintic[2] * t ** 2 + self.swing_quintic[3] * t ** 3 + self.swing_quintic[4] * t ** 4 + self.swing_quintic[5] * t ** 5
        self.swing_vel = self.swing_quintic[1] + self.swing_quintic[2] * 2 * t + self.swing_quintic[3] * 3 * t ** 2 + self.swing_quintic[4] * 4 * t ** 3 + self.swing_quintic[5] * 5 * t ** 4
        
        self.stand_pos = self.init_speed * self.time_elapsed + 0.5 * self.acc * self.time_elapsed ** 2
        self.stand_vel = self.init_speed + self.acc * self.time_elapsed

        return remain

    @property
    def swing_pos_vel(self):

        return self.swing_pos, self.swing_vel
    
    @property
    def stand_pos_vel(self):
        return self.stand_pos, self.stand_vel
    



class GaitPeriod:
    def __init__(self, t:float, init_vel: Tuple[float, float], target_vel: Tuple[float, float], acc: Tuple[float, float], stride: Tuple[float, float],max_acc: Tuple[float, float] = (0.8, 0.6)):
        self.init_v = np.array(init_vel) # 初始速度

        self.final_v = np.array(target_vel) # 末速度
        self.stride = np.array(stride) # 步幅
        self.period = 0 # 周期时间
        self.acc = np.array(acc) # 加速度
        self.max_acc = np.array(max_acc)
        self.started_at = t
        self.time_eslapsed = 0
        self.cur_v = np.array(init_vel) # 当前速度
        self.cur_pos = np.zeros(2   , dtype=np.float32) # 当前位移
        self._calc_params()
        self.phase = None

    def step(self, dt:float) -> Tuple[np.ndarray, np.ndarray, float]:

        if self.phase is None:
            self.phase = StepPhase(0, self.init_v.copy(), self.acc.copy(), self.stride.copy()/6, self.stride.copy())
        
        real_dt = self.period - self.time_eslapsed if (self.time_eslapsed + dt) > self.period else dt
        new_vel = self.cur_v + self.acc * real_dt

        #print(f"GaitPeriod step: dt={dt:.4f}, real_dt={real_dt:.4f}, new_vel={new_vel}, time_eslapsed={self.time_eslapsed:.4f}, period={self.period:.4f} ")
    
        delta_pos = (self.cur_v + new_vel) * real_dt / 2
        self.cur_v = new_vel.copy()
        self.cur_pos += delta_pos
        self.time_eslapsed += real_dt

        phase_remain = self.phase.step(real_dt)
        if phase_remain > 1e-5:
            self.phase = StepPhase(self.phase.id + 1, self.phase.final_speed.copy(), self.acc.copy(), self.stride.copy()/6, self.stride.copy())
            self.phase.step(phase_remain)

        return delta_pos.copy(), self.cur_v.copy(),  dt - real_dt
    
    def _calc_params(self):
        acc = (self.final_v ** 2 - self.init_v ** 2) / (2 * self.stride)
        print(f"GaitPeriod calc_params: raw acc={acc}, max_acc={self.max_acc}")
        self.acc = np.zeros(2, dtype=np.float32)

        self.acc[0] = acc[0] if (np.abs(acc[0]) <= self.max_acc[0]) else (self.max_acc[0] if acc[0] > 0 else -self.max_acc[0])
        self.acc[1] = acc[1] if (np.abs(acc[1]) <= self.max_acc[1]) else (self.max_acc[1] if acc[1] > 0 else -self.max_acc[1])
        print(f"GaitPeriod calc_params: raw acc={self.acc}")
        t_axis = np.zeros(2, dtype=np.float32)
        for i in range(2):
            if np.isclose(self.acc[i], 0.0):
                t_axis[i] = self.stride[i] / self.init_v[i] if not np.isclose(self.init_v[i], 0.0) else 0.0
            else:
                disc = self.init_v[i] ** 2 + 2 * self.acc[i] * self.stride[i]
                disc = max(disc, 0.0)
                sqrt_disc = np.sqrt(disc)
                t0 = (-self.init_v[i] + sqrt_disc) / self.acc[i]
                t1 = (-self.init_v[i] - sqrt_disc) / self.acc[i]
                candidates = [t for t in (t0, t1) if t > 0]
                t_axis[i] = max(candidates) if candidates else 0.0

        self.period = max(t_axis)
        print(f"GaitPeriod calc_params: init_v={self.init_v}, final_v={self.final_v}, stride={self.stride}, acc={self.acc}, period={self.period}")
        # 根据同步后的周期，反推新的末速度与对应轴步幅，保持加速度约束
        self.final_v = self.init_v + self.acc * self.period
        new_stride = np.zeros_like(self.stride)
        for i in range(2):
            if np.isclose(self.acc[i], 0.0):
                new_stride[i] = self.init_v[i] * self.period
            else:
                new_stride[i] = (self.final_v[i] ** 2 - self.init_v[i] ** 2) / (2 * self.acc[i])
        self.stride = new_stride

        if hasattr(self, "period_start_at"):
            self.period_end_at = self.period_start_at + self.period

        return

class MoveGaitPlanner:
    SWING_Z_SRIDE = 0.00
    PERIOD_STRIDE = (0.1, 0.02)
    MAX_ACC = (0.1, 0.05)

    BASE_Z = -0.08 # 重心离地高度  # LOW 0.05 HIGH 0.05
    BASE_RADUIS = 0.2 #落足点半径基准 

    FRONT_ARM_OFFSET_DEGREE = 60
    BACK_ARM_OFFSET_DEGREE = -35
    MIDDLE_ARM_OFFSET_DEGREE = 15

    def __init__(self, robot:HexaRobot):
        self.gait = GAIT("wave")
        self.pos = np.zeros(2)
        self.init_foot_pos:dict[int, np.array[float, float, float]] = dict()
        self.robot = robot
        self.vel = np.zeros(2)
        self.period = None
        self.tick_time = 0
        for arm in self.robot.arms.values():
            self.set_arm_init_pos(arm)

    @property
    def body_vel(self):
        return self.vel
    
    @property
    def body_pos(self):
        return self.pos
        
    @property
    def stride(self):
        return self.period.stride
    
    def set_arm_init_pos(self, arm: Arm):
        print(f"Setting init pos for arm {arm.id}")
        x_stride = self.PERIOD_STRIDE[0]
        if arm.id == Hexa.ARM_LF or arm.id == Hexa.ARM_RF:
            x_offset = self.BASE_RADUIS * 1.2 * np.sin(np.deg2rad(self.FRONT_ARM_OFFSET_DEGREE))
            y_offset = self.BASE_RADUIS * 1.2 * np.cos(np.deg2rad(self.FRONT_ARM_OFFSET_DEGREE)) * arm.yDir
           

        elif arm.id == Hexa.ARM_RM or arm.id == Hexa.ARM_LM:
            x_offset = self.BASE_RADUIS *  np.sin(np.deg2rad(self.MIDDLE_ARM_OFFSET_DEGREE))
            y_offset = self.BASE_RADUIS * np.cos(np.deg2rad(self.MIDDLE_ARM_OFFSET_DEGREE)) * arm.yDir
            #arm.set_foot_coor(np.array([x_offset, y_offset, self.BASE_Z]))
        elif arm.id == Hexa.ARM_LL or arm.id == Hexa.ARM_RL:
            x_offset = self.BASE_RADUIS * 1.1 * np.sin(np.deg2rad(self.BACK_ARM_OFFSET_DEGREE))
            y_offset = self.BASE_RADUIS * 1.1 * np.cos(np.deg2rad(self.BACK_ARM_OFFSET_DEGREE)) * arm.yDir

        if arm.id == Hexa.ARM_LF:
            self.init_foot_pos[arm.id] = np.array([x_offset -x_stride / 2, y_offset, self.BASE_Z])
        elif arm.id == Hexa.ARM_LM:
            self.init_foot_pos[arm.id] = np.array([x_offset - x_stride / 2 + x_stride/5, y_offset, self.BASE_Z])
        elif arm.id == Hexa.ARM_LL:
            self.init_foot_pos[arm.id] = np.array([x_offset - x_stride / 2 + x_stride/5 * 2, y_offset, self.BASE_Z])
        elif arm.id == Hexa.ARM_RL:
            self.init_foot_pos[arm.id] = np.array([x_offset - x_stride / 2 + x_stride/5 * 3, y_offset, self.BASE_Z])
        elif arm.id == Hexa.ARM_RM:
            self.init_foot_pos[arm.id] = np.array([x_offset - x_stride / 2 + x_stride/5 * 4, y_offset, self.BASE_Z])
        elif arm.id == Hexa.ARM_RF:
            self.init_foot_pos[arm.id] = np.array([x_offset - x_stride / 2 + x_stride/5 * 5, y_offset, self.BASE_Z])
   
        print(f"Arm {arm.id} init foot pos: {self.init_foot_pos[arm.id]}")
        arm.set_foot_coor(self.init_foot_pos[arm.id])
    
    

    def step(self, dt:float, target_vel:Tuple[float, float]):
        
        if self.period is None:
            self.period = GaitPeriod(self.tick_time, (0, 0), target_vel, self.MAX_ACC, self.PERIOD_STRIDE)

        pos, vel, remain = self.period.step(dt)
        self.vel = vel
        self.pos += pos
        self.tick_time += dt - remain

        if remain > 1e-5:
            self.period = GaitPeriod(self.tick_time, self.vel, target_vel, self.MAX_ACC, self.PERIOD_STRIDE)
            pos, vel, _ = self.period.step(remain)
            self.vel = vel
            self.pos += pos

            self.tick_time += remain




class body_speed_line:
    def __init__(self, ax:Axes):
        self.ax = ax
        self.ax.grid(True)
        self.line_p, = ax.plot([], [], 'r-',  label='Position')
        self.line_v, = ax.plot([], [], 'b-', label='Velocity')
        self.ax.legend()
        self.v = []
        self.p = []
        self.t = []
    def update(self, pos:Tuple[float, float], vel:Tuple[float, float], time:float):
        self.p.append(pos)
        self.v.append(vel)
        self.t.append(time)
        #self.line_p.set_data(self.t, np.array(self.p)[:,0])
        #self.line_v.set_data(self.t, np.array(self.v)[:,0])
        #self.ax.relim()
        #self.ax.autoscale_view()

    def show(self):
        self.line_p.set_data(self.t, np.array(self.p)[:,0])
        self.line_v.set_data(self.t, np.array(self.v)[:,0])
        self.ax.relim()
        self.ax.autoscale_view()

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from robot import HexaRobot, Arm
    
    plt.ion()
    fig = plt.figure(figsize=(20, 10))
    body_ax = fig.add_subplot(121)

    body_line = body_speed_line(body_ax)

    robot = HexaRobot()
    times = []
    gait = MoveGaitPlanner(robot)

    t = time.time()
    t_start = t
    for i in range(0, 5000):
        time.sleep(0.0005)
        dt = time.time() - t
        gait.step(dt, (0.05, 0))
        t += dt

        times.append(dt)
        body_line.update(gait.body_pos.copy(), gait.body_vel.copy(), (t - t_start) * 1000)
        
    #plt.pause(0.0001)

    body_line.show()
    plt.ioff()
    plt.show()
    print(np.max(times))