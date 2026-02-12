from matplotlib.axes import Axes
import numpy as np
from enum import Enum
from arm import Arm
import time
import hexa_define as Hexa
from typing import Tuple
from robot import HexaRobot
import builtins
from datetime import datetime


def print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    builtins.print(f"[{timestamp}]", *args, **kwargs)


class GAIT:
    TRIPLE = "tiple"
    WAVE = "wave"
    RIPLE = "riple"

    RIPLE_SEQUENCE = {
        0: 0,
        1: 4,
        2: 2,
        3: 5,
        4: 1,
        5: 3
    }

    WAVE_SEQUENCE = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }

    TRIPLE_SEQUENCE = {
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
            self.squence = self.WAVE_SEQUENCE
        elif gait_type == self.RIPLE:
            self.squence = self.RIPLE_SEQUENCE
        else:
            self.squence = self.TRIPLE_SEQUENCE
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



"""
class StepPhase:
    def __init__(self, id, init_speed: Tuple[float, float], final_speed: Tuple[float, float],  period_stride: Tuple[float, float]):
        self.id = id
        self.init_speed = np.array(init_speed, dtype=float) 
        self.final_speed = np.array(final_speed, dtype=float)
        self.time_span = 0.0
        self.time_elapsed = 0.0
        self.swing_pos = np.zeros(3, dtype=float)
        self.swing_vel = np.zeros(3, dtype=float)
        self.stand_pos = np.zeros(3, dtype=float)
        self.stand_vel = np.zeros(3, dtype=float)
        # 如果提供了运动参数，则解算阶段时间与末速度
     
        print(f"StepPhase init: id={id}, init_speed={self.init_speed}, final_speed={self.final_speed}, period_stride={period_stride}")
        self.swing_quintic = solve_quintic_for_T(-period_stride[0]/2, period_stride[0]/2, -self.init_speed[0], -self.final_speed[0], 0, 0, self.time_span)


    def step(self, dt:float):
        print(f"StepPhase step: id={self.id}, time_elapsed={self.time_elapsed:.4f}, dt={dt:.4f}, time_span={self.time_span:.4f}")
        if self.time_elapsed + dt > self.time_span:
            self.time_elapsed = self.time_span
            remain = self.time_elapsed + dt - self.time_span
        else:
            self.time_elapsed += dt
            remain = 0.0
            
        t = self.time_elapsed

        new_sw_pos_x = self.swing_quintic[0] + self.swing_quintic[1] * t + self.swing_quintic[2] * t ** 2 + self.swing_quintic[3] * t ** 3 + self.swing_quintic[4] * t ** 4 + self.swing_quintic[5] * t ** 5
        new_sw_vel_x = self.swing_quintic[1] + self.swing_quintic[2] * 2 * t + self.swing_quintic[3] * 3 * t ** 2 + self.swing_quintic[4] * 4 * t ** 3 + self.swing_quintic[5] * 5 * t ** 4
        
        new_sw_pos = np.array([new_sw_pos_x, 0, 0])
        new_sw_vel = np.array([new_sw_vel_x, 0, 0])

        delta_sw_pos = new_sw_pos - self.swing_pos 

        self.swing_pos = new_sw_pos
        self.swing_vel = new_sw_vel

        new_st_pos_xy = self.init_speed * self.time_elapsed + 0.5 * self.acc * self.time_elapsed ** 2
        new_st_vel_xy = self.init_speed + self.acc * self.time_elapsed

        new_st_pos = np.array([new_st_pos_xy[0], new_st_pos_xy[1], 0])
        new_st_vel = np.array([new_st_vel_xy[0], new_st_vel_xy[1], 0])

        delta_st_pos = new_st_pos - self.stand_pos

        self.stand_pos = new_st_pos
        self.stand_vel = new_st_vel
     

        return delta_sw_pos, delta_st_pos,  remain

    @property
    def swing_pos_vel(self):

        return self.swing_pos.copy(), self.swing_vel.copy()
    
    @property
    def stand_pos_vel(self):
        return self.stand_pos.copy(), self.stand_vel.copy()
    
"""
class StepPhase:
    def __init__(self, id, init_speed: Tuple[float, float], final_speed: Tuple[float, float],  period_stride: Tuple[float, float], timestart:float, timespan:float ):
        self.id = id
        self.init_speed = np.array(init_speed, dtype=float) 
        self.final_speed = np.array(final_speed, dtype=float)
        self.time_span = timespan
        self.time_start = timestart
        print(f'id: {self.id}, timespan {timespan}, phase_time_start:{self.time_start} init: {init_speed}, final: {final_speed}')
        self.swing_quintic_x = solve_quintic_for_T(-period_stride[0]/2, period_stride[0]/2, -self.init_speed[0], -self.final_speed[0], 0, 0, self.time_span)
        self.swing_quintic_y = solve_quintic_for_T(-period_stride[1]/2, period_stride[1]/2, -self.init_speed[1], -self.final_speed[1], 0, 0, self.time_span)

    def swing_pv(self, phase_time:float) -> Tuple[np.ndarray, np.ndarray]:
        t = phase_time
        #print(f't = {t}')
        swing_pos_x = self.swing_quintic_x[0] + self.swing_quintic_x[1] * t + self.swing_quintic_x[2] * t ** 2 + self.swing_quintic_x[3] * t ** 3 + self.swing_quintic_x[4] * t ** 4 + self.swing_quintic_x[5] * t ** 5
        swing_vel_x = self.swing_quintic_x[1] + self.swing_quintic_x[2] * 2 * t + self.swing_quintic_x[3] * 3 * t ** 2 + self.swing_quintic_x[4] * 4 * t ** 3 + self.swing_quintic_x[5] * 5 * t ** 4

        swing_pos_y = self.swing_quintic_y[0] + self.swing_quintic_y[1] * t + self.swing_quintic_y[2] * t ** 2 + self.swing_quintic_y[3] * t ** 3 + self.swing_quintic_y[4] * t ** 4 + self.swing_quintic_y[5] * t ** 5
        swing_vel_y = self.swing_quintic_y[1] + self.swing_quintic_y[2] * 2 * t + self.swing_quintic_y[3] * 3 * t ** 2 + self.swing_quintic_y[4] * 4 * t ** 3 + self.swing_quintic_y[5] * 5 * t ** 4

        swing_pos = np.array([swing_pos_x, swing_pos_y, 0])
        swing_vel = np.array([swing_vel_x, swing_vel_y, 0])

        return swing_pos, swing_vel

class GaitPeriod:
    def __init__(self, robot:HexaRobot, t:float, init_vel: Tuple[float, float], target_vel: Tuple[float, float],stride: Tuple[float, float],max_acc: Tuple[float, float] = (0.8, 0.6)):
        self.robot = robot
        self.init_v = np.array(init_vel) # 初始速度
        self.target_v = np.array(target_vel) # 目标速度
        self.final_v = np.array(target_vel) # 末速度
        self.stride = np.array(stride) # 步幅
        self.period = 0 # 周期时间
        self.acc = np.zeros(2, dtype=float) # 加速度
        self.max_acc = np.array(max_acc)
        self.started_at = t
        self.time_eslapsed = 0
        self.cur_v = np.array(init_vel) # 当前速度
        self.cur_pos = np.zeros(2   , dtype=np.float32) # 当前位移
        self._calc_params()
        self.phase = None
        self.phase_array = []

    def next_phase(self, phase_id:int) :

        # 计算该相位的起始位置和结束位置（基于距离划分）
        phase_start_pos = phase_id * self.stride / 6.0
        phase_end_pos = (phase_id + 1) * self.stride / 6.0
        
        # 计算相位起始时间：从周期0时刻（t=0）运动到phase起始位置所需时间
        # 给定：周期初速度(self.init_v)、周期加速度(self.acc)、相位起始位置(phase_start_pos)
        # 求解：s = v0*t + 0.5*a*t^2，得到时间t
        phase_time_start = 0.0
        if not np.isclose(self.acc[0], 0.0):
            # 有加速度：0.5*a*t^2 + v0*t - s = 0
            # 使用求根公式：t = (-v0 ± sqrt(v0^2 + 2*a*s)) / a
            disc = self.init_v[0]**2 + 2 * self.acc[0] * phase_start_pos[0]
            disc = max(disc, 0.0)
            sqrt_disc = np.sqrt(disc)
            t0 = (-self.init_v[0] + sqrt_disc) / self.acc[0]
            t1 = (-self.init_v[0] - sqrt_disc) / self.acc[0]
            candidates = [t for t in (t0, t1) if t >= 0]
            phase_time_start = min(candidates) if candidates else 0.0
        elif not np.isclose(self.init_v[0], 0.0):
            # 无加速度：匀速运动 t = s / v0
            phase_time_start = phase_start_pos[0] / self.init_v[0]
        
        # 计算相位结束时间：从周期0时刻运动到phase结束位置所需时间
        phase_time_end = 0.0
        if not np.isclose(self.acc[0], 0.0):
            disc = self.init_v[0]**2 + 2 * self.acc[0] * phase_end_pos[0]
            disc = max(disc, 0.0)
            sqrt_disc = np.sqrt(disc)
            t0 = (-self.init_v[0] + sqrt_disc) / self.acc[0]
            t1 = (-self.init_v[0] - sqrt_disc) / self.acc[0]
            candidates = [t for t in (t0, t1) if t >= 0]
            phase_time_end = min(candidates) if candidates else 0.0
        elif not np.isclose(self.init_v[0], 0.0):
            phase_time_end = phase_end_pos[0] / self.init_v[0]
        
        # 相位持续时间 = 结束时间 - 起始时间
        phase_time_span = phase_time_end - phase_time_start
        
        # 使用运动学公式 v^2 = v0^2 + 2*a*s 计算相位的初速度和末速度
        phase_init_vel = np.zeros(2, dtype=float)
        phase_final_vel = np.zeros(2, dtype=float)
        
        for i in range(2):  # x和y两个方向
            # 相位起始速度：v_start^2 = v_init^2 + 2*a*s_start
            v_sq_start = self.init_v[i]**2 + 2 * self.acc[i] * phase_start_pos[i]
            if v_sq_start >= 0:
                # 速度未反向，保持与初速度相同的符号
                phase_init_vel[i] = np.sqrt(v_sq_start) if self.init_v[i] >= 0 else -np.sqrt(v_sq_start)
            else:
                # 速度已经反向（初速度和加速度方向相反导致）
                phase_init_vel[i] = -np.sqrt(-v_sq_start) if self.init_v[i] >= 0 else np.sqrt(-v_sq_start)
            
            # 相位结束速度：v_end^2 = v_init^2 + 2*a*s_end
            v_sq_end = self.init_v[i]**2 + 2 * self.acc[i] * phase_end_pos[i]
            if v_sq_end >= 0:
                # 速度未反向，保持与初速度相同的符号
                phase_final_vel[i] = np.sqrt(v_sq_end) if self.init_v[i] >= 0 else -np.sqrt(v_sq_end)
            else:
                # 速度已经反向
                phase_final_vel[i] = -np.sqrt(-v_sq_end) if self.init_v[i] >= 0 else np.sqrt(-v_sq_end)
        
        self.phase = StepPhase(phase_id, phase_init_vel, phase_final_vel, self.stride, phase_time_start, phase_time_span)
        self.phase_array.append(self.phase)
        #print(f"Phase {phase_id}: start_pos={phase_start_pos}, end_pos={phase_end_pos}, init_vel={phase_init_vel}, final_vel={phase_final_vel}")


    def step(self, dt:float) -> Tuple[np.ndarray, np.ndarray, float]:
        
        real_dt = self.period - self.time_eslapsed if (self.time_eslapsed + dt) > self.period else dt
        new_vel = self.cur_v + self.acc * real_dt

    
        delta_pos = (self.cur_v + new_vel) * real_dt / 2
        self.cur_v = new_vel.copy()
        self.cur_pos += delta_pos
        self.time_eslapsed += real_dt

        # 相位划分：按照运动距离的六等分来计算，而非时间
        # 将当前位移除以步幅的1/6，得到当前处于第几个相位（0-5）
        phase_id = int(self.cur_pos[0] / ( self.stride[0] / 6)) % 6

        if self.phase is None or self.phase.id != phase_id:
            self.next_phase( phase_id )

        # 计算当前相位本周期内的起始时间
        phase_time_start = self.phase.time_start
        # 计算当前相位内已经过去的时间
        phase_time = self.time_eslapsed - phase_time_start
        
        swing_pos_offset, swing_vel = self.phase.swing_pv(phase_time)

        for arm_id, arm in self.robot.arms.items():
            sq_id = GAIT.WAVE_SEQUENCE.get(arm_id, 0)
            is_swing = (sq_id  == phase_id)
            if is_swing:
                arm.PositionBody = np.array([swing_pos_offset[0], swing_pos_offset[1], 0])
                arm.VelBody = swing_vel
            else:
                pos_x = self.cur_pos[0]
                if sq_id < phase_id:
                    pos_x -= (self.stride[0] + self.stride[0]/6)
                arm.PositionBody = np.array([-pos_x, 0, 0])
                arm.VelBody = -self.cur_v

        #print(f"GaitPeriod step: time_eslapsed={self.time_eslapsed:.4f}, real_dt={real_dt:.4f}, cur_v={self.cur_v}, delta_pos={delta_pos}, phase_id={phase_id}")


        return np.array([delta_pos[0], delta_pos[1], 0]), self.cur_v.copy(),  dt - real_dt
    
    def _update_leg_states(self, delta_sw_pos, delta_st_pos) -> None:
        phase_id = self.phase.id % 6

        for arm_id, arm in self.robot.arms.items():
            #base = arm.PositionBody
            sq_id = GAIT.WAVE_SEQUENCE.get(arm_id, 0)
            is_swing = (sq_id  == phase_id)
            if is_swing:
                arm.PositionBody = np.array([self.phase.swing_pos[0], 0, 0])
                arm.VelBody = self.phase.swing_vel
            else:
                pos_x = self.stride[0] / 2  - self.stride[0] / (6 - 1) * (phase_id) + self.phase.stand_pos[0]
                arm.PositionBody = np.array([-pos_x, 0, 0])
                arm.VelBody = -self.phase.stand_vel


    def _calc_params(self):
        acc = (self.target_v ** 2 - self.init_v ** 2) / (2 * self.stride)
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
    MAX_ACC = (0.01, 0.05)

    BASE_Z = -0.08 # 重心离地高度  # LOW 0.05 HIGH 0.05
    BASE_RADUIS = 0.2 #落足点半径基准 

    FRONT_ARM_OFFSET_DEGREE = 60
    BACK_ARM_OFFSET_DEGREE = -35
    MIDDLE_ARM_OFFSET_DEGREE = 15

    def __init__(self, robot:HexaRobot):
        self.gait = GAIT(GAIT.WAVE)
        self.pos = np.zeros(3)
        self.init_foot_pos:dict[int, np.array[float, float, float]] = dict()
        self.robot = robot
        self.vel = np.zeros(2)
        self.period = None
        self.tick_time = 0
        self.swing_legs: dict[int, dict[str, np.ndarray]] = {}
        self.stance_legs: dict[int, dict[str, np.ndarray]] = {}
        self.leg_states: dict[int, dict[str, object]] = {}
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
        #print(f"Setting init pos for arm {arm.id}")
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
        arm.PositionBody = self.init_foot_pos[arm.id]
    
    

    def step(self, dt:float, target_vel:Tuple[float, float]):
        
        if self.period is None:
            self.period = GaitPeriod(self.robot, self.tick_time, target_vel, target_vel, self.PERIOD_STRIDE, self.MAX_ACC,)

        while True:
            delta_pos, vel, remain = self.period.step(dt)
            self.vel = vel
            self.pos += delta_pos
            self.tick_time += dt - remain
            #self._update_leg_states(delta_pos)
            if remain > 1e-5:
                self.period = GaitPeriod(self.robot, self.tick_time, target_vel, target_vel,  self.PERIOD_STRIDE, self.MAX_ACC)
            else:
                break


    def _swing_z_offset(self) -> float:
        if self.period is None or self.period.phase is None:
            return 0.0
        if self.SWING_Z_SRIDE == 0.0:
            return 0.0
        span = self.period.phase.time_span
        if span <= 1e-6:
            return 0.0
        ratio = np.clip(self.period.phase.time_elapsed / span, 0.0, 1.0)
        return self.SWING_Z_SRIDE * np.sin(np.pi * ratio)

    def get_leg_pos_vel(self) -> dict[str, dict[int, dict[str, np.ndarray]]]:
        return {
            "swing": self.swing_legs,
            "stance": self.stance_legs,
        }
    
    def get_leg_state(self, leg_id:int) -> dict[str, object]:
        return self.leg_states.get(leg_id, {})


class pv_line:
    def __init__(self, ax:Axes, name:str):
        self.ax = ax
        self.ax_v = ax.twinx()
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Position (m)", color='r')
        self.ax_v.set_ylabel("Velocity (m/s)", color='b')
        self.ax.grid(True)
        self.ax.set_title(name)
        self.line_p, = ax.plot([], [], 'r-',  label='Position')
        self.line_v, = self.ax_v.plot([], [], 'b-', label='Velocity')
        self.ax.legend()
        self.v = []
        self.p = []
        self.t = []

    def update(self, pos:Tuple[float, float], vel:Tuple[float, float], time:float):
        self.p.append(pos)
        self.v.append(vel)
        self.t.append(time)


    def show(self):
        self.line_p.set_data(self.t, np.array(self.p)[:,0])
        self.line_v.set_data(self.t, np.array(self.v)[:,0])
        self.ax_v.relim()
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax_v.autoscale_view()

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from robot import HexaRobot, Arm
    
    plt.ion()
    fig = plt.figure(figsize=(20, 10))
    body_ax = fig.add_subplot(711)
    body_line = pv_line(body_ax, "Body Speed")
    leg_lines = {}
    for i in range(2, 8):
        leg_ax = fig.add_subplot(710 + i)
        leg_line = pv_line(leg_ax, f"Leg {i-2} Pos/Vel")
        leg_lines[i-2] = leg_line

    robot = HexaRobot()
    times = []
    gait = MoveGaitPlanner(robot)

    t = time.time()
    t_start = t

    for i in range(0, 10000):
        time.sleep(0.0005)
        dt = time.time() - t
        gait.step(dt, (0.02, 0))
        t += dt
        for i in range(0, 6):
            leg_line = leg_lines.get(i)

            pos = robot.arms.get(i).PositionBody
            vel = robot.arms.get(i).VelBody
            times.append(dt)
            body_line.update(gait.body_pos.copy(), gait.body_vel.copy(), (t - t_start) * 1000)
            leg_line.update(pos[:2], vel[:2], (t - t_start) * 1000)
    #plt.pause(0.0001)

    body_line.show()
    for i in range(0, 6):
        leg_lines[i].show()
    plt.ioff()
    plt.show()
    print(np.max(times))