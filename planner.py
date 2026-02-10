import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class LegState:
    """Per-leg state tracked in body frame."""

    id: int
    group: int
    state: str  # "stance" or "swing"
    phase_time: float
    stance_T: float
    swing_T: float
    pos: np.ndarray
    vel: np.ndarray
    lift_off_pos: np.ndarray
    target_pos: np.ndarray


class HexapodGaitPlanner:
    """Plan body motion (world) and foot motion (body) for a hexapod.

    Inputs: target velocity [vx, vy] in world frame.
    Outputs: body acc/vel/pos in world frame and per-leg foot pos/vel in body frame.
    """

    def __init__(
        self,
        stride_period: float = 0.6,
        duty_factor: float = 0.65,
        max_acc: Tuple[float, float] = (0.8, 0.6),
        base_z: float = -0.08,
        clearance: float = 0.03,
        nominal_xy: Dict[int, Tuple[float, float]] | None = None,
        stride_period_range: Tuple[float, float] = (0.35, 0.8),
        speed_for_min_period: float = 0.4,
        period_tau: float = 0.25,
    ) -> None:
        if duty_factor <= 0 or duty_factor >= 1:
            raise ValueError("duty_factor must be in (0, 1)")

        self.stride_period = stride_period
        self.min_stride_period, self.max_stride_period = stride_period_range
        self.speed_for_min_period = speed_for_min_period
        self.period_tau = period_tau
        self.duty_factor = duty_factor
        self.stance_T = stride_period * duty_factor
        self.swing_T = stride_period * (1.0 - duty_factor)
        self.base_z = base_z
        self.clearance = clearance

        self.max_acc = np.array(max_acc, dtype=float)
        self.body_vel = np.zeros(2, dtype=float)
        self.body_pos = np.zeros(2, dtype=float)
        self.body_acc = np.zeros(2, dtype=float)

        self.nominal_xy = self._build_nominal_xy(nominal_xy)
        self.legs: Dict[int, LegState] = self._init_legs()

    def step(self, target_vel: Tuple[float, float], dt: float) -> Dict[str, object]:
        prev_vel = self.body_vel.copy()
        self._update_stride_period(target_vel, dt)
        self._update_body(target_vel, dt, prev_vel)

        for leg in self.legs.values():
            self._update_leg(leg, dt, prev_vel, target_vel)

        return {
            "body": {
                "pos_world": self.body_pos.copy(),
                "vel_world": self.body_vel.copy(),
                "acc_world": self.body_acc.copy(),
            },
            "legs": {
                leg.id: {
                    "pos_body": leg.pos.copy(),
                    "vel_body": leg.vel.copy(),
                    "state": leg.state,
                }
                for leg in self.legs.values()
            },
        }

    # --- internal helpers ---
    def _update_body(self, target_vel: Tuple[float, float], dt: float, prev_vel: np.ndarray) -> None:
        target = np.array(target_vel, dtype=float)
        acc_cmd = np.clip((target - prev_vel) / dt, -self.max_acc, self.max_acc)
        self.body_acc = acc_cmd
        self.body_vel = prev_vel + self.body_acc * dt
        self.body_pos += prev_vel * dt + 0.5 * self.body_acc * dt * dt

    def _update_leg(self, leg: LegState, dt: float, prev_body_vel: np.ndarray, target_vel: Tuple[float, float]) -> None:
        leg.phase_time += dt
        if leg.state == "stance":
            delta = -(prev_body_vel * dt + 0.5 * self.body_acc * dt * dt)
            leg.pos[:2] += delta
            leg.pos[2] = self.base_z
            leg.vel[:2] = -self.body_vel
            leg.vel[2] = 0.0

            if leg.phase_time >= leg.stance_T:
                leg.state = "swing"
                leg.phase_time = 0.0
                leg.lift_off_pos = leg.pos.copy()
                leg.target_pos = self._compute_target_foothold(leg.id, target_vel)
        else:
            s = min(leg.phase_time / leg.swing_T, 1.0)
            pos, vel = self._swing_profile(leg.lift_off_pos, leg.target_pos, s, leg.swing_T)
            leg.pos = pos
            leg.vel = vel

            if leg.phase_time >= leg.swing_T:
                leg.state = "stance"
                leg.phase_time = 0.0
                leg.pos = leg.target_pos.copy()
                leg.vel = np.zeros(3, dtype=float)

    def _compute_target_foothold(self, leg_id: int, target_vel: Tuple[float, float]) -> np.ndarray:
        nominal = self.nominal_xy[leg_id]
        v_des = np.array(target_vel, dtype=float)
        step = v_des * self.stride_period * 0.5
        goal_xy = nominal + step
        return np.array([goal_xy[0], goal_xy[1], self.base_z], dtype=float)

    def _swing_profile(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        s: float,
        T: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        dblend = (30 * s * s - 60 * s**3 + 30 * s**4) / T

        pos_xy = p0[:2] + (p1[:2] - p0[:2]) * blend
        vel_xy = (p1[:2] - p0[:2]) * dblend

        pos_z = self.base_z + self.clearance * np.sin(np.pi * s)
        vel_z = self.clearance * np.pi * np.cos(np.pi * s) / T

        pos = np.array([pos_xy[0], pos_xy[1], pos_z], dtype=float)
        vel = np.array([vel_xy[0], vel_xy[1], vel_z], dtype=float)
        return pos, vel

    def _build_nominal_xy(
        self, nominal_xy: Dict[int, Tuple[float, float]] | None
    ) -> Dict[int, np.ndarray]:
        if nominal_xy is not None:
            return {k: np.array(v, dtype=float) for k, v in nominal_xy.items()}

        return {
            0: np.array([0.12, 0.10], dtype=float),  # LF
            1: np.array([0.04, 0.11], dtype=float),  # LM
            2: np.array([-0.08, 0.10], dtype=float),  # LR
            3: np.array([-0.08, -0.10], dtype=float),  # RR
            4: np.array([0.04, -0.11], dtype=float),  # RM
            5: np.array([0.12, -0.10], dtype=float),  # RF
        }

    def _init_legs(self) -> Dict[int, LegState]:
        legs: Dict[int, LegState] = {}
        tripod_a = {0, 4, 3}  # LF, RM, RR
        tripod_b = {5, 1, 2}  # RF, LM, LR

        for leg_id, xy in self.nominal_xy.items():
            group = 0 if leg_id in tripod_a else 1
            state = "stance" if group == 0 else "swing"
            phase_time = 0.0 if state == "stance" else self.swing_T * 0.5
            pos = np.array([xy[0], xy[1], self.base_z], dtype=float)
            legs[leg_id] = LegState(
                id=leg_id,
                group=group,
                state=state,
                phase_time=phase_time,
                stance_T=self.stance_T,
                swing_T=self.swing_T,
                pos=pos,
                vel=np.zeros(3, dtype=float),
                lift_off_pos=pos.copy(),
                target_pos=pos.copy(),
            )

        return legs

    def _update_stride_period(self, target_vel: Tuple[float, float], dt: float) -> None:
        speed = np.linalg.norm(target_vel)
        span = self.max_stride_period - self.min_stride_period
        if span <= 0:
            return

        ratio = speed / self.speed_for_min_period if self.speed_for_min_period > 1e-6 else 1.0
        ratio = float(np.clip(ratio, 0.0, 1.0))
        target_T = self.max_stride_period - span * ratio

        if self.period_tau > 1e-6:
            alpha = np.clip(dt / self.period_tau, 0.0, 1.0)
        else:
            alpha = 1.0

        old_stride = self.stride_period
        new_stride = old_stride + (target_T - old_stride) * alpha

        if np.isclose(new_stride, old_stride):
            return

        old_stance = self.stance_T
        old_swing = self.swing_T

        self.stride_period = new_stride
        self.stance_T = new_stride * self.duty_factor
        self.swing_T = new_stride * (1.0 - self.duty_factor)

        for leg in self.legs.values():
            if leg.state == "stance":
                ratio_phase = leg.phase_time / old_stance if old_stance > 1e-6 else 0.0
                leg.stance_T = self.stance_T
                leg.swing_T = self.swing_T
                leg.phase_time = min(ratio_phase * leg.stance_T, leg.stance_T)
            else:
                ratio_phase = leg.phase_time / old_swing if old_swing > 1e-6 else 0.0
                leg.stance_T = self.stance_T
                leg.swing_T = self.swing_T
                leg.phase_time = min(ratio_phase * leg.swing_T, leg.swing_T)


