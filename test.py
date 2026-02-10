import numpy as np
import matplotlib.pyplot as plt

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

# 你的参数
q0, qf = 0, 2      # 位移 s=2
v0, vf = -1, -3    # 初末速度负
a0, af = 0, 0      # 起止加速度为0（平滑）
T = 3.0            # 选择总时间

coeff = solve_quintic_for_T(q0, qf, v0, vf, a0, af, T)
print("五次多项式系数:", coeff)

# 生成轨迹
t = np.linspace(0, T, 300)
pos = coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3 + coeff[4]*t**4 + coeff[5]*t**5
vel = coeff[1] + 2*coeff[2]*t + 3*coeff[3]*t**2 + 4*coeff[4]*t**3 + 5*coeff[5]*t**4
acc = 2*coeff[2] + 6*coeff[3]*t + 12*coeff[4]*t**2 + 20*coeff[5]*t**3

# 绘图
fig, axes = plt.subplots(3, 1, figsize=(8, 8))
axes[0].plot(t, pos, 'b')
axes[0].set_ylabel('Position')
axes[0].grid(True)

axes[1].plot(t, vel, 'r')
axes[1].set_ylabel('Velocity')
axes[1].grid(True)

axes[2].plot(t, acc, 'g')
axes[2].set_ylabel('Acceleration')
axes[2].set_xlabel('Time')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# 验证边界条件
print(f"t=0: pos={pos[0]:.3f}, vel={vel[0]:.3f}, acc={acc[0]:.3f}")
print(f"t=T: pos={pos[-1]:.3f}, vel={vel[-1]:.3f}, acc={acc[-1]:.3f}")