import numpy as np
from itertools import product

# ----------------- 参数区域（可改） -----------------
L0, L1, L2 = 0.05465, 0.05, 0.11  # 米
A0_range = (-np.pi, np.pi)
A1_range = (-np.pi,   np.pi)
A2_range = (-np.pi,   np.pi)

# 栅格采样分辨率（包含端点），总点数约=N0*N1*N2，建议先小一些观察时间
N0, N1, N2 = 11, 13, 15

# 随机样本数
N_rand = 1000

# 数值容差
POS_EPS = 1e-9          # 末端位置几何一致容差（米）
ANG_EPS = max(1e-7, 20*np.finfo(float).eps)  # ~1e-15 上限保护
# ---------------------------------------------------

def ang_wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def ang_diff(a, b):
    return ang_wrap_pi(a - b)

def map_to_interval_equiv(a, lo, hi, ref=None):
    """
    将角 a 通过加减 2π 的等价变换映射到 [lo, hi] 内，不改变几何。
    - 若 ref 给出，则选择与 ref 差的绝对值最小的那个等价角。
    - 假设区间长度 0 < (hi-lo) <= 2π（满足你的所有关节设定）。
    """
    w = hi - lo
    assert w > 0.0 and w <= 2*np.pi + 1e-12
    # 先把 a 映射到 [lo, lo+2π)
    a0 = (a - lo) % (2*np.pi) + lo
    if a0 <= hi + 1e-12:
        # 已在区间内，若需要靠近 ref，可尝试 a0±2π（通常没必要，因为长度<=2π）
        if ref is None:
            return a0
        # 在所有可能的 a0+2kπ 中选择最接近 ref 的且落在 [lo,hi] 的
        candidates = [a0]
        a_m = a0 - 2*np.pi
        a_p = a0 + 2*np.pi
        if lo - 1e-12 <= a_m <= hi + 1e-12: candidates.append(a_m)
        if lo - 1e-12 <= a_p <= hi + 1e-12: candidates.append(a_p)
        return min(candidates, key=lambda t: abs(ang_diff(t, ref)))
    else:
        # a0 在 (hi, lo+2π)，再减掉 2π 必然回到 [lo,hi]
        a1 = a0 - 2*np.pi
        if ref is None:
            return a1
        candidates = [a1]
        a_m = a1 - 2*np.pi
        a_p = a1 + 2*np.pi
        if lo - 1e-12 <= a_m <= hi + 1e-12: candidates.append(a_m)
        if lo - 1e-12 <= a_p <= hi + 1e-12: candidates.append(a_p)
        return min(candidates, key=lambda t: abs(ang_diff(t, ref)))

def fk(A0, A1, A2, L0, L1, L2):
    c1, s1   = np.cos(A1), np.sin(A1)
    c12, s12 = np.cos(A1 + A2), np.sin(A1 + A2)
    S = L0 + L1*c1 + L2*c12
    x = -np.sin(A0) * S
    y =  np.cos(A0) * S
    z =  L1*s1 + L2*s12
    return np.array([x, y, z])

def ik(x, y, z, L0, L1, L2, hint=None,
                A0_range=(-np.pi, np.pi),
                A1_range=(-np.pi/2,   np.pi/2),
                A2_range=(-np.pi*15/18,   np.pi)):  # 修改A2_range，将下限限制到约-120度(-2.09 rad)
    """
    关键变化：
    1) A0 不再硬夹，只通过等价分支 (A0, S) 与 (A0±π, -S) 选择；
    2) A1/A0 在返回时用 map_to_interval_equiv() 进入限幅（保持几何不变）；
    3) A2 仍用 wrap 至 [-π, π]，它等价区间就是关节极限。
    """
    
    # 1) 由 XY 求 A0 的原始值与 |S|
    A0_raw = np.arctan2(-x, y)
    S_abs  = np.hypot(x, y)

    # 生成两个等价分支：确保都能代表 [-pi/2,pi/2] 的区间
    A0a, Sa = A0_raw, S_abs
    if A0a > np.pi/2:    A0a, Sa = A0a - np.pi, -Sa
    if A0a < -np.pi/2:   A0a, Sa = A0a + np.pi, -Sa
    A0b, Sb = ang_wrap_pi(A0a + np.pi), -Sa
    # 如果有hint，优先选择与hint[0]最接近的分支
    if hint is not None:
        # 计算两个分支与hint[0]的差异
        diff_a = abs(ang_diff(A0a, hint[0]))
        diff_b = abs(ang_diff(A0b, hint[0]))
        
        # 选择差异较小的分支
        if diff_a <= diff_b:
            branches = [(A0a, Sa)]
        else:
            branches = [(A0b, Sb)]
    else:
        branches = [(A0a, Sa), (A0b, Sb)]
    candidates = []
    for A0c, S in [(A0a, Sa), (A0b, Sb)]:
        Y, Z = S - L0, z
        r2 = Y*Y + Z*Z
        if r2 > (L1 + L2)**2 + 1e-12 or r2 < (L1 - L2)**2 - 1e-12:
            continue

        c = (r2 - L1*L1 - L2*L2) / (2*L1*L2)
        c = float(np.clip(c, -1.0, 1.0))
        s = np.sqrt(max(0.0, 1 - c*c))
        # 只生成肘上解 (sgn = -1)
        sgn = -1  # 只使用肘上解
        A2 = np.arctan2(sgn*s, c)
        A1 = np.arctan2(Z, Y) - np.arctan2(L2*np.sin(A2), L1 + L2*np.cos(A2))

        # 把 A0, A1 映射进用户给的区间（用 hint 作为参考，尽量贴近 hint）
        A0_m = map_to_interval_equiv(A0c, A0_range[0], A0_range[1],
                                     ref=(hint[0] if hint is not None else None))
        A1_m = map_to_interval_equiv(ang_wrap_pi(A1), A1_range[0], A1_range[1],
                                     ref=(hint[1] if hint is not None else None))
        A2_m = ang_wrap_pi(A2)  # A2 的区间正好是等价主值区间
                # 确保A2在[0, π]范围内
                # 确保A2在[0, π]范围内
            # 确保所有关节在指定范围内
        if (A0_range[0] <= A0_m <= A0_range[1] + 1e-12 and
            A1_range[0] <= A1_m <= A1_range[1] + 1e-12 and
            A2_range[0] <= A2_m <= A2_range[1] + 1e-12):
            
            # # 添加额外的关节限制，避免腿部接近平伸状态（防止机器人"趴下"）
            # # 对于腿部机构，A1（髋关节俯仰）和A2（膝关节）需要保持一定角度
            # # A1应该保持正值（腿部向上弯曲），A2应该在合理范围内
            # # 检查是否接近平伸状态（A1接近0度，A2接近0度）
            # if abs(A1_m) < 0.35 and abs(A2_m) < 0.5:  # 更严格的限制，从0.2和0.3改为0.35和0.5
            #     continue  # 跳过这个解，因为它会导致腿部接近平伸状态
            
            candidates.append(np.array([A0_m, A1_m, A2_m]))
            # elbow_type = "down" if sgn > 0 else "up"
            # print(f"  {elbow_type}解: "
            #       f"A0 = {np.degrees(A0_m):.2f}°, "
            #       f"A1 = {np.degrees(A1_m):.2f}°, "
            #       f"A2 = {np.degrees(A2_m):.2f}°")
    if not candidates:
        if hint is not None:
            # 如果IK无解，返回hint的限制版本，而不是默认值
            # 这样可以保持关节角度的连续性
            result = np.array([
                map_to_interval_equiv(hint[0], A0_range[0], A0_range[1]),
                map_to_interval_equiv(hint[1], A1_range[0], A1_range[1]),
                map_to_interval_equiv(hint[2], A2_range[0], A2_range[1])
            ])
            print(f"  IK无解，使用限制后的hint: "
                  f"A0 = {np.degrees(result[0]):.2f}°, "
                  f"A1 = {np.degrees(result[1]):.2f}°, "
                  f"A2 = {np.degrees(result[2]):.2f}°")
            return result
        else:
            # 修改：如果无解，返回零位附近的值，但保持在合理范围内
            # 原先返回零位，现在返回默认角度（保持在关节范围内）
            safe_result = np.array([
                0.0,  # A0
                np.clip(0.0, A1_range[0], A1_range[1]),  # A1 - 使用0，但限制在范围内
                np.clip(-np.pi/2, A2_range[0], A2_range[1])  # A2 - 限制在范围内
            ])
            print(f"  IK无解，使用安全默认值: "
                  f"A0 = {np.degrees(safe_result[0]):.2f}°, "
                  f"A1 = {np.degrees(safe_result[1]):.2f}°, "
                  f"A2 = {np.degrees(safe_result[2]):.2f}°")
            return safe_result

    # 分支选择
    if hint is not None:
        def cost(A):
            d0_weight = 10.0  # A0差异的权重因子
            d = np.array([d0_weight * ang_diff(A[0], hint[0]),
                          ang_diff(A[1], hint[1]),
                          ang_diff(A[2], hint[2])])
            return float(np.dot(d, d))
        sol = min(candidates, key=cost)
    else:
        # 即使没有hint，也优先选择A0接近0的解（机械臂前方）
        sol = min(candidates, key=lambda A: abs(A[0]))

    sol = np.array([ang_wrap_pi(sol[0]),
                    ang_wrap_pi(sol[1]),
                    ang_wrap_pi(sol[2])])

    return sol

def grid(l, h, n):
    if n <= 1: return np.array([(l+h)/2.0])
    return np.linspace(l, h, n)

def angle_err(a, b):
    d = np.array([ang_diff(a[0], b[0]), ang_diff(a[1], b[1]), ang_diff(a[2], b[2])])
    return np.linalg.norm(d, ord=np.inf), d  # 返回无穷范数和分量差

def angle_equiv_err(a, b):
    """
    用 sin/cos 向量的差来度量角度等价性，避免 ±π 主值切换处的抖动误报。
    返回无穷范数（每关节 max）和分量误差。
    """
    d_sin = np.sin(a) - np.sin(b)
    d_cos = np.cos(a) - np.cos(b)
    # 每个关节取二者范数作为该关节的“等价角误差”
    e_j = np.sqrt(d_sin**2 + d_cos**2)  # 三个关节的误差
    return float(np.max(e_j)), e_j

def pos_err(p, q):
    return np.linalg.norm(p - q)

def in_limits(A):
    ok = (A0_range[0]-1e-12 <= A[0] <= A0_range[1]+1e-12) \
         and (A1_range[0]-1e-12 <= A[1] <= A1_range[1]+1e-12) \
         and (-np.pi-1e-12 <= A[2] <= np.pi+1e-12)
    return ok

# ----------------- 验证主流程 -----------------
def validate():
    worst_hint = {"ang_err": -1.0, "A": None, "A_back": None}
    worst_geo  = {"pos_err": -1.0, "A": None, "A_free": None}

    bad_hint = 0
    bad_geo  = 0
    total    = 0

    # 1) 栅格采样
    A0s = grid(*A0_range, N0)
    A1s = grid(*A1_range, N1)
    A2s = grid(*A2_range, N2)

    for A0, A1, A2 in product(A0s, A1s, A2s):
        A = np.array([A0, A1, A2], dtype=float)
        p = fk(*A, L0, L1, L2)

        # (a) 带 hint 回代
        Ab = ik(*p, L0, L1, L2, hint=A)
        e_inf, dvec = angle_err(Ab, A)
        if e_inf > ANG_EPS or not in_limits(Ab):
            bad_hint += 1
        if e_inf > worst_hint["ang_err"]:
            worst_hint.update({"ang_err": e_inf, "A": A.copy(), "A_back": Ab.copy()})

        # (b) 不带 hint 几何一致
        Afree = ik(*p, L0, L1, L2, hint=None)
        p2 = fk(*Afree, L0, L1, L2)
        pe = pos_err(p2, p)
        if pe > POS_EPS:
            bad_geo += 1
        if pe > worst_geo["pos_err"]:
            worst_geo.update({"pos_err": pe, "A": A.copy(), "A_free": Afree.copy()})

        total += 1

    # 2) 随机采样
    rng = np.random.default_rng(1234)
    for _ in range(N_rand):
        A = np.array([
            rng.uniform(*A0_range),
            rng.uniform(*A1_range),
            rng.uniform(*A2_range),
        ])
        p = fk(*A, L0, L1, L2)

        Ab = ik(*p, L0, L1, L2, hint=A)
        e_inf, dvec = angle_equiv_err(Ab, A)
        if e_inf > ANG_EPS or not in_limits(Ab):
            bad_hint += 1
        if e_inf > worst_hint["ang_err"]:
            worst_hint.update({"ang_err": e_inf, "A": A.copy(), "A_back": Ab.copy()})

        Afree = ik(*p, L0, L1, L2, hint=None)
        p2 = fk(*Afree, L0, L1, L2)
        pe = pos_err(p2, p)
        if pe > POS_EPS:
            bad_geo += 1
        if pe > worst_geo["pos_err"]:
            worst_geo.update({"pos_err": pe, "A": A.copy(), "A_free": Afree.copy()})

        total += 1

    # 3) 端点/边界附加测试（每维上下限笛卡尔积）
    for A0 in [A0_range[0], A0_range[1]]:
        for A1 in [A1_range[0], A1_range[1]]:
            for A2 in [A2_range[0], A2_range[1]]:
                A = np.array([A0, A1, A2], float)
                p = fk(*A, L0, L1, L2)

                Ab = ik(*p, L0, L1, L2, hint=A)
                e_inf, _ = angle_equiv_err(Ab, A)
                if e_inf > ANG_EPS or not in_limits(Ab):
                    bad_hint += 1
                if e_inf > worst_hint["ang_err"]:
                    worst_hint.update({"ang_err": e_inf, "A": A.copy(), "A_back": Ab.copy()})

                Afree = ik(*p, L0, L1, L2, hint=None)
                p2 = fk(*Afree, L0, L1, L2)
                pe = pos_err(p2, p)
                if pe > POS_EPS:
                    bad_geo += 1
                if pe > worst_geo["pos_err"]:
                    worst_geo.update({"pos_err": pe, "A": A.copy(), "A_free": Afree.copy()})

                total += 1

    # 4) 汇总
    print(f"总样本数: {total}")
    print(f"[带 hint 回代] 违反容差数量: {bad_hint}, 最大角度无穷范数误差: {worst_hint['ang_err']:.3e} rad")
    print("  最差样例（A_true -> A_back）:")
    print("    A_true =", worst_hint["A"])
    print("    A_back =", worst_hint["A_back"])
    print(f"[不带 hint 几何] 违反容差数量: {bad_geo}, 最大位置误差: {worst_geo['pos_err']:.3e} m")
    print("  最差样例（A_true -> A_free，并回 FK 比较）:")
    print("    A_true =", worst_geo["A"])
    print("    A_free =", worst_geo["A_free"])

if __name__ == "__main__":
    validate()
