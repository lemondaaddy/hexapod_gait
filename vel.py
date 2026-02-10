from model import Body
import numpy as np
def height_to_N(height:float = 0.5, stance:float = 0.09, weight: float = 3, leg_num: int = 6):
    init_vel = np.sqrt(2*9.81*height)
    acc = init_vel **2 / (2 * stance)
    t = np.sqrt(2 * stance / acc) 
    g = acc / 9.8 + 1

    N = weight * 9.8 * g

    return t, acc, N / leg_num

def vel_at_pos(move: float, acc: float):
    return np.sqrt(2 * acc * move)

if __name__ == "__main__":
    acc, N, g = height_to_N(0.5, 0.09, 3, 6)
    print(N, g)
