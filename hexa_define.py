import numpy as np

ARM_LF = 0
ARM_LM = 1
ARM_LL = 2
ARM_RL = 3
ARM_RM = 4
ARM_RF = 5

ARM_BASE_COOR = {
    ARM_LF: (0.058, 0.061, 0),
    ARM_LM: (0.016, 0.061, 0.003),
    ARM_LL: (-0.057, 0.055, 0),
    ARM_RL: (-0.057, -0.055, 0),
    ARM_RM: (0.016, -0.061,0.003),
    ARM_RF: (0.058, -0.061, 0)
}

ARM_LENGTH  = {
    ARM_LF: (0.053, 0.05, 0.128),
    ARM_LM: (0.05, 0.05, 0.110),
    ARM_LL: (0.05, 0.05, 0.110),
    ARM_RL: (0.05, 0.05, 0.110),
    ARM_RM: (0.05, 0.05, 0.110),
    ARM_RF: (0.053, 0.05, 0.128)
}

ARM_JOINT_LIMIT = {
    ARM_LF: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    ARM_LM: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    ARM_LL: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    ARM_RL: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    ARM_RM: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
    ARM_RF: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
}