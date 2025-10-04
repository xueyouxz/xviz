import numpy as np


def quaternion_to_euler_angle(rotation):
    w = rotation[0]
    x = rotation[1]
    y = rotation[2]
    z = rotation[3]

    ysqr = y * y
    t0 = -2.0 * (ysqr + z * z) + 1.0
    t1 = 2.0 * (x * y + w * z)
    t2 = -2.0 * (x * z - w * y)
    t3 = 2.0 * (y * z + w * x)
    t4 = -2.0 * (x * x + ysqr) + 1.0

    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    pitch = np.arcsin(t2)
    roll = np.arctan2(t3, t4)
    yaw = np.arctan2(t1, t0)

    return roll, pitch, yaw
