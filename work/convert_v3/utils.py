import numpy as np
from scipy.spatial.transform import Rotation
from typing import List


def hex_to_rgba(hex_color: str) -> List[int]:
    """将十六进制颜色转换为RGBA数组"""
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    if len(hex_color) == 6:
        hex_color += 'FF'
    elif len(hex_color) == 8:
        pass
    else:
        raise ValueError(f"无效的颜色格式: {hex_color}")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = int(hex_color[6:8], 16)

    return [r, g, b, a]


def quaternion_to_euler(quaternion: List[float]) -> tuple:
    """将四元数转换为欧拉角"""
    rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    euler = rotation.as_euler('xyz', degrees=False)
    return tuple(euler)

