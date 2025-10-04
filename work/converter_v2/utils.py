"""
工具函数模块
提供坐标转换、几何计算等通用功能
"""
import re

import numpy as np
from pyquaternion import Quaternion
from typing import Tuple, List, Union


def quaternion_to_euler(q: Quaternion) -> Tuple[float, float, float]:
    """
    将四元数转换为欧拉角(roll, pitch, yaw)

    Args:
        q: 四元数对象

    Returns:
        (roll, pitch, yaw)元组,单位为弧度
    """
    yaw, pitch, roll = q.yaw_pitch_roll
    return (roll, pitch, yaw)


def global_to_vehicle_frame(point: np.ndarray,
                            ego_translation: np.ndarray,
                            ego_rotation: Quaternion) -> np.ndarray:
    """
    将全局坐标系下的点转换到车辆坐标系

    转换步骤:
    1. 平移: 点相对于车辆中心的偏移
    2. 旋转: 应用车辆旋转的逆变换

    Args:
        point: 全局坐标系下的点[x, y, z]
        ego_translation: 车辆在全局坐标系的位置
        ego_rotation: 车辆在全局坐标系的旋转(四元数)

    Returns:
        车辆坐标系下的点[x, y, z]
    """
    # 平移到车辆原点
    point_centered = point - ego_translation

    # 逆旋转到车辆坐标系
    point_vehicle = ego_rotation.inverse.rotate(point_centered)

    return point_vehicle


def vehicle_to_global_frame(point: np.ndarray,
                            ego_translation: np.ndarray,
                            ego_rotation: Quaternion) -> np.ndarray:
    """
    将车辆坐标系下的点转换到全局坐标系

    Args:
        point: 车辆坐标系下的点[x, y, z]
        ego_translation: 车辆在全局坐标系的位置
        ego_rotation: 车辆在全局坐标系的旋转(四元数)

    Returns:
        全局坐标系下的点[x, y, z]
    """
    # 旋转到全局坐标系
    point_rotated = ego_rotation.rotate(point)

    # 平移到全局位置
    point_global = point_rotated + ego_translation

    return point_global


def compute_box_vertices(center: np.ndarray,
                         size: List[float],
                         rotation: Quaternion) -> np.ndarray:
    """
    计算3D边界框的8个顶点坐标

    Args:
        center: 边界框中心点[x, y, z]
        size: 边界框尺寸[width, length, height]
        rotation: 旋转四元数

    Returns:
        8x3数组,每行是一个顶点的坐标
    """
    w, l, h = size

    # 在局部坐标系定义8个顶点
    # 顺序: 前左下, 前右下, 后右下, 后左下, 前左上, 前右上, 后右上, 后左上
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])

    # 应用旋转
    corners = rotation.rotation_matrix.dot(corners)

    # 应用平移
    corners[0, :] += center[0]
    corners[1, :] += center[1]
    corners[2, :] += center[2]

    return corners.T


def compute_velocity(positions: List[np.ndarray],
                     timestamps: List[int]) -> np.ndarray:
    """
    从位置序列计算平均速度

    Args:
        positions: 位置列表,每个元素是[x, y, z]
        timestamps: 时间戳列表(微秒)

    Returns:
        速度向量[vx, vy, vz] (m/s)
    """
    if len(positions) < 2:
        return np.zeros(3)

    # 计算总位移
    displacement = positions[-1] - positions[0]

    # 计算时间差(转换为秒)
    time_diff = (timestamps[-1] - timestamps[0]) / 1e6

    if time_diff < 0.01:  # 避免除零
        return np.zeros(3)

    # 计算平均速度
    velocity = displacement / time_diff

    return velocity



def interpolate_trajectory(trajectory: List[dict],
                           target_timestamps: List[int]) -> List[dict]:
    """
    在给定时间戳处插值轨迹

    Args:
        trajectory: 轨迹点列表,每个包含'timestamp'和'translation'
        target_timestamps: 目标时间戳列表

    Returns:
        插值后的轨迹点列表
    """
    if len(trajectory) < 2:
        return []

    interpolated = []

    for target_ts in target_timestamps:
        # 找到时间戳范围
        before_idx = None
        after_idx = None

        for i, point in enumerate(trajectory):
            if point['timestamp'] <= target_ts:
                before_idx = i
            if point['timestamp'] >= target_ts and after_idx is None:
                after_idx = i
                break

        if before_idx is None or after_idx is None:
            continue

        if before_idx == after_idx:
            # 精确匹配
            interpolated.append(trajectory[before_idx])
        else:
            # 线性插值
            t_before = trajectory[before_idx]['timestamp']
            t_after = trajectory[after_idx]['timestamp']

            alpha = (target_ts - t_before) / (t_after - t_before)

            pos_before = trajectory[before_idx]['translation']
            pos_after = trajectory[after_idx]['translation']

            interpolated_pos = pos_before + alpha * (pos_after - pos_before)

            interpolated.append({
                'timestamp': target_ts,
                'translation': interpolated_pos
            })

    return interpolated


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到[-pi, pi]范围

    Args:
        angle: 角度(弧度)

    Returns:
        归一化后的角度
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def get_2d_bbox_from_3d(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """
    从3D边界框顶点计算2D边界框

    Args:
        vertices: 8x3的顶点数组

    Returns:
        (min_x, min_y, max_x, max_y)
    """
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])

    return (min_x, min_y, max_x, max_y)
