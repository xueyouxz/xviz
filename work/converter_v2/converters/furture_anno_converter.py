"""
未来对象位置预测转换器
将NuScenes对象的未来轨迹(3秒内,6步)转换为XVIZ格式
"""
import numpy as np
from typing import Dict, Any, List
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class FutureAnnoConverter:
    """
    未来对象转换器

    预测并可视化对象在未来3秒内的位置
    采用线性插值方法基于历史轨迹预测未来位置
    """

    def __init__(self, nusc: NuScenes, config: Dict[str, Any]):
        """
        初始化未来对象转换器

        Args:
            nusc: NuScenes数据集实例
            config: 配置字典
        """
        self.nusc = nusc
        self.config = config

        # XVIZ流名称
        self.FUTURE_TRAJECTORY_STREAM = '/object/future_trajectory'
        self.FUTURE_BOXES_STREAM = '/object/future_boxes'

        # 预测参数
        self.prediction_horizon = 3.0  # 预测时长(秒)
        self.prediction_steps = 6  # 预测步数
        self.time_step = self.prediction_horizon / self.prediction_steps  # 每步时间间隔

    def load(self, frames: List[Dict[str, Any]]):
        """
        加载数据并构建对象轨迹

        Args:
            frames: 所有帧的信息列表
        """
        self.frames = frames

        # 构建对象轨迹用于预测
        self.object_trajectories = self._build_object_trajectories()

    def _build_object_trajectories(self) -> Dict[str, List[Dict]]:
        """
        构建所有对象的完整轨迹

        Returns:
            字典: {instance_token: [{'timestamp': ts, 'translation': xyz, ...}]}
        """
        trajectories = {}

        for frame in self.frames:
            sample = self.nusc.get('sample', frame['token'])

            # 遍历该帧的所有标注
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']

                if instance_token not in trajectories:
                    trajectories[instance_token] = []

                trajectories[instance_token].append({
                    'timestamp': frame['timestamp'],
                    'translation': np.array(ann['translation']),
                    'rotation': Quaternion(ann['rotation']),
                    'size': ann['size'],
                    'velocity': np.array(ann.get('velocity', [0, 0, 0]))  # 有些标注包含速度
                })

        # 按时间戳排序
        for instance_token in trajectories:
            trajectories[instance_token].sort(key=lambda x: x['timestamp'])

        return trajectories

    def convert_message(self, message_index: int, xviz_builder, frame: Dict[str, Any]):
        """
        转换单帧的未来预测数据为XVIZ格式

        Args:
            message_index: 帧索引
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        sample = self.nusc.get('sample', frame['token'])
        current_timestamp = frame['timestamp']

        # 获取车辆位姿
        ego_pose = self._get_ego_pose(sample)

        # 遍历所有标注对象,预测未来位置
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            instance_token = ann['instance_token']

            # 预测该对象的未来轨迹
            future_positions = self._predict_future_positions(
                instance_token,
                current_timestamp,
                ann
            )

            if future_positions is None or len(future_positions) == 0:
                continue

            # 转换并添加未来轨迹
            self._convert_future_trajectory(
                instance_token,
                future_positions,
                ego_pose,
                xviz_builder
            )

            # 转换并添加未来边界框
            self._convert_future_boxes(
                instance_token,
                future_positions,
                ann['size'],
                ego_pose,
                xviz_builder
            )

    def _predict_future_positions(self, instance_token: str,
                                  current_timestamp: int,
                                  current_ann: Dict) -> List[Dict]:
        """
        预测对象未来的位置

        预测方法:
        1. 如果有速度信息,使用匀速运动模型
        2. 否则,使用历史轨迹的平均速度

        Args:
            instance_token: 对象实例token
            current_timestamp: 当前时间戳(微秒)
            current_ann: 当前标注信息

        Returns:
            未来位置列表,每个包含{timestamp, translation, rotation}
        """
        if instance_token not in self.object_trajectories:
            return None

        trajectory = self.object_trajectories[instance_token]

        # 找到当前时刻在轨迹中的位置
        current_idx = None
        for i, point in enumerate(trajectory):
            if point['timestamp'] == current_timestamp:
                current_idx = i
                break

        if current_idx is None:
            return None

        # 估计速度
        velocity = self._estimate_velocity(trajectory, current_idx)

        if velocity is None or np.linalg.norm(velocity) < 0.1:
            # 速度太小或无法估计,对象可能静止,不预测
            return None

        # 预测未来位置
        current_position = trajectory[current_idx]['translation']
        current_rotation = trajectory[current_idx]['rotation']

        future_positions = []

        for step in range(1, self.prediction_steps + 1):
            # 时间增量(秒)
            dt = step * self.time_step

            # 使用匀速模型预测位置
            future_pos = current_position + velocity * dt

            # 假设朝向不变(简化模型)
            future_positions.append({
                'timestamp': current_timestamp + int(dt * 1e6),  # 转换为微秒
                'translation': future_pos,
                'rotation': current_rotation
            })

        return future_positions

    def _estimate_velocity(self, trajectory: List[Dict], current_idx: int) -> np.ndarray:
        """
        从历史轨迹估计速度

        使用最近的几个点计算平均速度

        Args:
            trajectory: 对象轨迹
            current_idx: 当前位置索引

        Returns:
            速度向量[vx, vy, vz] (m/s)
        """
        # 检查是否有标注的速度信息
        current_point = trajectory[current_idx]
        if 'velocity' in current_point and np.linalg.norm(current_point['velocity']) > 0:
            # 将速度转换为3D向量（添加z分量为0）
            velocity_2d = np.array(current_point['velocity'][:2])
            return np.array([velocity_2d[0], velocity_2d[1], 0.0])

        # 使用历史轨迹计算速度
        # 取最近的3个点(如果有的话)
        lookback = min(3, current_idx + 1)
        if lookback < 2:
            return None

        # 计算位移和时间差
        start_idx = current_idx - lookback + 1
        positions = [p['translation'] for p in trajectory[start_idx:current_idx + 1]]
        timestamps = [p['timestamp'] for p in trajectory[start_idx:current_idx + 1]]

        # 计算平均速度
        total_displacement = positions[-1] - positions[0]
        total_time = (timestamps[-1] - timestamps[0]) / 1e6  # 转换为秒

        if total_time < 0.01:
            return None

        velocity = total_displacement / total_time

        # 保持3D速度向量，但z分量设为0（通常我们只关注水平运动）
        return np.array([velocity[0], velocity[1], 0.0])

    def _convert_future_trajectory(self, instance_token: str,
                                   future_positions: List[Dict],
                                   ego_pose: Dict,
                                   xviz_builder):
        """
        转换未来轨迹为XVIZ格式

        Args:
            instance_token: 对象实例token
            future_positions: 未来位置列表
            ego_pose: 车辆位姿
            xviz_builder: XVIZ构建器
        """
        # 转换所有未来位置到车辆坐标系
        trajectory_points = []
        for pos_data in future_positions:
            global_pos = pos_data['translation']
            vehicle_pos = self._global_to_vehicle(global_pos, ego_pose)
            trajectory_points.extend(vehicle_pos.tolist())

        # 添加未来轨迹线到XVIZ
        # 使用虚线样式表示这是预测
        xviz_builder.primitive(self.FUTURE_TRAJECTORY_STREAM) \
            .polyline(trajectory_points) \
            .id(f"{instance_token}_future") \


    def _convert_future_boxes(self, instance_token: str,
                              future_positions: List[Dict],
                              size: List[float],
                              ego_pose: Dict,
                              xviz_builder):
        """
        转换未来边界框为XVIZ格式

        仅显示最后一步的边界框(3秒后的位置)

        Args:
            instance_token: 对象实例token
            future_positions: 未来位置列表
            size: 对象尺寸[width, length, height]
            ego_pose: 车辆位姿
            xviz_builder: XVIZ构建器
        """
        # 只显示最后一步的边界框
        if len(future_positions) == 0:
            return

        final_position = future_positions[-1]

        # 转换到车辆坐标系
        global_pos = final_position['translation']
        vehicle_pos = self._global_to_vehicle(global_pos, ego_pose)

        # 计算车辆坐标系下的旋转
        ego_rotation = Quaternion(ego_pose['rotation'])
        rotation_vehicle = ego_rotation.inverse * final_position['rotation']
        yaw = rotation_vehicle.yaw_pitch_roll[0]

        # 添加未来边界框到XVIZ
        xviz_builder.primitive(self.FUTURE_BOXES_STREAM) \
            .polygon([
            vehicle_pos[0], vehicle_pos[1], vehicle_pos[2],
            size[0], size[1], size[2],
            yaw
        ]) \
            .id(f"{instance_token}_future_box") \


    def _get_ego_pose(self, sample: Dict) -> Dict:
        """
        获取车辆在全局坐标系的位姿

        Args:
            sample: 样本数据

        Returns:
            位姿字典
        """
        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

        return {
            'translation': ego_pose['translation'],
            'rotation': ego_pose['rotation']
        }

    def _global_to_vehicle(self, point: np.ndarray, ego_pose: Dict) -> np.ndarray:
        """
        将全局坐标转换为车辆坐标系

        Args:
            point: 全局坐标点
            ego_pose: 车辆位姿

        Returns:
            车辆坐标系下的点
        """
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        point_centered = point - ego_translation
        point_vehicle = ego_rotation.inverse.rotate(point_centered)

        return point_vehicle

    def get_metadata(self, xviz_metadata):
        """
        定义未来预测流的元数据

        Args:
            xviz_metadata: XVIZ元数据构建器
        """
        # 定义未来轨迹流
        xviz_metadata.stream(self.FUTURE_TRAJECTORY_STREAM) \
            .category('primitive') \
            .type('polyline') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
            'stroke_color': [255, 200, 0, 200],
            'stroke_width': 0.15
        })

        # 定义未来边界框流
        xviz_metadata.stream(self.FUTURE_BOXES_STREAM) \
            .category('primitive') \
            .type('polygon') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
            'extruded': True,
            'fill_color': [255, 200, 0],
            'stroke_color': [255, 200, 0],
            'stroke_width': 0.1
        })