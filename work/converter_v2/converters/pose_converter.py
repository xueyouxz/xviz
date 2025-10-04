"""
车辆位姿转换器
将NuScenes的ego_pose转换为XVIZ的/vehicle_pose流

这是最关键的转换器：
- 必须首先执行，为其他VEHICLE_RELATIVE坐标提供参考
- 定义车辆在每一帧的位置和朝向
- 使streetscape.gl能够正确渲染3D场景
"""
import numpy as np
from typing import Dict, Any, List
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


class PoseConverter:
    """
    车辆位姿转换器

    关键作用:
    1. 从ego_pose获取车辆的全局位置和朝向
    2. 转换为XVIZ所需的格式
    3. 为VEHICLE_RELATIVE坐标系提供参考

    没有正确的pose，3D场景将无法显示！
    """

    def __init__(self, nusc: NuScenes, config: Dict[str, Any]):
        """
        初始化位姿转换器

        Args:
            nusc: NuScenes数据集实例
            config: 配置字典
        """
        self.nusc = nusc
        self.config = config

        # XVIZ流名称 - 这是标准的vehicle pose流
        self.VEHICLE_POSE = '/vehicle_pose'
        self.VEHICLE_TRAJECTORY = '/vehicle/trajectory'  # 可选：显示车辆轨迹

        # 存储所有帧的pose数据
        self.poses_by_frame = {}



    def load(self, frames: List[Dict[str, Any]]):
        """
        加载所有帧的ego_pose数据

        Args:
            frames: 所有帧的信息列表
        """
        self.frames = frames

        print(f"Loading {len(frames)} ego poses...")

        for i, frame in enumerate(frames):
            # 获取sample
            sample = self.nusc.get('sample', frame['token'])

            # 获取ego_pose
            ego_pose = self._get_ego_pose(sample)

            # 转换为XVIZ格式
            pose_data = self._convert_pose(ego_pose, frame['timestamp'])

            # 存储
            self.poses_by_frame[frame['token']] = pose_data



        print(f"  ✓ Loaded {len(self.poses_by_frame)} poses")

    def convert_message(self, message_index: int, xviz_builder, frame: Dict[str, Any]):
        """
        为当前帧设置vehicle_pose

        这是最关键的步骤：没有正确的pose，3D场景无法渲染！

        Args:
            message_index: 帧索引
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        # 获取该帧的pose数据
        pose_data = self.poses_by_frame[frame['token']]

        # 设置vehicle_pose - 这是XVIZ的核心！
        xviz_builder.pose(self.VEHICLE_POSE) \
            .timestamp(pose_data['timestamp']) \
            .map_origin(0, 0, 0) \
            .position(
            pose_data['position'][0],
            pose_data['position'][1],
            pose_data['position'][2]
        ) .orientation(
            pose_data['orientation'][0],  # roll
            pose_data['orientation'][1],  # pitch
            pose_data['orientation'][2]  # yaw
        )

        # 可选：添加车辆轨迹线
        if self.config.get('show_vehicle_trajectory', False):
            self._add_trajectory(message_index, xviz_builder)

    def _get_ego_pose(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        从sample获取ego_pose

        Args:
            sample: NuScenes sample数据

        Returns:
            ego_pose字典
        """
        # 从任意传感器获取ego_pose_token
        # 使用LIDAR_TOP作为参考
        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

        return ego_pose

    def _convert_pose(self, ego_pose: Dict[str, Any], timestamp: int) -> Dict[str, Any]:
        """
        将NuScenes的ego_pose转换为XVIZ格式

        NuScenes ego_pose格式:
        {
            'token': str,
            'translation': [x, y, z],  # 全局坐标，单位：米
            'rotation': [w, x, y, z],  # 四元数
            'timestamp': int            # 微秒
        }

        XVIZ pose格式:
        {
            'timestamp': float,              # 秒
            'position': [x, y, z],          # 米
            'orientation': [roll, pitch, yaw]  # 弧度
        }

        Args:
            ego_pose: NuScenes ego_pose
            timestamp: 时间戳（微秒）

        Returns:
            XVIZ格式的pose字典
        """
        # 提取位置（米）
        position = ego_pose['translation']

        # # 转换四元数为欧拉角
        # # NuScenes使用的是 [w, x, y, z] 格式
        # quaternion = Quaternion(ego_pose['rotation'])
        #
        # # 获取欧拉角 (yaw, pitch, roll)
        # # 注意：pyquaternion的yaw_pitch_roll返回顺序是 (yaw, pitch, roll)
        # yaw, pitch, roll = quaternion.yaw_pitch_roll

        quaternion = ego_pose['rotation']  # [w, x, y, z]
        rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        euler = rotation.as_euler('xyz', degrees=False)
        roll, pitch, yaw = euler
        # XVIZ期望的是 (roll, pitch, yaw) 顺序
        orientation = [roll, pitch, yaw]

        # 转换时间戳：微秒 -> 秒
        timestamp_seconds = timestamp / 1e6

        return {
            'timestamp': timestamp_seconds,
            'position': position,
            'orientation': orientation,
            'raw_quaternion': ego_pose['rotation']  # 保留原始数据供调试
        }

    def _add_trajectory(self, message_index: int, xviz_builder):
        """
        添加车辆轨迹线（显示过去和未来的路径）

        Args:
            message_index: 当前帧索引
            xviz_builder: XVIZ构建器
        """
        start_idx = 0
        end_idx = min(len(self.frames), message_index + 6)

        # 收集轨迹点
        trajectory_points = []
        for i in range(start_idx, end_idx):
            frame_token = self.frames[i]['token']
            pose = self.poses_by_frame[frame_token]
            position = pose['position']
            trajectory_points.extend(position)

        # 添加轨迹线到XVIZ
        if len(trajectory_points) > 0:
            xviz_builder.primitive(self.VEHICLE_TRAJECTORY) \
                .polyline(trajectory_points)

    def get_metadata(self, xviz_metadata):
        """
        定义pose流的元数据

        Args:
            xviz_metadata: XVIZ元数据构建器
        """
        # 定义vehicle_pose流
        # 注意：pose流的category必须是'pose'
        xviz_metadata.stream(self.VEHICLE_POSE) \
            .category('pose')

        # 如果启用了轨迹显示
        if self.config.get('show_vehicle_trajectory', False):
            xviz_metadata.stream(self.VEHICLE_TRAJECTORY) \
                .category('primitive') \
                .type('polyline') \
                .coordinate('IDENTITY')

    def get_poses(self) -> Dict[str, Dict]:
        """
        获取所有pose数据（供其他转换器使用）

        Returns:
            pose字典：{frame_token: pose_data}
        """
        return self.poses_by_frame


# 调试辅助函数
def debug_pose(ego_pose: Dict[str, Any]):
    """
    打印ego_pose的详细信息（用于调试）

    Args:
        ego_pose: NuScenes ego_pose
    """
    print("\n" + "=" * 60)
    print("Ego Pose Debug Info")
    print("=" * 60)

    print(f"Position: {ego_pose['translation']}")
    print(f"  X: {ego_pose['translation'][0]:.2f} m")
    print(f"  Y: {ego_pose['translation'][1]:.2f} m")
    print(f"  Z: {ego_pose['translation'][2]:.2f} m")

    q = Quaternion(ego_pose['rotation'])
    yaw, pitch, roll = q.yaw_pitch_roll

    print(f"\nRotation (Quaternion): {ego_pose['rotation']}")
    print(f"Rotation (Euler):")
    print(f"  Roll:  {np.degrees(roll):.2f}°")
    print(f"  Pitch: {np.degrees(pitch):.2f}°")
    print(f"  Yaw:   {np.degrees(yaw):.2f}°")

    print(f"\nTimestamp: {ego_pose['timestamp']}")
    print(f"  Seconds: {ego_pose['timestamp'] / 1e6:.3f}")

    print("=" * 60 + "\n")
