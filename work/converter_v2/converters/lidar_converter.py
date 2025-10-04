"""
激光雷达数据转换器
将NuScenes的LiDAR点云数据转换为XVIZ格式
"""
import os
import numpy as np
from typing import Dict, Any, List
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


class LidarConverter:
    """
    激光雷达点云转换器

    NuScenes LiDAR数据格式:
    - 点云存储为二进制文件(.pcd.bin)
    - 每个点包含5个值: (x, y, z, intensity, ring_index)
    - 坐标系为传感器坐标系,需要转换到车辆坐标系
    """

    def __init__(self, nusc: NuScenes, config: Dict[str, Any]):
        """
        初始化激光雷达转换器

        Args:
            nusc: NuScenes数据集实例
            config: 配置字典
        """
        self.nusc = nusc
        self.config = config
        self.data_root = config['data_root']

        # XVIZ流名称
        self.LIDAR_POINTS = '/lidar/points'

        # 传感器名称
        self.sensor_name = 'LIDAR_TOP'

        # 缓存传感器标定信息
        self.calibration = None

    def load(self, frames: List[Dict[str, Any]]):
        """
        加载激光雷达数据

        Args:
            frames: 所有帧的信息列表
        """
        self.frames = frames

        # 获取传感器标定信息(所有帧共享同一个标定)
        if len(frames) > 0:
            first_frame = frames[0]
            sample_data_token = first_frame['data'][self.sensor_name]
            sample_data = self.nusc.get('sample_data', sample_data_token)

            # 获取标定传感器信息
            cs_record = self.nusc.get('calibrated_sensor',
                                      sample_data['calibrated_sensor_token'])
            self.calibration = {
                'translation': cs_record['translation'],
                'rotation': cs_record['rotation']
            }

    def convert_message(self, message_index: int, xviz_builder, frame: Dict[str, Any]):
        """
        转换单帧激光雷达数据为XVIZ格式

        Args:
            message_index: 帧索引
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        # 获取激光雷达的sample_data token
        sample_data_token = frame['data'][self.sensor_name]
        sample_data = self.nusc.get('sample_data', sample_data_token)

        # 构建点云文件完整路径
        pcl_path = os.path.join(self.data_root, sample_data['filename'])

        # 检查文件是否存在
        if not os.path.exists(pcl_path):
            print(f"Warning: LiDAR file not found: {pcl_path}")
            return

        # 读取并解析点云数据
        point_cloud = self._load_point_cloud(pcl_path)

        if point_cloud is None or len(point_cloud['points']) == 0:
            return

        xviz_builder.primitive(self.LIDAR_POINTS) \
            .points(point_cloud['points']) \
            .colors(point_cloud['colors'])

    def get_metadata(self, xviz_metadata):
        """
        定义激光雷达流的元数据

        Args:
            xviz_metadata: XVIZ元数据构建器
        """
        # 从标定信息中提取位姿
        translation = self.calibration['translation']
        quaternion = self.calibration['rotation']
        rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
        # XVIZ期望的是 (roll, pitch, yaw) 顺序
        orientation = [roll, pitch, yaw]


        # 定义激光雷达流
        xviz_metadata.stream(self.LIDAR_POINTS) \
            .category('primitive') \
            .type('point') \
            .coordinate('VEHICLE_RELATIVE') \
            .pose(translation, orientation) \
            .stream_style({
            'fill_color': [255, 0, 0],
            'radius_pixels': 1
        })

    def _load_point_cloud(self, pcl_path: str) -> Dict[str, np.ndarray]:
        """
        加载并解析点云文件

        NuScenes点云格式:
        - 二进制文件,每个点5个float32值
        - x, y, z: 3D坐标 (米)
        - intensity: 反射强度 (0-255)
        - ring_index: 激光环索引

        Args:
            pcl_path: 点云文件路径

        Returns:
            字典包含points(Nx3)和colors(Nx4)数组
        """

        # 读取二进制点云数据
        # 数据类型: float32, 每个点5个值
        points_data = np.fromfile(pcl_path, dtype=np.float32)

        # 重塑为(N, 5)数组
        points_data = points_data.reshape(-1, 5)

        num_points = points_data.shape[0]

        # 提取xyz坐标(前3列)
        # XVIZ需要展平的数组: [x1,y1,z1,x2,y2,z2,...]
        points = points_data[:, :3].flatten().astype(np.float32)

        # 根据强度值生成颜色
        # intensity范围通常是0-255
        intensity = points_data[:, 3]

        # 归一化强度值到0-1
        intensity_normalized = np.clip(intensity / 255.0, 0, 1)

        # 创建RGBA颜色数组
        # 使用蓝色调色板,强度越高越亮
        colors = np.zeros((num_points, 4), dtype=np.uint8)
        colors[:, 0] = 80 + intensity_normalized * 80  # R
        colors[:, 1] = 80 + intensity_normalized * 80  # G
        colors[:, 2] = 80 + intensity_normalized * 60  # B
        colors[:, 3] = 255  # Alpha

        # 展平颜色数组
        colors = colors.flatten()

        return {
            'points': points,
            'colors': colors
        }


