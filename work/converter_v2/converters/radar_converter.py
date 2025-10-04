"""
毫米波雷达数据转换器
将NuScenes的雷达点云数据转换为XVIZ格式
"""
import os
import numpy as np
from typing import Dict, Any, List
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class RadarConverter:
    """
    毫米波雷达转换器

    NuScenes有5个雷达传感器:
    - RADAR_FRONT
    - RADAR_FRONT_LEFT
    - RADAR_FRONT_RIGHT
    - RADAR_BACK_LEFT
    - RADAR_BACK_RIGHT

    雷达点云格式:
    - 18个值: x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp,
             is_quality_valid, ambig_state, x_rms, y_rms, invalid_state,
             pdh0, vx_rms, vy_rms
    """

    def __init__(self, nusc: NuScenes, config: Dict[str, Any]):
        """
        初始化雷达转换器

        Args:
            nusc: NuScenes数据集实例
            config: 配置字典
        """
        self.nusc = nusc
        self.config = config
        self.data_root = config['data_root']

        # 所有雷达通道
        self.radar_channels = [
            'RADAR_FRONT',
            'RADAR_FRONT_LEFT',
            'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT',
            'RADAR_BACK_RIGHT'
        ]

        # 为每个雷达创建XVIZ流名称
        self.radar_streams = {
            radar: f'/radar/{radar.lower()}'
            for radar in self.radar_channels
        }

        # 缓存每个雷达的标定信息
        self.calibrations = {}

    def load(self, frames: List[Dict[str, Any]]):
        """
        加载雷达数据和标定信息

        Args:
            frames: 所有帧的信息列表
        """
        self.frames = frames

        # 获取所有雷达的标定信息
        if len(frames) > 0:
            first_frame = frames[0]

            for radar_channel in self.radar_channels:
                sample_data_token = first_frame['data'][radar_channel]
                sample_data = self.nusc.get('sample_data', sample_data_token)

                # 获取标定传感器信息
                cs_record = self.nusc.get('calibrated_sensor',
                                          sample_data['calibrated_sensor_token'])

                self.calibrations[radar_channel] = {
                    'translation': cs_record['translation'],
                    'rotation': cs_record['rotation']
                }

    def convert_message(self, message_index: int, xviz_builder, frame: Dict[str, Any]):
        """
        转换单帧所有雷达数据为XVIZ格式

        Args:
            message_index: 帧索引
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        # 遍历所有雷达
        for radar_channel in self.radar_channels:
            try:
                self._convert_radar(radar_channel, xviz_builder, frame)
            except Exception as e:
                print(f"Error converting radar {radar_channel} "
                      f"at frame {message_index}: {str(e)}")

    def _convert_radar(self, radar_channel: str, xviz_builder, frame: Dict[str, Any]):
        """
        转换单个雷达的点云数据

        Args:
            radar_channel: 雷达通道名称
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        # 获取雷达的sample_data token
        sample_data_token = frame['data'][radar_channel]
        sample_data = self.nusc.get('sample_data', sample_data_token)

        # 构建雷达数据文件完整路径
        radar_path = os.path.join(self.data_root, sample_data['filename'])

        # 检查文件是否存在
        if not os.path.exists(radar_path):
            print(f"Warning: Radar file not found: {radar_path}")
            return

        # 读取并解析雷达点云数据
        radar_data = self._load_radar_pointcloud(radar_path)

        if radar_data is None or len(radar_data['points']) == 0:
            return

        # 将雷达点云数据添加到XVIZ
        stream_name = self.radar_streams[radar_channel]
        xviz_builder.primitive(stream_name) \
            .points(radar_data['points']) \
            .colors(radar_data['colors'])

    def _load_radar_pointcloud(self, radar_path: str) -> Dict[str, np.ndarray]:
        """
        加载并解析雷达点云文件

        雷达点云格式:
        - 每个点18个float32值
        - 主要使用: x, y, z (位置), vx, vy (速度), rcs (雷达截面)

        Args:
            radar_path: 雷达数据文件路径

        Returns:
            字典包含points(Nx3)和colors(Nx4)数组
        """
        try:
            # 读取二进制雷达数据
            radar_data = np.fromfile(radar_path, dtype=np.float32)

            # 重塑为(N, 18)数组
            radar_data = radar_data.reshape(-1, 18)

            num_points = radar_data.shape[0]

            # 提取位置信息(前3列: x, y, z)
            positions = radar_data[:, :3]

            # 展平为XVIZ格式: [x1,y1,z1,x2,y2,z2,...]
            points = positions.flatten().astype(np.float32)

            # 提取速度和RCS信息用于着色
            # vx_comp, vy_comp (索引8, 9): 补偿后的速度
            # rcs (索引5): 雷达截面,反映目标的反射强度
            vx_comp = radar_data[:, 8]
            vy_comp = radar_data[:, 9]
            rcs = radar_data[:, 5]

            # 计算速度大小
            velocity = np.sqrt(vx_comp ** 2 + vy_comp ** 2)

            # 根据速度和RCS生成颜色
            # 速度越大,颜色越偏红;RCS越大,亮度越高
            colors = self._generate_radar_colors(velocity, rcs, num_points)

            return {
                'points': points,
                'colors': colors
            }

        except Exception as e:
            print(f"Error loading radar data {radar_path}: {str(e)}")
            return None

    def _generate_radar_colors(self, velocity: np.ndarray, rcs: np.ndarray,
                               num_points: int) -> np.ndarray:
        """
        根据速度和RCS生成雷达点的颜色

        着色策略:
        - 速度大: 偏红色(运动目标)
        - 速度小: 偏蓝色(静止目标)
        - RCS大: 更亮(强反射)

        Args:
            velocity: 速度数组
            rcs: 雷达截面数组
            num_points: 点数量

        Returns:
            RGBA颜色数组(Nx4,展平)
        """
        # 归一化速度(0-1范围)
        # 通常车辆速度在0-30 m/s范围
        velocity_normalized = np.clip(velocity / 30.0, 0, 1)

        # 归一化RCS(对数尺度)
        # RCS可能跨越很大范围,使用对数归一化
        rcs_db = 10 * np.log10(np.maximum(rcs, 1e-10))
        rcs_normalized = np.clip((rcs_db + 10) / 30.0, 0, 1)

        # 创建RGBA颜色数组
        colors = np.zeros((num_points, 4), dtype=np.uint8)

        # R通道: 由速度决定,速度越大越红
        colors[:, 0] = (50 + velocity_normalized * 205).astype(np.uint8)

        # G通道: 中等亮度
        colors[:, 1] = (50 + rcs_normalized * 100).astype(np.uint8)

        # B通道: 由速度反向决定,速度越小越蓝
        colors[:, 2] = (50 + (1 - velocity_normalized) * 205).astype(np.uint8)

        # Alpha通道: 由RCS决定,反射强度越大越不透明
        colors[:, 3] = (100 + rcs_normalized * 155).astype(np.uint8)

        # 展平数组
        return colors.flatten()

    def get_metadata(self, xviz_metadata):
        """
        定义所有雷达流的元数据

        Args:
            xviz_metadata: XVIZ元数据构建器
        """
        for radar_channel in self.radar_channels:
            stream_name = self.radar_streams[radar_channel]
            calibration = self.calibrations[radar_channel]

            # 提取位姿信息
            translation = calibration['translation']
            rotation = calibration['rotation']

            # 将四元数转换为欧拉角
            q = Quaternion(rotation)
            euler = self._quaternion_to_euler(q)

            # 定义雷达流
            xviz_metadata.stream(stream_name) \
                .category('primitive') \
                .type('point') \
                .coordinate('VEHICLE_RELATIVE') \
                .pose(translation, euler) \
                .stream_style({
                'fill_color': [255, 0, 0],  # 粉红色基础色
                'radius_pixels': 2  # 稍大的点
            })

    def _quaternion_to_euler(self, q: Quaternion) -> tuple:
        """
        将四元数转换为欧拉角(roll, pitch, yaw)

        Args:
            q: 四元数

        Returns:
            (roll, pitch, yaw)元组,单位为弧度
        """
        yaw, pitch, roll = q.yaw_pitch_roll
        return (roll, pitch, yaw)