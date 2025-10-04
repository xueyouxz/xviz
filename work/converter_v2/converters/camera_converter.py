"""
相机图像转换器
将NuScenes的6个相机图像转换为XVIZ格式
"""
import os
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Dict, Any, List
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class CameraConverter:
    """
    相机图像转换器

    NuScenes有6个相机:
    - CAM_FRONT: 前向相机
    - CAM_FRONT_LEFT: 左前相机
    - CAM_FRONT_RIGHT: 右前相机
    - CAM_BACK: 后向相机
    - CAM_BACK_LEFT: 左后相机
    - CAM_BACK_RIGHT: 右后相机
    """

    def __init__(self, nusc: NuScenes, config: Dict[str, Any]):
        """
        初始化相机转换器

        Args:
            nusc: NuScenes数据集实例
            config: 配置字典
        """
        self.nusc = nusc
        self.config = config
        self.data_root = config['data_root']

        # 图像缩放参数(减小数据量)
        self.image_max_width = config.get('image_max_width', 400)
        self.image_max_height = config.get('image_max_height', 300)

        # 所有相机通道名称
        self.camera_channels = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]

        # 为每个相机创建XVIZ流名称
        self.camera_streams = {
            cam: f'/camera/{cam.lower()}'
            for cam in self.camera_channels
        }

        # 缓存每个相机的标定信息
        self.calibrations = {}

    def load(self, frames: List[Dict[str, Any]]):
        """
        加载相机数据和标定信息

        Args:
            frames: 所有帧的信息列表
        """
        self.frames = frames

        # 获取所有相机的标定信息
        if len(frames) > 0:
            first_frame = frames[0]

            for cam_channel in self.camera_channels:
                sample_data_token = first_frame['data'][cam_channel]
                sample_data = self.nusc.get('sample_data', sample_data_token)

                # 获取标定传感器信息
                cs_record = self.nusc.get('calibrated_sensor',
                                          sample_data['calibrated_sensor_token'])

                self.calibrations[cam_channel] = {
                    'translation': cs_record['translation'],
                    'rotation': cs_record['rotation'],
                    'camera_intrinsic': cs_record['camera_intrinsic']
                }

    def convert_message(self, message_index: int, xviz_builder, frame: Dict[str, Any]):
        """
        转换单帧所有相机图像为XVIZ格式

        Args:
            message_index: 帧索引
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        # 遍历所有相机
        for cam_channel in self.camera_channels:
            self._convert_camera(cam_channel, xviz_builder, frame)

    def _convert_camera(self, cam_channel: str, xviz_builder, frame: Dict[str, Any]):
        """
        转换单个相机的图像

        Args:
            cam_channel: 相机通道名称
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        # 获取相机的sample_data token
        sample_data_token = frame['data'][cam_channel]
        sample_data = self.nusc.get('sample_data', sample_data_token)

        # 构建图像文件完整路径
        img_path = os.path.join(self.data_root, sample_data['filename'])

        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            return

        # 加载并处理图像
        image_data, width, height = self._load_and_process_image(img_path)

        if image_data is None:
            return

        # 将图像添加到XVIZ
        # XVIZ支持多种图像格式,这里使用JPEG
        stream_name = self.camera_streams[cam_channel]
        xviz_builder.primitive(stream_name).image(image_data).dimensions(width, height)

    def _load_and_process_image(self, img_path: str) -> tuple:
        """
        加载并处理图像

        处理步骤:
        1. 加载图像
        2. 调整大小(如果配置了最大尺寸)
        3. 转换为JPEG格式
        4. 返回字节数据

        Args:
            img_path: 图像文件路径

        Returns:
            元组: (JPEG格式的图像字节数据, 宽度, 高度)
        """

        # 使用PIL加载图像
        img = Image.open(img_path)

        # 如果设置了最大尺寸,调整图像大小
        if self.image_max_width > 0 and self.image_max_height > 0:
            # 保持宽高比缩放
            img.thumbnail(
                (self.image_max_width, self.image_max_height),
                Image.Resampling.LANCZOS
            )

        # 转换为JPEG格式并保存到字节缓冲区
        buffer = BytesIO()

        # 如果是PNG或其他带alpha通道的格式,转换为RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            # 创建白色背景
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # 获取最终图像尺寸
        width, height = img.size

        # 保存为JPEG,质量设为85
        img.save(buffer, format='JPEG', quality=85)

        # 获取字节数据
        image_bytes = buffer.getvalue()

        return image_bytes, width, height

    def get_metadata(self, xviz_metadata):
        """
        定义所有相机流的元数据

        Args:
            xviz_metadata: XVIZ元数据构建器
        """
        for cam_channel in self.camera_channels:
            stream_name = self.camera_streams[cam_channel]
            calibration = self.calibrations[cam_channel]

            # 提取位姿信息
            translation = calibration['translation']
            rotation = calibration['rotation']

            # 将四元数转换为欧拉角
            q = Quaternion(rotation)
            euler = self._quaternion_to_euler(q)

            # 定义相机流
            xviz_metadata.stream(stream_name) \
                .category('primitive') \
                .type('image') \
                .coordinate('VEHICLE_RELATIVE') \
                .pose(translation, euler)

            # 添加相机内参(可选)
            # camera_intrinsic = calibration['camera_intrinsic']
            # xviz_metadata.stream_metadata(stream_name, {
            #     'camera_intrinsic': camera_intrinsic
            # })

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
