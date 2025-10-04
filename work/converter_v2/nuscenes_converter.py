"""
NuScenes到XVIZ协议转换器主类
负责协调各个子转换器,管理转换流程
"""
import os
from typing import Dict, Any, List
from nuscenes.nuscenes import NuScenes

from work.converter_v2.converters.pose_converter import PoseConverter
from xviz_avs.builder import XVIZBuilder, XVIZMetadataBuilder

from converters.lidar_converter import LidarConverter
from converters.camera_converter import CameraConverter
from converters.annotation_converter import AnnotationConverter
from converters.furture_anno_converter import FutureAnnoConverter
from converters.radar_converter import RadarConverter
from xviz_avs.message import XVIZMessage


class NuscenesConverter:
    """
    NuScenes数据集到XVIZ协议的转换器

    主要功能:
    1. 初始化各个数据源的转换器
    2. 加载场景中的所有帧数据
    3. 生成XVIZ元数据
    4. 逐帧转换数据为XVIZ消息
    """

    def __init__(self, scene_name: str, output_path: str, nuscenes: NuScenes, config: Dict[str, Any]):
        """
        初始化转换器

        Args:
            scene_name: 场景名称
            output_path: 输出路径
            nuscenes: NuScenes数据集实例
            config: 配置字典
        """
        self.scene_name = scene_name
        self.output_path = output_path
        self.nusc = nuscenes
        self.config = config

        # 场景和帧数据
        self.scene = None
        self.frames = []  # 存储所有关键帧的token和时间戳
        self.sample_count = 0

        # 初始化各个转换器
        self.pose_converter = PoseConverter(
            nusc=self.nusc,
            config=config
        )
        # 激光雷达转换器
        self.lidar_converter = LidarConverter(
            nusc=self.nusc,
            config=config
        )

        # 相机转换器(6个相机)
        self.camera_converter = CameraConverter(
            nusc=self.nusc,
            config=config
        )

        # 毫米波雷达转换器
        self.radar_converter = RadarConverter(
            nusc=self.nusc,
            config=config
        )

        # 标注对象转换器(包含3D边界框)
        self.annotation_converter = AnnotationConverter(
            nusc=self.nusc,
            config=config
        )

        # 未来轨迹转换器(预测未来3秒的位置)
        self.future_object_converter = FutureAnnoConverter(
            nusc=self.nusc,
            config=config
        )

        # 转换器列表,顺序很重要
        self.converters = [
            self.pose_converter,
            self.lidar_converter,
            self.camera_converter,
            # self.radar_converter,
            self.annotation_converter,
            self.future_object_converter
        ]

    def initialize(self):
        """
        初始化转换器
        1. 加载场景数据
        2. 加载所有帧的信息
        3. 初始化各个子转换器
        """
        print(f"Initializing converter for scene: {self.scene_name}")

        # 加载场景
        self._load_scene()

        # 加载所有关键帧
        self._load_frames()

        # 初始化各个转换器
        for converter in self.converters:
            converter.load(self.frames)

        print(f"Loaded {self.sample_count} frames")

    def _load_scene(self):
        """
        根据场景名称加载场景数据
        """
        # 在所有场景中查找匹配的场景
        for scene in self.nusc.scene:
            if scene['name'] == self.scene_name:
                self.scene = scene
                break

        if self.scene is None:
            raise ValueError(f"Scene {self.scene_name} not found")

    def _load_frames(self):
        """
        加载场景中的所有关键帧

        NuScenes数据结构:
        - scene: 包含first_sample_token和last_sample_token
        - sample: 关键帧,包含时间戳和各传感器数据的token
        - sample_data: 具体的传感器数据
        """
        self.frames = []

        # 获取第一个关键帧
        sample_token = self.scene['first_sample_token']

        # 遍历所有关键帧
        while sample_token:
            sample = self.nusc.get('sample', sample_token)

            # 构建帧数据结构
            frame_data = {
                'token': sample_token,
                'timestamp': sample['timestamp'],  # 微秒级时间戳
                'scene_token': sample['scene_token'],
                'data': sample['data']  # 包含所有传感器的sample_data token
            }

            self.frames.append(frame_data)

            # 移动到下一个关键帧
            sample_token = sample['next']

        self.sample_count = len(self.frames)

    def get_metadata(self) -> XVIZMessage:
        """
        生成XVIZ元数据

        元数据包含:
        1. 所有数据流的定义(类型、坐标系、样式等)
        2. 时间范围
        3. UI配置
        4. 数据集信息

        Returns:
            XVIZ元数据字典
        """
        # 创建元数据构建器
        xviz_metadata = XVIZMetadataBuilder()

        # 设置时间范围(转换为秒)
        start_time = self.frames[0]['timestamp'] / 1e6
        end_time = self.frames[-1]['timestamp'] / 1e6
        xviz_metadata.start_time(start_time).end_time(end_time)

        # 让各个转换器添加他们的流元数据
        for converter in self.converters:
            converter.get_metadata(xviz_metadata)

        # # # 添加数据集信息
        # xviz_metadata.log_info({
        #     'scene': self.scene_name,
        #     'license': 'CC BY-NC-SA 4.0',
        #     'source': {
        #         'title': 'nuScenes dataset',
        #         'link': 'https://www.nuscenes.org/'
        #     }
        # })

        # 添加UI配置(可选)
        self._add_ui_config(xviz_metadata)

        return xviz_metadata.get_message()

    def _add_ui_config(self, xviz_metadata: XVIZMetadataBuilder):
        """
        添加UI配置,定义可视化界面的布局和面板

        Args:
            xviz_metadata: XVIZ元数据构建器
        """
        # 定义视频面板配置
        video_config = {
            'cameras': [
                'CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT'
            ]
        }

        # 可以在这里添加更多UI配置
        # xviz_metadata.ui({...})

    def convert_message(self, message_index: int) -> XVIZMessage:
        """
        转换指定索引的帧为XVIZ消息

        Args:
            message_index: 帧索引

        Returns:
            XVIZ消息字典
        """
        if message_index >= self.sample_count:
            return None

        frame = self.frames[message_index]

        # 创建XVIZ构建器
        xviz_builder = XVIZBuilder(metadata=self.get_metadata())

        for converter in self.converters:
            converter.convert_message(message_index, xviz_builder, frame)

        # 构建并返回消息
        return xviz_builder.get_message()
