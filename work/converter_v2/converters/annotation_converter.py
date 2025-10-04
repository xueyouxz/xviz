"""
标注对象转换器
将NuScenes的3D边界框标注转换为XVIZ格式
包含对象的历史轨迹(过去3秒)
"""
import numpy as np
from typing import Dict, Any, List
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class AnnotationConverter:
    """
    标注对象转换器

    将NuScenes的3D检测框转换为XVIZ的3D边界框
    同时提取对象的历史轨迹
    """

    def __init__(self, nusc: NuScenes, config: Dict[str, Any]):
        """
        初始化标注转换器

        Args:
            nusc: NuScenes数据集实例
            config: 配置字典
        """
        self.nusc = nusc
        self.config = config

        # XVIZ流名称
        self.OBJECTS_STREAM = '/object/boxes'
        self.TRAJECTORY_STREAM = '/object/trajectory'

        # NuScenes类别到可视化颜色的映射
        self.CATEGORY_COLORS = {
            # 车辆类
            'vehicle.car': [0, 255, 0],  # 绿色
            'vehicle.truck': [0, 200, 0],  # 深绿色
            'vehicle.bus': [0, 150, 0],  # 更深绿色
            'vehicle.trailer': [0, 100, 0],
            'vehicle.construction': [100, 150, 0],
            'vehicle.emergency': [255, 0, 0],  # 红色
            'vehicle.motorcycle': [0, 255, 255],  # 青色
            'vehicle.bicycle': [0, 200, 255],

            # 行人类
            'human.pedestrian': [255, 255, 0],  # 黄色

            # 其他
            'movable_object.barrier': [128, 128, 128],  # 灰色
            'movable_object.trafficcone': [255, 128, 0],  # 橙色
            'movable_object.pushable_pullable': [200, 200, 200],
            'movable_object.debris': [150, 150, 150],

            # 静态物体
            'static_object.bicycle_rack': [100, 100, 100],
        }

        # 默认颜色
        self.DEFAULT_COLOR = [128, 128, 255]  # 浅蓝色

    def load(self, frames: List[Dict[str, Any]]):
        """
        加载标注数据

        Args:
            frames: 所有帧的信息列表
        """
        self.frames = frames

        # 构建对象轨迹字典: {instance_token: [历史位置列表]}
        self.object_trajectories = self._build_object_trajectories()

    def _build_object_trajectories(self) -> Dict[str, List[Dict]]:
        """
        构建所有对象的历史轨迹

        遍历所有帧,记录每个对象实例在不同时刻的位置

        Returns:
            字典: {instance_token: [{'timestamp': ts, 'translation': xyz, 'rotation': q}]}
        """
        trajectories = {}

        for frame in self.frames:
            sample = self.nusc.get('sample', frame['token'])

            # 遍历该帧的所有标注
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']

                # 如果是该对象的第一次出现,初始化列表
                if instance_token not in trajectories:
                    trajectories[instance_token] = []

                # 添加该时刻的位置信息
                trajectories[instance_token].append({
                    'timestamp': frame['timestamp'],
                    'translation': ann['translation'],
                    'rotation': ann['rotation'],
                    'size': ann['size']
                })

        # 按时间戳排序每个对象的轨迹
        for instance_token in trajectories:
            trajectories[instance_token].sort(key=lambda x: x['timestamp'])

        return trajectories

    def convert_message(self, message_index: int, xviz_builder, frame: Dict[str, Any]):
        """
        转换单帧的标注数据为XVIZ格式

        Args:
            message_index: 帧索引
            xviz_builder: XVIZ构建器
            frame: 当前帧数据
        """
        sample = self.nusc.get('sample', frame['token'])
        current_timestamp = frame['timestamp']

        # 获取车辆位姿(用于坐标转换)
        ego_pose = self._get_ego_pose(sample)

        # 遍历所有标注对象
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # 转换3D边界框
            self._convert_bbox(ann, ego_pose, xviz_builder)

            # 提取并转换历史轨迹(过去3秒)
            self._convert_trajectory(
                ann,
                current_timestamp,
                ego_pose,
                xviz_builder
            )

    def _convert_bbox(self, ann: Dict, ego_pose: Dict, xviz_builder):
        """
        转换单个3D边界框

        Args:
            ann: 标注信息
            ego_pose: 车辆位姿
            xviz_builder: XVIZ构建器
        """
        # 提取边界框信息
        center = np.array(ann['translation'])  # 中心点(全局坐标)
        size = ann['size']  # [width, length, height]
        rotation = Quaternion(ann['rotation'])  # 旋转四元数

        # 转换到车辆坐标系
        # center_vehicle = self._global_to_vehicle(center, ego_pose)

        # 计算车辆坐标系下的旋转
        # ego_rotation = Quaternion(ego_pose['rotation'])
        # rotation_vehicle = ego_rotation.inverse * rotation
        # yaw = rotation_vehicle.yaw_pitch_roll[0]

        # 获取类别和颜色
        category = ann['category_name']

        # 获取对象ID和属性
        instance_token = ann['instance_token']
        instance = self.nusc.get('instance', instance_token)

        # 构建对象信息
        object_id = instance_token  # 使用instance_token作为唯一ID

        # 添加3D边界框到XVIZ
        # XVIZ的边界框定义: [中心x, 中心y, 中心z, x尺寸, y尺寸, z尺寸, yaw]
        xviz_builder.primitive(self.OBJECTS_STREAM) \
            .polygon([
            center_vehicle[0], center_vehicle[1], center_vehicle[2],
            size[0], size[1], size[2],  # width, length, height
            yaw
        ]) \
            .id(object_id)


    def _convert_trajectory(self, ann: Dict, current_timestamp: int,
                            ego_pose: Dict, xviz_builder):
        """
        转换对象的历史轨迹(过去3秒)

        Args:
            ann: 当前帧的标注信息
            current_timestamp: 当前时间戳(微秒)
            ego_pose: 当前车辆位姿
            xviz_builder: XVIZ构建器
        """
        instance_token = ann['instance_token']

        # 获取该对象的完整轨迹
        if instance_token not in self.object_trajectories:
            return

        trajectory = self.object_trajectories[instance_token]

        # 筛选过去3秒的轨迹点
        # 3秒 = 3,000,000微秒
        time_window = 3_000_000
        past_trajectory = [
            point for point in trajectory
            if current_timestamp >= point['timestamp'] >= current_timestamp - time_window
        ]

        if len(past_trajectory) < 2:
            # 轨迹点太少,不绘制
            return

        # 转换轨迹点到车辆坐标系
        trajectory_points = []
        for point in past_trajectory:
            global_pos = np.array(point['translation'])
            vehicle_pos = self._global_to_vehicle(global_pos, ego_pose)
            trajectory_points.extend(vehicle_pos.tolist())

        # 添加轨迹线到XVIZ
        xviz_builder.primitive(self.TRAJECTORY_STREAM) \
            .polyline(trajectory_points) \
            .id(instance_token) \


    def _get_ego_pose(self, sample: Dict) -> Dict:
        """
        获取车辆在全局坐标系的位姿

        Args:
            sample: 样本数据

        Returns:
            位姿字典,包含translation和rotation
        """
        # 从任意传感器获取ego_pose_token
        # 这里使用LIDAR_TOP
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
            point: 全局坐标点[x, y, z]
            ego_pose: 车辆位姿

        Returns:
            车辆坐标系下的点[x, y, z]
        """
        # 提取车辆位置和旋转
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        # 平移到车辆原点
        point_centered = point - ego_translation

        # 旋转到车辆坐标系(逆旋转)
        point_vehicle = ego_rotation.inverse.rotate(point_centered)

        return point_vehicle

    def _get_color_for_category(self, category: str) -> List[int]:
        """
        根据类别获取颜色

        Args:
            category: 类别名称

        Returns:
            RGBA颜色数组
        """
        return self.CATEGORY_COLORS.get(category, self.DEFAULT_COLOR)

    def get_metadata(self, xviz_metadata):
        """
        定义标注对象流的元数据

        Args:
            xviz_metadata: XVIZ元数据构建器
        """
        # 定义3D边界框流
        xviz_metadata.stream(self.OBJECTS_STREAM) \
            .category('primitive') \
            .type('polygon') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
            'extruded': True,
            'fill_color': [0, 255, 0],
            'stroke_color': [0, 255, 0],
            'stroke_width': 0.2
        })

        # 定义轨迹流
        xviz_metadata.stream(self.TRAJECTORY_STREAM) \
            .category('primitive') \
            .type('polyline') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
            'stroke_color': [255, 255, 255],
            'stroke_width': 0.1
        })