import os

import numpy as np
from nuscenes import NuScenes

from xviz_avs import XVIZBuilder, XVIZMetadataBuilder
from work.converter_v1.utils import quaternion_to_euler_angle


class LidarConverter:

    def __init__(self, nuscenes: NuScenes,frames):
        self.frames = frames
        self.nuscenes = nuscenes
        self.LIDAR_POINTS = '/lidar'

    def convert(self, frame_index, xb: XVIZBuilder):
        frame = self.frames[frame_index]
        lidar_data = frame['lidar_data']
        lidar_file = os.path.join(self.nuscenes.dataroot, lidar_data['filename'])
        points_data = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)

        # 提取坐标和强度
        points = points_data[:, :3]  # x, y, z
        intensities = points_data[:, 3]  # intensity
        # 构建点云
        xb.primitive(self.LIDAR_POINTS) \
            .points(points.flatten().tolist()) \
            # .colors(colors.flatten().tolist())

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        lidar_sensor = self.frames[0]['sensors']['LIDAR_TOP']
        lidar_calibrated_sensor = self.nuscenes.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
        translation = lidar_calibrated_sensor['translation']
        orientation = quaternion_to_euler_angle(lidar_calibrated_sensor['rotation'])
        xmb.stream(self.LIDAR_POINTS) \
            .category('primitive') \
            .type('point') \
            .coordinate('VEHICLE_RELATIVE') \
            .pose(translation, orientation)
