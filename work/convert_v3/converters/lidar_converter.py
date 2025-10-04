import os
import numpy as np
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation
from nuscenes.nuscenes import NuScenes
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder


class LidarConverter:
    def __init__(self, nuscenes: NuScenes, frames: List[Dict[str, Any]]):
        self.nuscenes = nuscenes
        self.frames = frames
        self.LIDAR_POINTS = '/lidar/points'

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        xmb.stream(self.LIDAR_POINTS) \
            .category('primitive') \
            .type('point') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
            'fill_color': [40, 0, 170, 255],
            'radius_pixels': 1
        })

    def convert(self, frame_index: int, xb: XVIZBuilder):
        frame = self.frames[frame_index]
        lidar_data = frame['lidar_data']
        lidar_file = os.path.join(self.nuscenes.dataroot, lidar_data['filename'])

        if not os.path.exists(lidar_file):
            print(f"警告: 激光雷达文件不存在：{lidar_file}")
            return

        points_data = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)

        points = points_data[:, :3]
        intensities = points_data[:, 3]

        calibrated_sensor_token = lidar_data['calibrated_sensor_token']
        calibrated_sensor = self.nuscenes.get('calibrated_sensor', calibrated_sensor_token)
        
        sensor_rotation = Rotation.from_quat([
            calibrated_sensor['rotation'][1],
            calibrated_sensor['rotation'][2],
            calibrated_sensor['rotation'][3],
            calibrated_sensor['rotation'][0]
        ])
        sensor_translation = np.array(calibrated_sensor['translation'])
        
        points_vehicle = sensor_rotation.apply(points) + sensor_translation

        colors = np.zeros((len(points), 4), dtype=np.uint8)
        normalized_intensity = intensities / 255.0
        colors[:, 0] = (80 + normalized_intensity * 80).astype(np.uint8)
        colors[:, 1] = (80 + normalized_intensity * 80).astype(np.uint8)
        colors[:, 2] = (80 + normalized_intensity * 60).astype(np.uint8)
        colors[:, 3] = 255

        xb.primitive(self.LIDAR_POINTS) \
            .points(points_vehicle.flatten().tolist()) \
            .colors(colors.flatten().tolist())

