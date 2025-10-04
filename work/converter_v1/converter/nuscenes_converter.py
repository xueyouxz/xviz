import os
from typing import Dict, Any

from work.converter_v1.converter.annotation_converter import AnnotationConverter
from work.converter_v1.converter.camera_converter import CameraConverter
from work.converter_v1.converter.coordinate_converter import CoordinateConverter
from work.converter_v1.converter.future_object_converter import FutureAnnoConverter
from work.converter_v1.converter.lidar_converter import LidarConverter
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder


class NuscenesConverter:
    def __init__(self, scene_name, output_path, nuscenes, config):
        self.metadata = None
        self.converters = None
        self.timestamps = None
        self.frames = None
        self.scene_name = scene_name
        self.output_path = output_path
        self.nuscenes = nuscenes
        self.image_max_width = config['image_max_width']
        self.image_max_height = config['image_max_height']
        self.config = config
        self.sample_count = 0

    def load(self):
        scene = [s for s in self.nuscenes.scene if s['name'] == self.scene_name][0]
        samples = []
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = self.nuscenes.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']

        self.frames = []
        self.timestamps = []

        for sample in samples:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nuscenes.get('sample_data', lidar_token)

            frame_data = {
                'token': sample['token'],
                'sample': sample,
                'lidar_data': lidar_data,
                'timestamp': lidar_data['timestamp'] / 1e6,
                'sensors': self._get_frame_sensors(sample),
                'ego_pose_token': lidar_data['ego_pose_token']
            }

            self.frames.append(frame_data)
            self.timestamps.append(frame_data['timestamp'])
        self.sample_count = len(self.frames)

    def _get_frame_sensors(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sensors = {}
        for sensor_name, sample_data_token in sample['data'].items():
            sample_data = self.nuscenes.get('sample_data', sample_data_token)
            sensors[sensor_name] = sample_data
        return sensors

    def initialize(self):
        self.load()
        os.makedirs(self.output_path, exist_ok=True)

        coordinate_converter = CoordinateConverter(nuscenes=self.nuscenes, frames=self.frames)
        object_converter = AnnotationConverter(nuscenes=self.nuscenes, frames=self.frames)
        lidar_converter = LidarConverter(nuscenes=self.nuscenes, frames=self.frames)
        camera_converter = CameraConverter(nuscenes=self.nuscenes, frames=self.frames,
                                           image_max_width=self.image_max_width,
                                           image_max_height=self.image_max_height)
        future_object_converter = FutureAnnoConverter(nuscenes=self.nuscenes, frames=self.frames)
        self.converters = [coordinate_converter,
                           object_converter,
                           lidar_converter,
                           camera_converter,
                           future_object_converter]

        self.metadata = self.get_metadata()

    def get_metadata(self):
        xviz_metadata_builder = XVIZMetadataBuilder()
        xviz_metadata_builder.start_time(self.timestamps[0]).end_time(self.timestamps[-1])
        for converter in self.converters:
            converter.get_metadata(xviz_metadata_builder)
        return xviz_metadata_builder.get_message()

    def convert_message(self, sample_index):
        xviz_builder = XVIZBuilder(metadata=self.metadata)
        xviz_builder.timestamp(self.timestamps[sample_index])
        for converter in self.converters:
            converter.convert(sample_index, xviz_builder)
        return xviz_builder.get_message()
