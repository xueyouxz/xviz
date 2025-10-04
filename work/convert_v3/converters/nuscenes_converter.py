import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes

from work.convert_v3.converters.map_converter import MapConverter
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder
from xviz_avs.io import XVIZGLBWriter, DirectorySource
from tqdm import tqdm

from work.convert_v3.converters.pose_converter import PoseConverter
from work.convert_v3.converters.lidar_converter import LidarConverter
from work.convert_v3.converters.annotation_converter import AnnotationConverter
from work.convert_v3.converters.camera_converter import CameraConverter


class NuScenesConverter:
    def __init__(self, nuscenes_root: str, output_dir: str, version: str = 'v1.0-mini',
                 image_max_width: int = 400, image_max_height: int = 300, nuscenes_obj: Optional[Any] = None,
                 map_explorer: Optional[Any] = None):
        self.nuscenes_root = Path(nuscenes_root)
        self.output_dir = Path(output_dir)
        self.version = version
        self.map_explorer = map_explorer
        self.image_max_width = image_max_width
        self.image_max_height = image_max_height

        self.data_root = self.nuscenes_root
        self.dataroot = str(self.data_root)

        self.frames = []
        self.timestamps = []
        self.metadata = None
        self.converters = []

        self.nusc = nuscenes_obj

    def load_nuscenes_data(self, scene_name: Optional[str] = None):
        if self.nusc is None:
            self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=True)

        if scene_name:
            scenes = [s for s in self.nusc.scene if s['name'] == scene_name]
            if not scenes:
                raise ValueError(f"未找到场景：{scene_name}")
            scene = scenes[0]
            self._load_scene_data(scene)

    def _load_scene_data(self, scene: Dict[str, Any]):
        self.current_scene = scene

        samples = []
        sample_token = scene['first_sample_token']

        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']

        self.frames = []
        self.timestamps = []

        for sample in samples:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)

            frame_data = {
                'sample': sample,
                'lidar_data': lidar_data,
                'timestamp': lidar_data['timestamp'] / 1e6,
                'sensors': self._get_frame_sensors(sample),
                'ego_pose_token': lidar_data['ego_pose_token']
            }

            self.frames.append(frame_data)
            self.timestamps.append(frame_data['timestamp'])

    def _get_frame_sensors(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sensors = {}

        for sensor_name, sample_data_token in sample['data'].items():
            sample_data = self.nusc.get('sample_data', sample_data_token)
            sensors[sensor_name] = sample_data

        return sensors

    def initialize(self):
        pose_converter = PoseConverter(nuscenes=self.nusc, frames=self.frames)
        lidar_converter = LidarConverter(nuscenes=self.nusc, frames=self.frames)
        annotation_converter = AnnotationConverter(nuscenes=self.nusc, frames=self.frames)
        map_converter = MapConverter(nuscenes=self.nusc,map_explorer=self.map_explorer, frames=self.frames)
        camera_converter = CameraConverter(
            nuscenes=self.nusc,
            frames=self.frames,
            image_max_width=self.image_max_width,
            image_max_height=self.image_max_height
        )

        self.converters = [
            pose_converter,
            lidar_converter,
            annotation_converter,
            map_converter,
            camera_converter
        ]

        self.metadata = self.get_metadata()

    def get_metadata(self) -> Any:
        builder = XVIZMetadataBuilder()

        if self.timestamps:
            builder.start_time(self.timestamps[0]).end_time(self.timestamps[-1])

        for converter in self.converters:
            converter.get_metadata(builder)

        return builder.get_message()

    def convert_frame(self, frame_index: int) -> Any:
        if frame_index >= len(self.frames):
            return None

        builder = XVIZBuilder(metadata=self.metadata)

        for converter in self.converters:
            converter.convert(frame_index, builder)

        return builder.get_message()

    def convert_scene(self, index: int, scene_name: Optional[str] = None, output_name: Optional[str] = None,
                      sample_limit: Optional[int] = None):
        self.load_nuscenes_data(scene_name)

        self.initialize()

        if output_name is None:
            output_name = scene_name or "nuscenes_scene"

        scene_output_dir = self.output_dir / output_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        sink = DirectorySource(str(scene_output_dir))
        writer = XVIZGLBWriter(sink)

        writer.write_message(self.metadata)

        frame_count = len(self.frames)
        if sample_limit is not None:
            frame_count = min(sample_limit, frame_count)

        for i in tqdm(range(frame_count), desc=f"No.{index + 1}, scene name: {scene_name}", unit=" keyframes"):
            message = self.convert_frame(i)
            if message:
                writer.write_message(message)

        writer.close()
