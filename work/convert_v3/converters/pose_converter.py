from typing import List, Dict, Any
from nuscenes.nuscenes import NuScenes
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder
from work.convert_v3.utils import quaternion_to_euler


class PoseConverter:
    def __init__(self, nuscenes: NuScenes, frames: List[Dict[str, Any]]):
        self.nuscenes = nuscenes
        self.frames = frames
        self.VEHICLE_POSE = '/vehicle_pose'
        self.VEHICLE_TRAJECTORY = '/vehicle/trajectory'

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        xmb.stream(self.VEHICLE_POSE).category('pose')

        xmb.stream(self.VEHICLE_TRAJECTORY) \
            .category('primitive') \
            .type('polyline') \
            .coordinate('IDENTITY') \
            .stream_style({
            'stroke_color': [87, 173, 87, 170],
            'stroke_width': 1.4,
            'stroke_width_min_pixels': 1
        })

    def convert(self, frame_index: int, xb: XVIZBuilder):
        frame = self.frames[frame_index]
        ego_pose_token = frame['ego_pose_token']
        ego_pose = self.nuscenes.get('ego_pose', ego_pose_token)

        position = ego_pose['translation']
        x, y, z = position

        quaternion = ego_pose['rotation']
        roll, pitch, yaw = quaternion_to_euler(quaternion)

        xb.pose(self.VEHICLE_POSE) \
            .timestamp(frame['timestamp']) \
            .map_origin(0, 0, 0) \
            .position(x, y, z) \
            .orientation(roll, pitch, yaw)

        trajectory = self._get_vehicle_trajectory(frame_index)
        if trajectory:
            flattened_trajectory = []
            for point in trajectory:
                flattened_trajectory.extend(point)
            xb.primitive(self.VEHICLE_TRAJECTORY).polyline(flattened_trajectory)

    def _get_vehicle_trajectory(self, frame_index: int, lookahead: int = 6) -> List[List[float]]:
        trajectory = []
        end_index = min(len(self.frames), frame_index + lookahead)

        for i in range(frame_index, end_index):
            frame = self.frames[i]
            ego_pose = self.nuscenes.get('ego_pose', frame['ego_pose_token'])
            pos = ego_pose['translation']
            trajectory.append([pos[0], pos[1], pos[2]])

        return trajectory

