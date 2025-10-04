import numpy as np
from nuscenes import NuScenes

from work.converter_v1.utils import quaternion_to_euler_angle
from xviz_avs import XVIZMetadataBuilder, XVIZBuilder


class CoordinateConverter:
    def __init__(self, nuscenes: NuScenes, frames):
        self.frames = frames
        self.pose_by_frames = {}
        self.timestamps = []
        self.nuscenes = nuscenes
        self.EGO_POSE = '/ego_pose'
        self.EGO_TRAJECTORY = '/ego/trajectory'
        self.init()

    def init(self):
        for i, frame in enumerate(self.frames):
            ego_pose_token = frame['ego_pose_token']
            ego_pose = self.nuscenes.get('ego_pose', ego_pose_token)
            roll, pitch, yaw = quaternion_to_euler_angle(ego_pose['rotation'])
            timestamp = ego_pose['timestamp'] / 1e6
            self.timestamps.append(timestamp)
            self.pose_by_frames[frame['token']] = dict(
                timestamp=timestamp / 1e6,
                x=ego_pose['translation'][0],
                y=ego_pose['translation'][1],
                z=ego_pose['translation'][2],
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                raw_data=ego_pose
            )

    def convert(self, frame_index, xb: XVIZBuilder):
        frame_token = self.frames[frame_index]['token']
        pose = self.pose_by_frames[frame_token]

        xb.pose(self.EGO_POSE) \
            .timestamp(pose['timestamp']) \
            .map_origin(0, 0, 0) \
            .position(pose['x'], pose['y'], pose['z']) \
            .orientation(pose['roll'], pose['pitch'], pose['yaw'])

        ego_trajectory = self._get_ego_trajectory(frame_index, min(len(self.frames), frame_index + 7))

        xb.primitive(self.EGO_TRAJECTORY) \
            .polyline(ego_trajectory)

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        xmb.stream(self.EGO_POSE) \
            .category('pose') \
            .stream(self.EGO_TRAJECTORY) \
            .category('primitive') \
            .type('polyline') \
            .coordinate('IDENTITY')

    def get_pose(self):
        return self.pose_by_frames

    def _get_ego_trajectory(self, start_index, end_index):
        trajectory = []
        for i in range(start_index, end_index):
            frame_token = self.frames[i]['token']
            pose = self.pose_by_frames[frame_token]
            trajectory.append([pose['x'], pose['y'], pose['z']])
        return np.asarray(trajectory).flatten().tolist()
