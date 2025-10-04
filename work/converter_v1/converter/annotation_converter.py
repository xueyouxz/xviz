import numpy as np
from nuscenes import NuScenes

from work.converter_v1.utils import quaternion_to_euler_angle
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder

CATEGORY_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",

}
CATEGORY = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck"
]


class AnnotationConverter:

    def __init__(self, nuscenes: NuScenes, frames):
        self.frames = frames
        self.nuscenes = nuscenes
        self.anns_by_frame = {}
        self.timestamps = []
        self.frames = frames
        self.ANNOTATIONS = '/annotations'
        self.init()

    def init(self):
        for i, frame in enumerate(self.frames):
            self.anns_by_frame[frame['token']] = {}
            for ann_token in frame['sample']['anns']:
                ann = self.nuscenes.get('sample_annotation', ann_token)
                if CATEGORY_MAPPING.get(ann["category_name"]) is None:
                    continue
                ins_token = ann['instance_token']
                self.anns_by_frame[frame['token']][ins_token] = self._parse_ann_data(ann)

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        for category in CATEGORY:
            xmb.stream(f'{self.ANNOTATIONS}/{category}') \
                .category('primitive') \
                .type('polygon') \
                .coordinate('IDENTITY')

            xmb.stream(f'{self.ANNOTATIONS}/{category}/trajectory') \
                .category('primitive') \
                .type('polygon') \
                .coordinate('IDENTITY')

    def convert(self, frame_index, xb: XVIZBuilder):
        frame_token = self.frames[frame_index]['token']
        frame_annotations = self.anns_by_frame[frame_token]
        if frame_annotations is not None:
            for anno_token in frame_annotations.keys():
                anno = frame_annotations[anno_token]
                xb.primitive(f'{self.ANNOTATIONS}/{anno["category"]}') \
                    .polygon(anno["vertices"]) \
                    .classes([anno["category"]]) \
                    .id(anno["token"])
                anno_trajectory = self._get_obj_trajectory(anno, frame_index, min(len(self.frames), frame_index + 7))
                xb.primitive(f'{self.ANNOTATIONS}/{anno["category"]}/trajectory').polyline(anno_trajectory)

    def _get_obj_trajectory(self, obj, start_index, end_index):
        trajectory = []
        for i in range(start_index, end_index):
            frame_token = self.frames[i]['token']
            objs = self.anns_by_frame[frame_token]
            if objs.get(obj['instance_token']) is  None:
                return np.asarray(trajectory).flatten().tolist()
            frame_obj = objs.get(obj['instance_token'])
            trajectory.append([frame_obj['x'], frame_obj['y'], frame_obj['z']])
        return np.asarray(trajectory).flatten().tolist()

    def _parse_ann_data(self, ann):
        translation = ann['translation']
        size = ann['size']
        roll, pitch, yaw = quaternion_to_euler_angle(ann['rotation'])

        category = CATEGORY_MAPPING[ann['category_name']]
        bounds = [
            [-size[1] / 2, -size[0] / 2, 0],
            [-size[1] / 2, size[0] / 2, 0],
            [size[1] / 2, size[0] / 2, 0],
            [size[1] / 2, -size[0] / 2, 0],
            [-size[1] / 2, -size[0] / 2, 0]
        ]
        return dict(
            token=ann['token'],
            instance_token=ann['instance_token'],
            category=category,
            bounds=bounds,
            x=translation[0],
            y=translation[1],
            z=translation[2],
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            vertices=np.asarray(bounds).flatten().tolist()  # todo: reference xviz-trajectory-helper.js transform coordinate
        )
