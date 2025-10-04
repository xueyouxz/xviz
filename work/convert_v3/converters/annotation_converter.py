from typing import List, Dict, Any, Optional
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder
from work.convert_v3.utils import hex_to_rgba

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

# 不同类别对象的典型宽度（米）
CATEGORY_WIDTH = {
    "barrier": 0.5,
    "bicycle": 0.6,
    "bus": 2.7,
    "car": 1.8,
    "construction_vehicle": 2.5,
    "motorcycle": 0.8,
    "pedestrian": 0.7,
    "traffic_cone": 0.3,
    "trailer": 2.5,
    "truck": 2.5
}


class AnnotationConverter:
    def __init__(self, nuscenes: NuScenes, frames: List[Dict[str, Any]]):
        self.nuscenes = nuscenes
        self.frames = frames
        self.OBJECTS_TRACKING_POINT = '/objects/tracking_point'
        self.OBJECT_PALETTE = self._get_object_palette()

    def _get_object_palette(self) -> Dict[str, Dict[str, List[int]]]:
        return {
            '/annotations/barrier': {
                'fill_color': hex_to_rgba('#87CEEB80'),
                'stroke_color': hex_to_rgba('#87CEEBFF')
            },
            '/annotations/bicycle': {
                'fill_color': hex_to_rgba('#FF69B480'),
                'stroke_color': hex_to_rgba('#FF69B4FF')
            },
            '/annotations/bus': {
                'fill_color': hex_to_rgba('#FF8C0080'),
                'stroke_color': hex_to_rgba('#FF8C00FF')
            },
            '/annotations/car': {
                'fill_color': hex_to_rgba('#00CED180'),
                'stroke_color': hex_to_rgba('#00CED1FF')
            },
            '/annotations/construction_vehicle': {
                'fill_color': hex_to_rgba('#FFD70080'),
                'stroke_color': hex_to_rgba('#FFD700FF')
            },
            '/annotations/motorcycle': {
                'fill_color': hex_to_rgba('#EE82EE80'),
                'stroke_color': hex_to_rgba('#EE82EEFF')
            },
            '/annotations/pedestrian': {
                'fill_color': hex_to_rgba('#FFA50080'),
                'stroke_color': hex_to_rgba('#FFA500FF')
            },
            '/annotations/traffic_cone': {
                'fill_color': hex_to_rgba('#FF634780'),
                'stroke_color': hex_to_rgba('#FF6347FF')
            },
            '/annotations/trailer': {
                'fill_color': hex_to_rgba('#9370DB80'),
                'stroke_color': hex_to_rgba('#9370DBFF')
            },
            '/annotations/truck': {
                'fill_color': hex_to_rgba('#32CD3280'),
                'stroke_color': hex_to_rgba('#32CD32FF')
            }
        }

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        xmb.stream(self.OBJECTS_TRACKING_POINT) \
            .category('primitive') \
            .type('circle') \
            .stream_style({
            'radius': 0.2,
            'fill_color': [255, 255, 0, 255]
        }) \
            .coordinate('IDENTITY')

        for category in CATEGORY:
            stream_name = f'/annotations/{category}'
            style = self.OBJECT_PALETTE[stream_name]
            
            # 定义标注对象流
            xmb.stream(stream_name) \
                .category('primitive') \
                .type('polygon') \
                .coordinate('IDENTITY') \
                .stream_style({
                'extruded': True,
                'fill_color': [0, 0, 0, 128]
            }) \
                .style_class(stream_name, style)
            
            # 为每个类别定义独立的轨迹流，使用与标注对象相同的颜色
            trajectory_stream = f'{stream_name}/trajectory'
            trajectory_width = CATEGORY_WIDTH.get(category, 1.0)
            xmb.stream(trajectory_stream) \
                .category('primitive') \
                .type('polyline') \
                .coordinate('IDENTITY') \
                .stream_style({
                'stroke_color': style['stroke_color'],
                'stroke_width': trajectory_width,
                'stroke_width_min_pixels': 1
            })

    def convert(self, frame_index: int, xb: XVIZBuilder):
        frame = self.frames[frame_index]
        sample = frame['sample']

        annotations = []
        for ann_token in sample['anns']:
            ann = self.nuscenes.get('sample_annotation', ann_token)
            annotations.append(ann)

        for ann in annotations:
            category = self._map_nuscenes_category(ann['category_name'])
            if not category:
                continue

            bbox_3d = self._get_3d_bbox(ann)
            if bbox_3d is None:
                continue

            center = ann['translation']

            flattened_vertices = []
            for vertex in bbox_3d:
                flattened_vertices.extend(vertex)

            xb.primitive(category) \
                .polygon(flattened_vertices) \
                .classes([category]) \
                .style({'height': ann['size'][2]}) \
                .id(ann['token'])

            xb.primitive(self.OBJECTS_TRACKING_POINT) \
                .circle([center[0], center[1], 0], 0.2) \
                .id(ann['token'])

        self._add_object_trajectories(annotations, xb, frame_index)

    def _map_nuscenes_category(self, nuscenes_category: str) -> Optional[str]:
        category = CATEGORY_MAPPING.get(nuscenes_category)
        if category:
            return f'/annotations/{category}'
        return None

    def _get_3d_bbox(self, annotation: Dict[str, Any]) -> Optional[List[List[float]]]:
        box = Box(
            center=annotation['translation'],
            size=annotation['size'],
            orientation=Quaternion(annotation['rotation'])
        )
        corners_3d = box.corners()

        bottom_indices = [2, 3, 7, 6]
        bottom_corners = corners_3d[:, bottom_indices]

        vertices = []
        for i in range(4):
            vertices.append([
                float(bottom_corners[0, i]),
                float(bottom_corners[1, i]),
                float(bottom_corners[2, i])
            ])

        vertices.append(vertices[0])

        return vertices

    def _add_object_trajectories(self, annotations: List[Dict[str, Any]], xb: XVIZBuilder, frame_index: int):
        lookahead = 6
        end_index = min(len(self.frames), frame_index + lookahead)

        for ann in annotations:
            instance_token = ann['instance_token']
            category = self._map_nuscenes_category(ann['category_name'])
            
            if not category:
                continue
            
            trajectory = []

            for i in range(frame_index, end_index):
                future_frame = self.frames[i]
                future_sample = future_frame['sample']

                for future_ann_token in future_sample['anns']:
                    future_ann = self.nuscenes.get('sample_annotation', future_ann_token)
                    if future_ann['instance_token'] == instance_token:
                        pos = future_ann['translation']
                        trajectory.append([pos[0], pos[1], 0])
                        break

            if len(trajectory) > 1:
                flattened_trajectory = []
                for point in trajectory:
                    flattened_trajectory.extend(point)
                
                # 使用对应类别的轨迹流
                trajectory_stream = f'{category}/trajectory'
                xb.primitive(trajectory_stream).polyline(flattened_trajectory)

