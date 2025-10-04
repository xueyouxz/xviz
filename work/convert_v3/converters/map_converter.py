import numpy as np
from typing import List, Dict, Any, Optional
from nuscenes.nuscenes import NuScenes
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder
from shapely.geometry import LineString, Polygon
from work.convert_v3.map_utils import get_map_geom


class MapConverter:
    def __init__(self, nuscenes: NuScenes, map_explorer: Optional[Any] = None, frames: List[Dict[str, Any]] = None):
        self.nuscenes = nuscenes
        self.map_explorer = map_explorer
        self.frames = frames
        self.location = self._get_scene_location()
        
        self.MAP_DIVIDER = '/map/divider'
        self.MAP_PED_CROSSING = '/map/ped_crossing'
        self.MAP_BOUNDARY = '/map/boundary'
        self.MAP_DRIVABLE_AREA = '/map/drivable_area'

    def _get_scene_location(self) -> Optional[str]:
        if not self.frames:
            return None
        first_frame = self.frames[0]
        sample = first_frame['sample']
        scene_token = sample['scene_token']
        scene = self.nuscenes.get('scene', scene_token)
        log = self.nuscenes.get('log', scene['log_token'])
        return log['location']

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        xmb.stream(self.MAP_DIVIDER) \
            .category('primitive') \
            .type('polyline') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
                'stroke_color': [255, 255, 255, 255],
                'stroke_width': 0.2,
                'stroke_width_min_pixels': 1
            })
        
        xmb.stream(self.MAP_PED_CROSSING) \
            .category('primitive') \
            .type('polyline') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
                'stroke_color': [255, 217, 82, 255],
                'stroke_width': 0.3,
                'stroke_width_min_pixels': 1
            })
        
        xmb.stream(self.MAP_BOUNDARY) \
            .category('primitive') \
            .type('polyline') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
                'stroke_color': [255, 179, 0, 255],
                'stroke_width': 0.3,
                'stroke_width_min_pixels': 1
            })
        
        xmb.stream(self.MAP_DRIVABLE_AREA) \
            .category('primitive') \
            .type('polygon') \
            .coordinate('VEHICLE_RELATIVE') \
            .stream_style({
                'fill_color': [100, 100, 100, 64],
                'stroke_color': [100, 100, 100, 128],
                'stroke_width': 0.1,
                'stroke_width_min_pixels': 1
            })

    def convert(self, frame_index: int, xb: XVIZBuilder):
        if not self.map_explorer or not self.location:
            return
        
        frame = self.frames[frame_index]
        ego_pose_token = frame['ego_pose_token']
        ego_pose = self.nuscenes.get('ego_pose', ego_pose_token)
        
        translation = ego_pose['translation']
        rotation = ego_pose['rotation']
        
        roi_size = (60, 120)
        
        map_geoms = get_map_geom(
            map_explorer=self.map_explorer,
            location=self.location,
            translation=translation,
            rotation=rotation,
            roi_size=roi_size
        )
        
        self._convert_dividers(map_geoms.get('divider', []), xb)
        self._convert_ped_crossings(map_geoms.get('ped_crossing', []), xb)
        self._convert_boundaries(map_geoms.get('boundary', []), xb)
        self._convert_drivable_areas(map_geoms.get('drivable_area', []), xb)

    def _convert_dividers(self, dividers: List[LineString], xb: XVIZBuilder):
        for idx, divider in enumerate(dividers):
            vertices = self._linestring_to_vertices(divider)
            if vertices:
                xb.primitive(self.MAP_DIVIDER).polyline(vertices)

    def _convert_ped_crossings(self, ped_crossings: List[LineString], xb: XVIZBuilder):
        for idx, ped_crossing in enumerate(ped_crossings):
            vertices = self._linestring_to_vertices(ped_crossing)
            if vertices:
                xb.primitive(self.MAP_PED_CROSSING).polyline(vertices)

    def _convert_boundaries(self, boundaries: List[LineString], xb: XVIZBuilder):
        for idx, boundary in enumerate(boundaries):
            vertices = self._linestring_to_vertices(boundary)
            if vertices:
                xb.primitive(self.MAP_BOUNDARY).polyline(vertices)

    def _convert_drivable_areas(self, drivable_areas: List[Polygon], xb: XVIZBuilder):
        for idx, drivable_area in enumerate(drivable_areas):
            vertices = self._polygon_to_vertices(drivable_area)
            if vertices:
                xb.primitive(self.MAP_DRIVABLE_AREA).polygon(vertices)

    def _linestring_to_vertices(self, linestring: LineString) -> List[float]:
        coords = np.array(linestring.coords)
        vertices = []
        for coord in coords:
            vertices.extend([float(coord[0]), float(coord[1]), 0.0])
        return vertices

    def _polygon_to_vertices(self, polygon: Polygon) -> List[float]:
        coords = np.array(polygon.exterior.coords)
        vertices = []
        for coord in coords:
            vertices.extend([float(coord[0]), float(coord[1]), 0.0])
        return vertices
