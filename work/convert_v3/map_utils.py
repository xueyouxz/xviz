from typing import List, Union, Dict, Optional, Tuple

import numpy as np
from numpy._typing import NDArray
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from pyquaternion import Quaternion
from shapely import Polygon, strtree, LineString, ops, LinearRing
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def _union_ped(ped_geoms: List[Polygon]) -> List[Polygon]:
    ''' merge close ped crossings.

    Args:
        ped_geoms (list): list of Polygon

    Returns:
        union_ped_geoms (Dict): merged ped crossings
    '''
    if not ped_geoms:
        return []

    def get_rec_direction(geom):
        rect = geom.minimum_rotated_rectangle
        rect_v_p = np.array(rect.exterior.coords)[:3]
        rect_v = rect_v_p[1:] - rect_v_p[:-1]
        v_len = np.linalg.norm(rect_v, axis=-1)
        longest_v_i = v_len.argmax()

        return rect_v[longest_v_i], v_len[longest_v_i]

    tree = strtree.STRtree(ped_geoms)

    final_pgeom = []
    remain_idx = set(range(len(ped_geoms)))
    
    for i, pgeom in enumerate(ped_geoms):
        if i not in remain_idx:
            continue
        
        remain_idx.discard(i)
        pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
        final_pgeom.append(pgeom)

        query_results = tree.query(pgeom)
        
        if isinstance(query_results, np.ndarray):
            candidate_indices = query_results
        else:
            candidate_indices = []
            for o in query_results:
                for j, geom in enumerate(ped_geoms):
                    if geom is o:
                        candidate_indices.append(j)
                        break
        
        for o_idx in candidate_indices:
            if o_idx not in remain_idx or o_idx == i:
                continue

            o = ped_geoms[o_idx]
            o_v, o_v_norm = get_rec_direction(o)
            cos = pgeom_v.dot(o_v) / (pgeom_v_norm * o_v_norm)
            if 1 - np.abs(cos) < 0.01:
                final_pgeom[-1] = final_pgeom[-1].union(o)
                remain_idx.discard(o_idx)

    results = []
    for p in final_pgeom:
        results.extend(split_collections(p))
    return results


def get_map_geom(map_explorer: NuScenesMapExplorer, location: str,
                 translation: Union[List, NDArray],
                 rotation: Union[List, NDArray], roi_size=(30, 60)) -> Dict[str, List[Union[LineString, Polygon]]]:
    ''' Extract geometries given `location` and self pose, self may be lidar or ego.

    Args:
        location (str): city name
        translation (array): self2global translation, shape (3,)
        rotation (array): self2global quaternion, shape (4, )

    Returns:
        geometries (Dict): extracted geometries by category.
    '''

    # (center_x, center_y, len_y, len_x) in nuscenes format
    patch_box = (translation[0], translation[1], roi_size[1], roi_size[0])
    local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2,
                      roi_size[0] / 2, roi_size[1] / 2)
    rotation = Quaternion(rotation)
    yaw = quaternion_yaw(rotation) / np.pi * 180

    # get dividers
    lane_dividers = map_explorer._get_layer_line(
        patch_box, yaw, 'lane_divider')

    road_dividers = map_explorer._get_layer_line(
        patch_box, yaw, 'road_divider')

    all_dividers = []
    for line in lane_dividers + road_dividers:
        all_dividers += split_collections(line)

    # get ped crossings
    ped_crossings = []
    ped = map_explorer._get_layer_polygon(
        patch_box, yaw, 'ped_crossing')

    for p in ped:
        ped_crossings += split_collections(p)
    # some ped crossings are split into several small parts
    # we need to merge them
    ped_crossings = _union_ped(ped_crossings)

    ped_crossing_lines = []
    for p in ped_crossings:
        # extract exteriors to get a closed polyline
        line = get_ped_crossing_contour(p, local_patch)
        if line is not None:
            ped_crossing_lines.append(line)

    # get boundaries
    # we take the union of road segments and lanes as drivable areas
    # we don't take drivable area layer in nuScenes since its definition may be ambiguous
    road_segments = map_explorer._get_layer_polygon(
        patch_box, yaw, 'road_segment')
    lanes = map_explorer._get_layer_polygon(
        patch_box, yaw, 'lane')
    union_roads = ops.unary_union(road_segments)
    union_lanes = ops.unary_union(lanes)
    drivable_areas = ops.unary_union([union_roads, union_lanes])

    drivable_areas = split_collections(drivable_areas)

    # boundaries are defined as the contour of drivable areas
    boundaries = get_drivable_area_contour(drivable_areas, roi_size)

    return dict(
        divider=all_dividers,  # List[LineString]
        ped_crossing=ped_crossing_lines,  # List[LineString]
        boundary=boundaries,  # List[LineString]
        drivable_area=drivable_areas,  # List[Polygon],
    )


def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    ''' Split Multi-geoms to list and check is valid or is empty.

    Args:
        geom (BaseGeometry): geoms to be split or validate.

    Returns:
        geometries (List): list of geometries.
    '''
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon',
                              'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom, ]
        else:
            return []


def get_drivable_area_contour(drivable_areas: List[Polygon],
                              roi_size: Tuple) -> List[LineString]:
    ''' Extract drivable area contours to get list of boundaries.

    Args:
        drivable_areas (list): list of drivable areas.
        roi_size (tuple): bev range size

    Returns:
        boundaries (List): list of boundaries.
    '''
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2

    # a bit smaller than roi to avoid unexpected boundaries on edges
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)

    exteriors = []
    interiors = []

    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    results = []
    for ext in exteriors:
        # NOTE: we make sure all exteriors are clock-wise
        # such that each boundary's right-hand-side is drivable area
        # and left-hand-side is walk way

        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])
        lines = ext.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']

        results.extend(split_collections(lines))

    for inter in interiors:
        # NOTE: we make sure all interiors are counter-clock-wise
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])
        lines = inter.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']

        results.extend(split_collections(lines))

    return results


def get_ped_crossing_contour(polygon: Polygon,
                             local_patch: box) -> Optional[LineString]:
    ''' Extract ped crossing contours to get a closed polyline.
    Different from `get_drivable_area_contour`, this function ensures a closed polyline.

    Args:
        polygon (Polygon): ped crossing polygon to be extracted.
        local_patch (tuple): local patch params

    Returns:
        line (LineString): a closed line
    '''

    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.geom_type != 'LineString':
        # remove points in intersection results
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)

        # same instance but not connected.
        if lines.geom_type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))

            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)
    if not lines.is_empty:
        return lines

    return None
