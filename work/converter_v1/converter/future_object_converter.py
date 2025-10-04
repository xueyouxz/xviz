from nuscenes import NuScenes

from work.converter_v1.converter.annotation_converter import AnnotationConverter
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder

FUTURE_STEPS = 6  # 3 seconds


class FutureAnnoConverter(AnnotationConverter):
    def __init__(self, nuscenes: NuScenes, frames):
        super().__init__(nuscenes, frames)
        self.OBJECTS_FUTURES = '/objects/futures'

    def convert(self, frame_index, xb: XVIZBuilder):
        future_frame_limit = min(frame_index + FUTURE_STEPS, len(self.frames))
        for future_frame_index in range(frame_index, future_frame_limit):
            frame_token = self.frames[future_frame_index]['token']
            frame_annotations = self.anns_by_frame[frame_token]
            ego_pose_token = self.frames[future_frame_index]['ego_pose_token']
            ego_pose = self.nuscenes.get('ego_pose', ego_pose_token)
            timestamp = ego_pose['timestamp'] / 1e6
            for anno_token in frame_annotations.keys():
                anno = frame_annotations[anno_token]
                xb.future_instance(self.OBJECTS_FUTURES, timestamp) \
                    .polygon(anno['vertices']) \
                    .classes([anno['category']]) \
                    .id(anno['token'])

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        xmb.stream(self.OBJECTS_FUTURES) \
            .category('future_instance') \
            .type('polygon') \
            .coordinate('IDENTITY')
