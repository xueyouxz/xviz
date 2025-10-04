import io
import os
from typing import Tuple

import numpy as np
from PIL import Image
from nuscenes import NuScenes

from xviz_avs import XVIZBuilder, XVIZMetadataBuilder

CAMERA_SOURCES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]


class CameraConverter:
    def __init__(self, nuscenes: NuScenes, frames, image_max_width, image_max_height):
        self.frames = frames
        self.nuscenes = nuscenes
        self.image_max_width = image_max_width
        self.image_max_height = image_max_height

    def convert(self, frame_index, xb: XVIZBuilder):
        frame = self.frames[frame_index]
        for camera_name in CAMERA_SOURCES:
            camera_data = frame['sensors'][camera_name]
            image_file = os.path.join(self.nuscenes.dataroot, camera_data['filename'])
            image_data, width, height = self._resize_image(
                image_file, self.image_max_width, self.image_max_height
            )

            xb.primitive(f'/camera/{camera_name.lower()}') \
                .image(image_data) \
                .dimensions(width, height)

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        for camera_name in CAMERA_SOURCES:
            xmb.stream(f'/camera/{camera_name.lower()}') \
                .category('primitive') \
                .type('image')

    def _resize_image(self, image_path: str, max_width: int, max_height: int) -> Tuple[bytes, int, int]:

        with Image.open(image_path) as img:
            original_width, original_height = img.size
            new_width, new_height = self._get_resize_dimensions(
                original_width, original_height, max_width, max_height
            )
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            if resized_img.mode in ('RGBA', 'P'):
                resized_img = resized_img.convert('RGB')
            resized_img.save(buffer, format='JPEG', quality=100)
            return buffer.getvalue(), new_width, new_height

    def _get_resize_dimensions(self, width: int, height: int, max_width: int, max_height: int) -> Tuple[int, int]:
        """计算保持宽高比的调整后尺寸"""
        if max_width <= 0 and max_height <= 0:
            return width, height

        ratio = width / height

        if max_height > 0 and max_width > 0:
            resize_width = min(max_width, max_height * ratio)
            resize_height = min(max_height, max_width / ratio)
        elif max_height > 0:
            resize_width = max_height * ratio
            resize_height = max_height
        elif max_width > 0:
            resize_width = max_width
            resize_height = max_width / ratio
        else:
            resize_width = width
            resize_height = height

        return int(resize_width), int(resize_height)
