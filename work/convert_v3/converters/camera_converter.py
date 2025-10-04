import os
import io
from typing import List, Dict, Any, Tuple
from PIL import Image
from nuscenes.nuscenes import NuScenes
from xviz_avs import XVIZBuilder, XVIZMetadataBuilder


class CameraConverter:
    def __init__(self, nuscenes: NuScenes, frames: List[Dict[str, Any]], 
                 image_max_width: int, image_max_height: int):
        self.nuscenes = nuscenes
        self.frames = frames
        self.image_max_width = image_max_width
        self.image_max_height = image_max_height
        self.CAMERA_SOURCES = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

    def get_metadata(self, xmb: XVIZMetadataBuilder):
        for camera_name in self.CAMERA_SOURCES:
            stream_name = f'/camera/{camera_name.lower()}'
            xmb.stream(stream_name) \
                .category('primitive') \
                .type('image') \
                .coordinate('VEHICLE_RELATIVE')

    def convert(self, frame_index: int, xb: XVIZBuilder):
        frame = self.frames[frame_index]
        for camera_name in self.CAMERA_SOURCES:
            if camera_name not in frame['sensors']:
                continue

            camera_data = frame['sensors'][camera_name]
            image_file = os.path.join(self.nuscenes.dataroot, camera_data['filename'])

            if not os.path.exists(image_file):
                print(f"警告: 相机图像文件不存在：{image_file}")
                continue

            image_data, width, height = self._resize_image(
                image_file, self.image_max_width, self.image_max_height
            )

            if image_data is None:
                print(f"警告: 无法处理相机图像：{image_file}")
                continue

            stream_name = f'/camera/{camera_name.lower()}'

            xb.primitive(stream_name) \
                .image(image_data) \
                .dimensions(width, height)

    def _resize_image(self, image_path: str, max_width: int, max_height: int) -> Tuple[bytes, int, int]:
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            new_width, new_height = self._get_resize_dimensions(
                original_width, original_height, max_width, max_height
            )

            if new_width == original_width and new_height == original_height:
                with open(image_path, 'rb') as f:
                    return f.read(), original_width, original_height

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            if resized_img.mode in ('RGBA', 'P'):
                resized_img = resized_img.convert('RGB')

            resized_img.save(buffer, format='JPEG', quality=85)

            return buffer.getvalue(), new_width, new_height

    def _get_resize_dimensions(self, width: int, height: int, max_width: int, max_height: int) -> Tuple[int, int]:
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

