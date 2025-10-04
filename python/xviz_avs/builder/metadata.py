import logging
from easydict import EasyDict as edict
import numpy as np

from xviz_avs.message import XVIZMessage
from xviz_avs.builder.base_builder import build_object_style, build_stream_style
from xviz_avs.v2.session_pb2 import Metadata, StreamMetadata, LogInfo


class XVIZMetadataBuilder:
    def __init__(self, logger=logging.getLogger("xviz")):
        self._logger = logger

        self._data = Metadata(version="2.0.0")
        self._temp_ui_builder = None
        self._reset()

    def get_data(self):
        self._flush()

        metadata = self._data

        if self._temp_ui_builder:
            panels = self._temp_ui_builder.get_ui()

            for panel_key in panels.keys():
                metadata.ui_config[panel_key].name = panel_key
                metadata.ui_config[panel_key].config.update(panels[panel_key])

        return metadata

    def get_message(self):
        return XVIZMessage(metadata=self.get_data())

    def start_time(self, time):
        self._data.log_info.start_time = time
        return self

    def end_time(self, time):
        self._data.log_info.end_time = time
        return self

    def ui(self, ui_builder):
        self._temp_ui_builder = ui_builder
        return self

    def stream(self, stream_id):
        if self._stream_id:
            self._flush()

        self._stream_id = stream_id
        return self

    def category(self, category):
        '''
        Assign category for the stream. Used for validation in XVIZBuilder and not required for data.
        '''
        if isinstance(category, int):
            self._temp_stream.category = category
        elif isinstance(category, str):
            self._temp_stream.category = StreamMetadata.Category.Value(category.upper())
        else:
            self._logger.error("Invalid value type for category!")
        return self

    def type(self, t):
        '''
        Assign primitive type for the stream. Used for validation in XVIZBuilder and not required for data.
        '''
        if isinstance(t, int):
            self._temp_type = t
        elif isinstance(t, str):
            self._temp_type = t.upper()
        else:
            self._logger.error("Invalid value type for category!")
        return self

    def source(self, source):
        self._temp_stream.source = source
        return self

    def unit(self, u):
        self._temp_stream.units = u
        return self

    def coordinate(self, coordinate):
        self._temp_stream.coordinate = coordinate
        return self

    def transform_matrix(self, matrix):
        matrix = np.array(matrix).ravel()
        self._temp_stream.transform.extend(matrix.tolist())
        return self

    def pose(self, position={}, orientation={}):
        """
        Set pose transformation for the stream using position and orientation.
        Similar to JavaScript version implementation.

        Args:
            position: dict with keys 'x', 'y', 'z' (default: 0)
            orientation: dict with keys 'roll', 'pitch', 'yaw' (default: 0, in radians)
        """
        # Extract position values with defaults
        x = position[0]
        y = position[1]
        z = position[2]

        # Extract orientation values with defaults (in radians)
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]

        # Create rotation matrices for roll, pitch, yaw
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # Roll rotation matrix (around X-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r]
        ])

        # Pitch rotation matrix (around Y-axis)
        R_y = np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ])

        # Yaw rotation matrix (around Z-axis)
        R_z = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix: R = R_z * R_y * R_x
        rotation_matrix = R_z @ R_y @ R_x

        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = [x, y, z]

        # Store as flattened array similar to transform_matrix method
        matrix = transform_matrix.ravel()
        del self._temp_stream.transform[:]
        self._temp_stream.transform.extend(matrix.tolist())

        return self

    def stream_style(self, style):
        self._temp_stream.stream_style.MergeFrom(build_stream_style(style))
        return self

    def style_class(self, name, style):
        if not self._stream_id:
            self._logger.error('A stream must set before adding a style rule.')
            return self
        style_class = self._temp_stream.style_classes.add()
        style_class.name = name
        style_class.style.MergeFrom(build_object_style(style))
        return self

    def log_info(self, data):
        self._data.log_info.MergeFrom(LogInfo(**data))
        return self

    def _flush(self):
        if self._stream_id:
            stream_data = self._temp_stream

            if stream_data.category in [1, 5]:
                stream_data.primitive_type = self._temp_type
            elif stream_data.category in [2, 3]:
                stream_data.scalar_type = self._temp_type

            self._data.streams[self._stream_id].MergeFrom(stream_data)

        self._reset()

    def _reset(self):
        self._stream_id = None
        self._temp_stream = StreamMetadata()
        self._temp_type = None
