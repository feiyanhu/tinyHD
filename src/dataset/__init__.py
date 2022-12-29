from typing import Any, Dict, Iterator

import torch

from .utils._video_opt import (
    Timebase,
    VideoMetaData,
    _HAS_VIDEO_OPT,
    _probe_video_from_file,
    _probe_video_from_memory,
    _read_video_from_file,
    _read_video_from_memory,
    _read_video_timestamps_from_file,
    _read_video_timestamps_from_memory,
)

from .utils.video import (
    read_video,
    read_video_timestamps,
    write_video,
)

if _HAS_VIDEO_OPT:

    def _has_video_opt() -> bool:
        return True


else:

    def _has_video_opt() -> bool:
        return False

__all__ = [
    "write_video",
    "read_video",
    "read_video_timestamps",
    "_read_video_from_file",
    "_read_video_timestamps_from_file",
    "_probe_video_from_file",
    "_read_video_from_memory",
    "_read_video_timestamps_from_memory",
    "_probe_video_from_memory",
    "_HAS_VIDEO_OPT",
    "_read_video_clip_from_memory",
    "_read_video_meta_data",
    "VideoMetaData",
    "Timebase",
    "ImageReadMode",
    "decode_image",
    "decode_jpeg",
    "decode_png",
    "encode_jpeg",
    "encode_png",
    "read_file",
    "read_image",
    "write_file",
    "write_jpeg",
    "write_png",
    "Video",
]