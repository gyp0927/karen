"""Image utilities for karen-coding-agent (pi coding-agent's `utils/image-*.ts`).

Re-exports mirroring pi's `index.ts` image surface, so callers can import the
whole pipeline from one place.
"""

from karen_agent.tools.image import detect_supported_image_mime_type

from .exif_orientation import apply_exif_orientation, get_exif_orientation
from .image_process import (
    IMAGE_TYPE_SNIFF_BYTES,
    ImageResizeOptions,
    ProcessImageResult,
    ResizedImage,
    convert_image_bytes_to_png,
    detect_supported_image_mime_type_from_file,
    format_dimension_note,
    process_image,
    resize_image,
    resize_image_in_process,
)
from .tool_result_images import ToolResultContent, normalize_tool_result_images

__all__ = [
    "IMAGE_TYPE_SNIFF_BYTES",
    "ImageResizeOptions",
    "ProcessImageResult",
    "ResizedImage",
    "ToolResultContent",
    "apply_exif_orientation",
    "convert_image_bytes_to_png",
    "detect_supported_image_mime_type",
    "detect_supported_image_mime_type_from_file",
    "format_dimension_note",
    "get_exif_orientation",
    "normalize_tool_result_images",
    "process_image",
    "resize_image",
    "resize_image_in_process",
]
