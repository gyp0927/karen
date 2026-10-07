"""Inline-image normalization and resizing (pi's `utils/image-process.ts`,
`utils/image-resize-core.ts`, `utils/image-resize.ts`, `utils/image-convert.ts`).

pi decodes/resizes with the photon Rust/WASM module and dispatches the resize
to a node worker thread. karen uses Pillow for the pixel work and runs the
blocking encode/resize via `asyncio.to_thread` (Pillow's C core releases the
GIL around decode/resize/encode, so a thread keeps the event loop responsive —
pi's worker-thread motivation), falling back inline if the thread can't start.
The algorithm, defaults, hint strings and edge cases are pi's verbatim.

Supported inline formats (everything else is converted to PNG, or omitted when
it cannot be decoded): PNG, JPEG, GIF, WebP.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
from typing import List, Optional, Union

from karen_agent.tools.image import detect_supported_image_mime_type

from .exif_orientation import apply_exif_orientation

__all__ = [
    "ImageResizeOptions",
    "ResizedImage",
    "ProcessImageResult",
    "resize_image_in_process",
    "resize_image",
    "process_image",
    "format_dimension_note",
    "detect_supported_image_mime_type_from_file",
    "convert_image_bytes_to_png",
]

# 4.5MB of base64 payload. Provides headroom below Anthropic's 5MB limit.
_DEFAULT_MAX_BYTES = int(4.5 * 1024 * 1024)

# Read this many bytes to sniff a file's image type (pi's `IMAGE_TYPE_SNIFF_BYTES`).
IMAGE_TYPE_SNIFF_BYTES = 4100


async def detect_supported_image_mime_type_from_file(path: str) -> Optional[str]:
    """Sniff a file's image mime type from its first bytes (pi's `mime.ts`)."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(IMAGE_TYPE_SNIFF_BYTES)
    except OSError:
        return None
    return detect_supported_image_mime_type(head)


class ImageResizeOptions:
    """Resize limits. Defaults match pi: 2000x2000, 4.5MB base64, JPEG quality 80."""

    def __init__(
        self,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_bytes: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> None:
        self.max_width = max_width if max_width is not None else 2000
        self.max_height = max_height if max_height is not None else 2000
        self.max_bytes = max_bytes if max_bytes is not None else _DEFAULT_MAX_BYTES
        self.jpeg_quality = jpeg_quality if jpeg_quality is not None else 80


class ResizedImage:
    def __init__(
        self,
        data: str,
        mime_type: str,
        original_width: int,
        original_height: int,
        width: int,
        height: int,
        was_resized: bool,
    ) -> None:
        self.data = data  # base64
        self.mime_type = mime_type
        self.original_width = original_width
        self.original_height = original_height
        self.width = width
        self.height = height
        self.was_resized = was_resized


class ProcessImageResult:
    """pi's `ProcessImageResult` union, as a small class with an `ok` flag."""

    def __init__(
        self,
        ok: bool,
        data: Optional[str] = None,
        mime_type: Optional[str] = None,
        hints: Optional[List[str]] = None,
        message: Optional[str] = None,
    ) -> None:
        self.ok = ok
        self.data = data
        self.mime_type = mime_type
        self.hints = hints or []
        self.message = message


def _base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower()


def _normalize_supported_image_mime_type(mime_type: str) -> Optional[str]:
    base = _base_mime_type(mime_type)
    if base == "image/png":
        return "image/png"
    if base in ("image/jpeg", "image/jpg"):
        return "image/jpeg"
    if base == "image/gif":
        return "image/gif"
    if base == "image/webp":
        return "image/webp"
    return None


def _load_image(input_bytes: bytes):
    """Decode + EXIF-transpose with Pillow; returns an Image or None.

    Unreadable bytes return None. An EXIF transpose failure is *not* swallowed
    here: pi's `resizeImageInProcess` wraps decode + `applyExifOrientation` in
    one try/catch and returns null, so a failure aborts the resize instead of
    sending a silently-unrotated image to the model. Callers wrap this call.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        image = Image.open(io.BytesIO(input_bytes))
        image.load()
    except Exception:
        return None
    return apply_exif_orientation(image, input_bytes)


def _encode_png(image) -> bytes:
    buffer = io.BytesIO()
    # Normalize palette/alpha modes the way photon's get_bytes() flattens to PNG.
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _encode_jpeg(image, quality: int) -> bytes:
    buffer = io.BytesIO()
    # JPEG has no alpha: flatten onto a white background like photon's JPEG encoder.
    if image.mode in ("RGBA", "LA", "P"):
        from PIL import Image

        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _encode_candidate(buffer: bytes, mime_type: str) -> dict:
    data = base64.b64encode(buffer).decode("utf-8")
    return {"data": data, "encoded_size": len(data.encode("utf-8")), "mime_type": mime_type}


def convert_image_bytes_to_png(input_bytes: bytes) -> Optional[bytes]:
    """Decode any image Pillow supports and re-encode as PNG (EXIF-applied)."""
    try:
        image = _load_image(input_bytes)
        if image is None:
            return None
        return _encode_png(image)
    except Exception:
        # pi's `convertImageBytesToPng` wraps decode + EXIF + encode in one
        # try/catch: any failure means "could not be converted".
        return None


def resize_image_in_process(
    input_bytes: bytes,
    mime_type: str,
    options: Optional[ImageResizeOptions] = None,
) -> Optional[ResizedImage]:
    """Resize to fit max dimensions and encoded size; None when it can't get below maxBytes.

    Strategy (pi's verbatim): resize to maxWidth/maxHeight, try PNG and JPEG
    picking the first under maxBytes, then progressively shrink dimensions by
    0.75 until 1x1. Returns the input unchanged when already within all limits.
    """
    opts = options if options is not None else ImageResizeOptions()
    # Model-level resize profiles (karen-ai's ModelImageResizeOptions) share the
    # same field names; resolve each against pi's defaults (None -> default).
    max_width = opts.max_width if opts.max_width is not None else 2000
    max_height = opts.max_height if opts.max_height is not None else 2000
    max_bytes = opts.max_bytes if opts.max_bytes is not None else _DEFAULT_MAX_BYTES
    jpeg_quality = opts.jpeg_quality if opts.jpeg_quality is not None else 80
    input_base64_size = ((len(input_bytes) + 2) // 3) * 4

    try:
        # Decode + EXIF inside the try, like pi (a transpose failure aborts the
        # resize rather than sending an unrotated image).
        image = _load_image(input_bytes)
        if image is None:
            return None

        original_width, original_height = image.size
        fmt = (mime_type.split("/")[1] if "/" in mime_type else "png") or "png"

        # Already within all limits (dimensions AND encoded size)?
        if original_width <= max_width and original_height <= max_height and input_base64_size < max_bytes:
            return ResizedImage(
                data=base64.b64encode(input_bytes).decode("utf-8"),
                mime_type=mime_type or f"image/{fmt}",
                original_width=original_width,
                original_height=original_height,
                width=original_width,
                height=original_height,
                was_resized=False,
            )

        # Initial dimensions respecting max limits (JS Math.round = round half up).
        target_width = original_width
        target_height = original_height
        if target_width > max_width:
            target_height = math.floor(target_height * max_width / target_width + 0.5)
            target_width = max_width
        if target_height > max_height:
            target_width = math.floor(target_width * max_height / target_height + 0.5)
            target_height = max_height

        from PIL import Image

        def try_encodings(width: int, height: int, jpeg_qualities: List[int]) -> List[dict]:
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            candidates = [_encode_candidate(_encode_png(resized), "image/png")]
            for quality in jpeg_qualities:
                candidates.append(_encode_candidate(_encode_jpeg(resized, quality), "image/jpeg"))
            return candidates

        quality_steps = list(dict.fromkeys([jpeg_quality, 85, 70, 55, 40]))
        current_width = target_width
        current_height = target_height

        while True:
            candidates = try_encodings(current_width, current_height, quality_steps)
            for candidate in candidates:
                if candidate["encoded_size"] < max_bytes:
                    return ResizedImage(
                        data=candidate["data"],
                        mime_type=candidate["mime_type"],
                        original_width=original_width,
                        original_height=original_height,
                        width=current_width,
                        height=current_height,
                        was_resized=True,
                    )

            if current_width == 1 and current_height == 1:
                break

            next_width = 1 if current_width == 1 else max(1, int(current_width * 0.75))
            next_height = 1 if current_height == 1 else max(1, int(current_height * 0.75))
            if next_width == current_width and next_height == current_height:
                break

            current_width = next_width
            current_height = next_height

        return None
    except Exception:
        return None


async def resize_image(
    input_bytes: bytes,
    mime_type: str,
    options: Optional[ImageResizeOptions] = None,
) -> Optional[ResizedImage]:
    """Run the blocking resize on a thread (pi's worker-thread motivation).

    Pillow's C core releases the GIL around decode/resize/encode, so a thread
    keeps the event loop responsive; fall back inline if the thread can't start
    (pi falls back to in-process when its worker fails to load).
    """
    try:
        return await asyncio.to_thread(resize_image_in_process, input_bytes, mime_type, options)
    except RuntimeError:
        return resize_image_in_process(input_bytes, mime_type, options)


def format_dimension_note(resized: ResizedImage) -> Optional[str]:
    """pi's `formatDimensionNote`: tell the model how to map coordinates back."""
    if not resized.was_resized:
        return None
    scale = resized.original_width / resized.width
    return (
        f"[Image: original {resized.original_width}x{resized.original_height}, "
        f"displayed at {resized.width}x{resized.height}. "
        f"Multiply coordinates by {scale:.2f} to map to original image.]"
    )


def _conversion_hint(from_mime: Optional[str], to_mime: str) -> Optional[str]:
    if not from_mime or from_mime == to_mime:
        return None
    return f"[Image converted from {from_mime} to {to_mime}.]"


async def _normalize_image(bytes_: bytes, mime_type: str):
    """Normalize to a supported inline format, converting to PNG if needed."""
    normalized_mime = _normalize_supported_image_mime_type(mime_type)
    if normalized_mime:
        return {"bytes": bytes_, "mime_type": normalized_mime, "converted_from": None}

    png_bytes = await asyncio.to_thread(convert_image_bytes_to_png, bytes_)
    if not png_bytes:
        return None
    return {"bytes": png_bytes, "mime_type": "image/png", "converted_from": _base_mime_type(mime_type)}


async def process_image(
    bytes_: bytes,
    mime_type: str,
    auto_resize_images: bool = True,
    resize_options: Optional[ImageResizeOptions] = None,
) -> ProcessImageResult:
    """Normalize + (optionally) resize one image for inline provider use.

    Mirrors pi's `processImage`: unsupported/undecodable input and images that
    can't get below the size limit are omitted with pi's exact messages.
    """
    normalized = await _normalize_image(bytes_, mime_type)
    if not normalized:
        return ProcessImageResult(
            ok=False,
            message="[Image omitted: could not be converted to a supported inline image format.]",
        )

    if auto_resize_images:
        resized = await resize_image(normalized["bytes"], normalized["mime_type"], resize_options)
        if not resized:
            return ProcessImageResult(
                ok=False,
                message="[Image omitted: could not be resized below the inline image size limit.]",
            )
        hints: List[str] = []
        converted = _conversion_hint(normalized["converted_from"], resized.mime_type)
        if converted:
            hints.append(converted)
        dimension_note = format_dimension_note(resized)
        if dimension_note:
            hints.append(dimension_note)
        return ProcessImageResult(ok=True, data=resized.data, mime_type=resized.mime_type, hints=hints)

    hints = []
    converted = _conversion_hint(normalized["converted_from"], normalized["mime_type"])
    if converted:
        hints.append(converted)
    return ProcessImageResult(
        ok=True,
        data=base64.b64encode(normalized["bytes"]).decode("utf-8"),
        mime_type=normalized["mime_type"],
        hints=hints,
    )
