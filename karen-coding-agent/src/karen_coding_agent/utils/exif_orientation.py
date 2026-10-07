"""EXIF orientation handling for inline images (pi's `utils/exif-orientation.ts`).

pi reads the orientation tag straight out of the JPEG/WebP container and then
applies it with photon pixel ops. karen keeps the exact same container parsing
(JPEG APP1 / WebP EXIF chunk -> TIFF IFD tag 0x0112) but applies the transform
with Pillow, which already exposes every one of the eight EXIF orientations as
a transpose operation.
"""

from __future__ import annotations

from typing import Optional


def _read_orientation_from_tiff(data: bytes, tiff_start: int) -> int:
    if tiff_start + 8 > len(data):
        return 1

    byte_order = (data[tiff_start] << 8) | data[tiff_start + 1]
    le = byte_order == 0x4949

    def read16(pos: int) -> int:
        if le:
            return data[pos] | (data[pos + 1] << 8)
        return (data[pos] << 8) | data[pos + 1]

    def read32(pos: int) -> int:
        if le:
            return data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16) | (data[pos + 3] << 24)
        return ((data[pos] << 24) | (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3]) & 0xFFFFFFFF

    ifd_offset = read32(tiff_start + 4)
    ifd_start = tiff_start + ifd_offset
    if ifd_start + 2 > len(data):
        return 1

    entry_count = read16(ifd_start)
    for i in range(entry_count):
        entry_pos = ifd_start + 2 + i * 12
        if entry_pos + 12 > len(data):
            return 1
        if read16(entry_pos) == 0x0112:
            value = read16(entry_pos + 8)
            return value if 1 <= value <= 8 else 1

    return 1


def _has_exif_header(data: bytes, offset: int) -> bool:
    return (
        data[offset] == 0x45
        and data[offset + 1] == 0x78
        and data[offset + 2] == 0x69
        and data[offset + 3] == 0x66
        and data[offset + 4] == 0x00
        and data[offset + 5] == 0x00
    )


def _find_jpeg_tiff_offset(data: bytes) -> int:
    offset = 2
    while offset < len(data) - 1:
        if data[offset] != 0xFF:
            return -1
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        if marker == 0xE1:
            if offset + 4 >= len(data):
                return -1
            segment_start = offset + 4
            if segment_start + 6 > len(data):
                return -1
            if _has_exif_header(data, segment_start):
                return segment_start + 6
        if offset + 4 > len(data):
            return -1
        length = (data[offset + 2] << 8) | data[offset + 3]
        offset += 2 + length
    return -1


def _find_webp_tiff_offset(data: bytes) -> int:
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = (
            data[offset + 4] | (data[offset + 5] << 8) | (data[offset + 6] << 16) | (data[offset + 7] << 24)
        )
        data_start = offset + 8
        if chunk_id == b"EXIF":
            if data_start + chunk_size > len(data):
                return -1
            # Some WebP files have an "Exif\0\0" prefix before the TIFF header.
            tiff_start = data_start + 6 if chunk_size >= 6 and _has_exif_header(data, data_start) else data_start
            return tiff_start
        # RIFF chunks are padded to even size.
        offset = data_start + chunk_size + (chunk_size % 2)
    return -1


def get_exif_orientation(data: bytes) -> int:
    """Read the EXIF orientation tag (1-8) from JPEG/WebP bytes; 1 when absent."""
    tiff_offset = -1

    # JPEG: starts with FF D8.
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        tiff_offset = _find_jpeg_tiff_offset(data)
    # WebP: starts with RIFF....WEBP.
    elif (
        len(data) >= 12
        and data[0] == 0x52
        and data[1] == 0x49
        and data[2] == 0x46
        and data[3] == 0x46
        and data[8] == 0x57
        and data[9] == 0x45
        and data[10] == 0x42
        and data[11] == 0x50
    ):
        tiff_offset = _find_webp_tiff_offset(data)

    if tiff_offset == -1:
        return 1
    return _read_orientation_from_tiff(data, tiff_offset)


def apply_exif_orientation(image: "object", original_bytes: bytes) -> "object":
    """Return `image` transposed to its EXIF orientation.

    `image` is a Pillow ``Image.Image``. Orientation 1 (or an unreadable tag)
    returns the image unchanged, mirroring pi's early return.
    """
    orientation = get_exif_orientation(original_bytes)
    if orientation == 1:
        return image

    from PIL import Image

    transpose = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    if transpose is None:
        return image
    return image.transpose(transpose)
