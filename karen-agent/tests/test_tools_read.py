"""The read tool (pi's read.ts behaviors), against the local filesystem."""

import base64
import os

import pytest

from karen_agent.tools import ImageProcessingFailed, ProcessedImage, create_read_tool
from karen_agent.tools.path_utils import normalize_tool_path

# A minimal valid 1x1 PNG.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
BMP_BYTES = (
    b"BM" + (58).to_bytes(4, "little") + b"\x00" * 4 + (54).to_bytes(4, "little")
    + (40).to_bytes(4, "little") + (1).to_bytes(4, "little") * 2
    + (1).to_bytes(2, "little") + (24).to_bytes(2, "little") + b"\x00" * 24 + b"pixel"
)


async def _run(tool, **params):
    return await tool.execute("call-1", params, None, None)


def test_normalize_tool_path():
    assert normalize_tool_path("@file.txt") == "file.txt"
    assert normalize_tool_path("a b.txt") == "a b.txt"
    assert normalize_tool_path("plain.txt") == "plain.txt"


async def test_read_text_file(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"hello\nworld\n")
    result = await _run(create_read_tool(str(tmp_path)), path="f.txt")
    assert result.content[0].text == "hello\nworld\n"
    assert result.details is None


async def test_read_with_offset(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"".join(f"l{i}\n".encode() for i in range(1, 11)))
    result = await _run(create_read_tool(str(tmp_path)), path="f.txt", offset=5)
    assert result.content[0].text == "l5\nl6\nl7\nl8\nl9\nl10\n"


async def test_read_with_limit_shows_remaining_hint(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"".join(f"l{i}\n".encode() for i in range(1, 11)))
    result = await _run(create_read_tool(str(tmp_path)), path="f.txt", limit=3)
    # The trailing newline makes an 11th empty line, so 8 lines remain.
    assert result.content[0].text == "l1\nl2\nl3\n\n[8 more lines in file. Use offset=4 to continue.]"


async def test_read_offset_beyond_end(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"l1\nl2\n")
    with pytest.raises(ValueError) as excinfo:
        await _run(create_read_tool(str(tmp_path)), path="f.txt", offset=20)
    assert str(excinfo.value) == "Offset 20 is beyond end of file (3 lines total)"


async def test_read_truncates_by_lines(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"".join(f"x{i}\n".encode() for i in range(1, 3001)))
    result = await _run(create_read_tool(str(tmp_path)), path="f.txt")
    text = result.content[0].text
    assert text.endswith("[Showing lines 1-2000 of 3001. Use offset=2001 to continue.]")
    assert result.details["truncation"]["truncated"] is True
    assert result.details["truncation"]["truncatedBy"] == "lines"


async def test_read_truncates_by_bytes(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"\n".join(b"x" * 600 for _ in range(100)))  # ~60KB
    result = await _run(create_read_tool(str(tmp_path)), path="f.txt")
    text = result.content[0].text
    assert "(50.0KB limit). Use offset=" in text
    assert result.details["truncation"]["truncatedBy"] == "bytes"


async def test_read_first_line_exceeds_limit(tmp_path):
    (tmp_path / "big.txt").write_bytes(b"x" * (60 * 1024))
    result = await _run(create_read_tool(str(tmp_path)), path="big.txt")
    assert result.content[0].text == (
        "[Line 1 is 60.0KB, exceeds 50.0KB limit. Use bash: sed -n '1p' big.txt | head -c 51200]"
    )
    assert result.details["truncation"]["firstLineExceedsLimit"] is True


async def test_read_continuation_offset_after_truncation(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"".join(f"x{i}\n".encode() for i in range(1, 3001)))
    result = await _run(create_read_tool(str(tmp_path)), path="f.txt", offset=2001, limit=2)
    assert result.content[0].text.startswith("x2001\nx2002")


async def test_read_image_png(tmp_path):
    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    result = await _run(create_read_tool(str(tmp_path)), path="img.png")
    assert result.content[0].text == "Read image file [image/png]"
    assert result.content[1].type == "image"
    assert result.content[1].mime_type == "image/png"
    assert base64.b64decode(result.content[1].data) == PNG_BYTES


async def test_read_bmp_omitted_without_processor(tmp_path):
    (tmp_path / "img.bmp").write_bytes(BMP_BYTES)
    result = await _run(create_read_tool(str(tmp_path)), path="img.bmp")
    assert result.content[0].text == (
        "Read image file [image/bmp]\n[Image omitted: configure an imageProcessor to convert BMP images.]"
    )
    assert len(result.content) == 1


async def test_read_image_with_processor(tmp_path):
    async def processor(data, mime_type, auto_resize):
        assert mime_type == "image/bmp"
        assert auto_resize is True
        return ProcessedImage(data=base64.b64encode(data).decode(), mime_type="image/png", hints=["converted to png"])

    (tmp_path / "img.bmp").write_bytes(BMP_BYTES)
    result = await _run(create_read_tool(str(tmp_path), image_processor=processor), path="img.bmp")
    assert result.content[0].text == "Read image file [image/png]\nconverted to png"
    assert result.content[1].mime_type == "image/png"


async def test_read_image_processor_failure(tmp_path):
    async def processor(data, mime_type, auto_resize):
        return ImageProcessingFailed(message="image too large")

    (tmp_path / "img.png").write_bytes(PNG_BYTES)
    result = await _run(create_read_tool(str(tmp_path), image_processor=processor), path="img.png")
    assert result.content[0].text == "Read image file [image/png]\nimage too large"
    assert len(result.content) == 1


async def test_read_at_prefixed_path(tmp_path):
    (tmp_path / "hello.txt").write_bytes(b"hi")
    result = await _run(create_read_tool(str(tmp_path)), path="@hello.txt")
    assert result.content[0].text == "hi"


async def test_read_resolves_am_pm_narrow_space_variant(tmp_path):
    narrow = "Meeting 3 PM.txt"
    (tmp_path / narrow).write_bytes(b"agenda")
    result = await _run(create_read_tool(str(tmp_path)), path="Meeting 3 PM.txt")
    assert result.content[0].text == "agenda"


async def test_read_defaults_to_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel.txt").write_bytes(b"relative")
    result = await _run(create_read_tool(), path="rel.txt")
    assert result.content[0].text == "relative"
