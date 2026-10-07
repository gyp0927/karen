"""`to_llm_content` — pi's `test/content.test.ts`, expectation for expectation."""

from karen_mcp import CallToolResult, to_llm_content


def test_converts_mcp_tool_result_content_for_an_llm():
    content = to_llm_content(
        {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "data": "aW1n", "mimeType": "image/png"},
                {"type": "audio", "data": "YQ==", "mimeType": "audio/wav"},
                {"type": "resource_link", "uri": "file:///a.txt", "name": "a.txt"},
                {"type": "resource", "resource": {"uri": "file:///b.txt", "text": "inline"}},
                {
                    "type": "resource",
                    "resource": {"uri": "file:///c.png", "blob": "Yw==", "mimeType": "image/png"},
                },
                {"type": "resource", "resource": {"uri": "file:///d.bin", "blob": "ZA=="}},
            ]
        }
    )

    assert content == [
        {"type": "text", "text": "hello"},
        {"type": "image", "data": "aW1n", "mimeType": "image/png"},
        {"type": "text", "text": "[audio audio/wav omitted]"},
        {"type": "text", "text": "a.txt: file:///a.txt"},
        {"type": "text", "text": "inline"},
        {"type": "image", "data": "Yw==", "mimeType": "image/png"},
        {"type": "text", "text": "[binary resource file:///d.bin (unknown type) omitted]"},
    ]


def test_falls_back_to_structured_content_when_there_are_no_content_blocks():
    assert to_llm_content({"content": [], "structuredContent": {"n": 1}}) == [
        {"type": "text", "text": '{\n  "n": 1\n}'}
    ]


def test_keeps_content_blocks_when_both_content_and_structured_content_are_present():
    content = to_llm_content({"content": [{"type": "text", "text": "hello"}], "structuredContent": {"n": 1}})
    assert content == [{"type": "text", "text": "hello"}]


def test_accepts_a_typed_result_and_reports_unknown_block_types():
    result = CallToolResult(content=[{"type": "video", "data": "x"}], isError=False)
    assert to_llm_content(result) == [{"type": "text", "text": "[unsupported MCP content video]"}]
    # A typed result with no content at all is an empty projection, not an error.
    assert to_llm_content(CallToolResult()) == []
