"""Tool-call argument validation tests (port of pi-ai's test/validation.test.ts).

The TypeBox-specific cases (Function-constructor fallback, generated-check
inspection) have no Python analogue and are not ported; TypeBox `Type.Object`
builders become their plain JSON-schema equivalents.
"""

import pytest

from karen_ai import Tool, ToolCall, validate_tool_arguments, validate_tool_call


def _tool(parameters, name="echo"):
    return Tool(name=name, description="Echo tool", parameters=parameters)


def _call(arguments, name="echo"):
    return ToolCall(id="tool-1", name=name, arguments=arguments)


def _with_plain_schema(schema, value):
    return _tool(
        {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    ), _call({"value": value})


def test_coerces_plain_json_schemas_with_primitive_rules():
    passing = [
        ({"type": "number"}, "42", 42),
        ({"type": "number"}, True, 1),
        ({"type": "number"}, None, 0),
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"type": "boolean"}, "false", False),
        ({"type": "boolean"}, 1, True),
        ({"type": "boolean"}, 0, False),
        ({"type": "string"}, None, ""),
        ({"type": "string"}, True, "true"),
        ({"type": "null"}, "", None),
        ({"type": "null"}, 0, None),
        ({"type": "null"}, False, None),
        ({"type": ["number", "string"]}, "1", "1"),
        ({"type": ["boolean", "number"]}, "1", 1),
    ]
    for schema, value, expected in passing:
        tool, tool_call = _with_plain_schema(schema, value)
        assert validate_tool_arguments(tool, tool_call) == {"value": expected}, (schema, value)


def test_rejects_invalid_coercions():
    failing = [
        ({"type": "boolean"}, "1"),
        ({"type": "boolean"}, "0"),
        ({"type": "null"}, "null"),
        ({"type": "integer"}, "42.1"),
    ]
    for schema, value in failing:
        tool, tool_call = _with_plain_schema(schema, value)
        with pytest.raises(ValueError, match="Validation failed"):
            validate_tool_arguments(tool, tool_call)


def test_treats_null_as_omission_for_optional_non_nullable_properties():
    tool = _tool(
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "number"},
                "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "metadata": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            },
            "required": ["path", "metadata"],
        }
    )
    tool_call = _call({"path": "file.txt", "offset": None, "nullable": None, "metadata": {"enabled": None}})
    assert validate_tool_arguments(tool, tool_call) == {
        "path": "file.txt",
        "nullable": None,
        "metadata": {},
    }


def test_preserves_optional_nulls_whose_referenced_schema_is_nullable():
    tool = _tool(
        {
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
        }
    )
    assert validate_tool_arguments(tool, _call({"value": None})) == {"value": None}


def test_preserves_value_matching_a_nullable_union_arm():
    tool = _tool(
        {"type": "object", "properties": {"value": {"anyOf": [{"type": "number"}, {"type": "null"}]}}}
    )
    assert validate_tool_arguments(tool, _call({"value": None})) == {"value": None}


def test_preserves_value_matching_a_oneof_nullable_arm():
    tool, tool_call = _with_plain_schema({"oneOf": [{"type": "number"}, {"type": "null"}]}, None)
    assert validate_tool_arguments(tool, tool_call) == {"value": None}


def test_coerces_nullable_unions_when_value_matches_no_arm():
    tool, tool_call = _with_plain_schema({"anyOf": [{"type": "number"}, {"type": "null"}]}, "42")
    assert validate_tool_arguments(tool, tool_call) == {"value": 42}


def test_accepts_null_for_nullable_array_schemas_with_items():
    tool, tool_call = _with_plain_schema({"type": ["array", "null"], "items": {"type": "string"}}, None)
    assert validate_tool_arguments(tool, tool_call) == {"value": None}


def test_validate_tool_call_finds_the_tool_by_name():
    tool = _tool({"type": "object", "properties": {}})
    assert validate_tool_call([tool], _call({})) == {}
    with pytest.raises(ValueError, match='Tool "missing" not found'):
        validate_tool_call([tool], _call({}, name="missing"))


def test_error_message_format():
    tool = _tool(
        {"type": "object", "properties": {"count": {"type": "number"}}, "required": ["count"]}
    )
    with pytest.raises(ValueError) as excinfo:
        validate_tool_arguments(tool, _call({"count": "nope"}))
    message = str(excinfo.value)
    assert message.startswith('Validation failed for tool "echo":')
    assert "- count: 'nope' is not of type 'number'" in message
    assert "Received arguments:" in message


def test_required_path_names_the_missing_property():
    tool = _tool(
        {"type": "object", "properties": {"count": {"type": "number"}}, "required": ["count"]}
    )
    with pytest.raises(ValueError) as excinfo:
        validate_tool_arguments(tool, _call({}))
    assert "- count: 'count' is a required property" in str(excinfo.value)
