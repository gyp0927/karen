"""Tests for constrained sampling (strict JSON schema + grammar tools)."""

import pytest

from karen_ai import Tool
from karen_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    UnsupportedStrictJsonSchemaError,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    make_strict_json_schema,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from karen_ai.types import GrammarSampling, JsonSchemaSampling


def _schema(**overrides):
    base = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }
    base.update(overrides)
    return base


def test_make_strict_json_schema_requires_all_properties():
    strict = make_strict_json_schema(_schema())
    assert strict["type"] == "object"
    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == {"query", "limit"}
    # Non-required property becomes nullable.
    assert strict["properties"]["limit"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    # Required property stays as-is.
    assert strict["properties"]["query"] == {"type": "string"}


def test_make_strict_json_schema_keeps_nullable_optional_as_is():
    strict = make_strict_json_schema(
        {
            "type": "object",
            "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        }
    )
    assert strict["properties"]["note"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert strict["required"] == ["note"]


def test_make_strict_json_schema_rejects_unsupported_keys():
    with pytest.raises(UnsupportedStrictJsonSchemaError):
        make_strict_json_schema({"type": "object", "properties": {"x": {"$ref": "#/$defs/y"}}})
    with pytest.raises(UnsupportedStrictJsonSchemaError):
        make_strict_json_schema({"type": "object", "properties": {}, "oneOf": []})
    with pytest.raises(UnsupportedStrictJsonSchemaError):
        make_strict_json_schema({"type": "object", "properties": {"t": {"items": [{"type": "string"}]}}})


def test_make_strict_json_schema_rejects_non_object_root():
    with pytest.raises(UnsupportedStrictJsonSchemaError):
        make_strict_json_schema({"type": "array", "items": {"type": "string"}})


def test_resolve_json_schema_strict_sampling():
    tool = Tool(
        name="search",
        description="s",
        parameters=_schema(),
        constrained_sampling=JsonSchemaSampling(strict="prefer"),
    )
    assert resolve_json_schema_strict_sampling(tool, supports_strict_mode=True) is True
    assert resolve_json_schema_strict_sampling(tool, supports_strict_mode=False) is None

    bad_tool = Tool(
        name="bad",
        description="b",
        parameters={"type": "object", "properties": {"x": {"$ref": "#/$defs/y"}}, "required": ["x"]},
        constrained_sampling=JsonSchemaSampling(strict="prefer"),
    )
    # Prefer: silently falls back when the schema can't be made strict.
    assert resolve_json_schema_strict_sampling(bad_tool, supports_strict_mode=True) is None
    bad_tool_require = bad_tool.model_copy(update={"constrained_sampling": JsonSchemaSampling(strict="require")})
    with pytest.raises(ValueError, match="requires JSON-schema constrained sampling"):
        resolve_json_schema_strict_sampling(bad_tool_require, supports_strict_mode=True)
    with pytest.raises(ValueError, match="strict tools are unsupported"):
        resolve_json_schema_strict_sampling(
            Tool(
                name="req",
                description="r",
                parameters=_schema(),
                constrained_sampling=JsonSchemaSampling(strict="require"),
            ),
            supports_strict_mode=False,
        )


def _grammar_tool():
    return Tool(
        name="run",
        description="run code",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        constrained_sampling=GrammarSampling(
            variants={"openai_lark": "start: /.+/", "openai_regex": ".+"}
        ),
    )


def test_resolve_grammar_constrained_sampling():
    tool = _grammar_tool()
    assert resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools=False) is None
    grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools=True)
    assert grammar is not None
    assert grammar.format == "lark"  # lark preferred over regex
    assert grammar.input_property == "code"


def test_resolve_grammar_constrained_sampling_requires_variant():
    tool = _grammar_tool().model_copy(update={"constrained_sampling": GrammarSampling(variants={})})
    with pytest.raises(ValueError, match="no supported grammar variant"):
        resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools=True)


def test_resolve_grammar_constrained_sampling_validates_schema():
    tool = _grammar_tool().model_copy(
        update={"parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}}
    )
    with pytest.raises(ValueError, match="cannot use grammar constrained sampling"):
        resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools=True)


def test_create_grammar_tool_input_properties():
    props = create_grammar_tool_input_properties([_grammar_tool()], supports_openai_grammar_tools=True)
    assert props == {"run": "code"}
    assert create_grammar_tool_input_properties([_grammar_tool()], supports_openai_grammar_tools=False) == {}


def test_append_grammar_tool_input_json_delta():
    buffer = GrammarToolInputJsonBuffer()
    assert append_grammar_tool_input_json_delta(buffer, "input", "hel", False) == '{"input":"hel'
    assert append_grammar_tool_input_json_delta(buffer, "input", "hello", False) == "lo"
    # No change, no close → no delta.
    assert append_grammar_tool_input_json_delta(buffer, "input", "hello", False) is None
    assert append_grammar_tool_input_json_delta(buffer, "input", 'hello "hi"', False) == ' \\"hi\\"'
    assert append_grammar_tool_input_json_delta(buffer, "input", 'hello "hi"!', True) == '!"}'
    assert buffer.closed
    # Closing again with identical input is a no-op.
    assert append_grammar_tool_input_json_delta(buffer, "input", 'hello "hi"!', True) is None
    # Changing after close is an error.
    with pytest.raises(ValueError, match="changed after it was closed"):
        append_grammar_tool_input_json_delta(buffer, "input", "different", True)
    with pytest.raises(ValueError, match="non-monotonically"):
        append_grammar_tool_input_json_delta(GrammarToolInputJsonBuffer(input="abc"), "input", "ab", False)
