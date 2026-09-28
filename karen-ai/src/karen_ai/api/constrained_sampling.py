"""Constrained-sampling helpers, mirroring api/constrained-sampling.ts.

Two mechanisms:
- JSON-schema strict mode: rewrite a tool's parameter schema into the strict
  subset providers accept (all properties required, nullable where optional,
  additionalProperties false).
- Grammar-constrained tools: expose a tool as a grammar (lark/regex) whose
  single string argument carries the raw input.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..types import Tool

UNSUPPORTED_STRICT_SCHEMA_KEYS = (
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
)


class UnsupportedStrictJsonSchemaError(Exception):
    pass


def _is_json_schema_object(value: Any) -> bool:
    return isinstance(value, dict)


def _is_structured_schema(schema: Any) -> bool:
    if not _is_json_schema_object(schema):
        return False
    raw_type = schema.get("type")
    types = [raw_type] if isinstance(raw_type, str) else list(raw_type) if isinstance(raw_type, list) else []
    return "object" in types or "array" in types or "properties" in schema or "items" in schema


def _schema_allows_null(schema: Any) -> bool:
    if not _is_json_schema_object(schema):
        return False
    raw_type = schema.get("type")
    if raw_type == "null" or (isinstance(raw_type, list) and "null" in raw_type):
        return True
    if schema.get("const") is None and "const" in schema:
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    any_of = schema.get("anyOf")
    return isinstance(any_of, list) and any(_schema_allows_null(variant) for variant in any_of)


def _make_json_schema_node_strict(schema: Any) -> None:
    if not _is_json_schema_object(schema):
        raise UnsupportedStrictJsonSchemaError("boolean schemas are unsupported")
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if schema.get(key) is not None:
            raise UnsupportedStrictJsonSchemaError(f"{key} schemas are unsupported")

    any_of = schema.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or len(any_of) == 0:
            raise UnsupportedStrictJsonSchemaError("anyOf must contain at least one schema")
        for variant in any_of:
            if _is_structured_schema(variant):
                raise UnsupportedStrictJsonSchemaError("object and array unions are unsupported")
            _make_json_schema_node_strict(variant)

    items = schema.get("items")
    if items is not None:
        if isinstance(items, list):
            raise UnsupportedStrictJsonSchemaError("tuple schemas are unsupported")
        _make_json_schema_node_strict(items)

    is_object_schema = schema.get("type") == "object"
    if schema.get("properties") is not None and not is_object_schema:
        raise UnsupportedStrictJsonSchemaError("properties require type object")
    if not is_object_schema:
        return
    additional = schema.get("additionalProperties")
    if additional is not None and additional is not False:
        raise UnsupportedStrictJsonSchemaError("schema-valued or true additionalProperties is unsupported")
    properties = schema.get("properties")
    if properties is not None and not _is_json_schema_object(properties):
        raise UnsupportedStrictJsonSchemaError("object properties must be a schema map")
    required_raw = schema.get("required")
    if required_raw is not None and (
        not isinstance(required_raw, list) or any(not isinstance(key, str) for key in required_raw)
    ):
        raise UnsupportedStrictJsonSchemaError("object required must be a string array")

    properties = properties or {}
    property_names = list(properties.keys())
    required = set(required_raw) if isinstance(required_raw, list) else set()
    if any(key not in property_names for key in required):
        raise UnsupportedStrictJsonSchemaError("required contains an unknown property")
    for key, prop in properties.items():
        _make_json_schema_node_strict(prop)
        if key not in required and not _schema_allows_null(prop):
            properties[key] = {"anyOf": [prop, {"type": "null"}]}
    schema["required"] = property_names
    schema["additionalProperties"] = False


def make_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a tool schema to the strict subset expected by provider constrained sampling."""
    cloned = copy.deepcopy(schema)
    if not _is_json_schema_object(cloned):
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    _make_json_schema_node_strict(cloned)
    if cloned.get("type") != "object":
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    return cloned


def get_json_schema_tool_parameters(tool: Tool, strict: Optional[bool]) -> Dict[str, Any]:
    return make_strict_json_schema(tool.parameters) if strict is True else tool.parameters


@dataclass
class GrammarConstrainedSampling:
    format: str  # "lark" | "regex"
    definition: str
    input_property: str


@dataclass
class GrammarToolInputJsonBuffer:
    input: str = ""
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: Dict[str, Any], input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> Optional[str]:
    """Advance the synthetic JSON buffer for a grammar tool's string argument.

    Returns the raw JSON fragment to emit as the tool-call delta, or None when
    nothing changed.
    """
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not next_input.startswith(buffer.input):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')

    input_delta = next_input[len(buffer.input):]
    if not close and len(input_delta) == 0:
        return None

    delta = ""
    if not buffer.started:
        delta += "{" + json.dumps(input_property) + ':"'
        buffer.started = True
    delta += json.dumps(input_delta)[1:-1]
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = tool.parameters
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")

    input_property = required[0]
    properties = schema.get("properties") or {}
    if input_property not in properties:
        raise ValueError(f"grammar constrained sampling requires a properties entry for {input_property}")
    if properties[input_property].get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {input_property} must have type string")
    return input_property


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> Optional[bool]:
    config = tool.constrained_sampling
    if not config or getattr(config, "type", None) != "json_schema":
        return None

    if supports_strict_mode:
        try:
            make_strict_json_schema(tool.parameters)
            return True
        except UnsupportedStrictJsonSchemaError as error:
            if config.strict != "require":
                return None
            raise ValueError(
                f'Tool "{tool.name}" requires JSON-schema constrained sampling, but {error}.'
            ) from error
    if config.strict == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool,
    supports_openai_grammar_tools: bool,
) -> Optional[GrammarConstrainedSampling]:
    config = tool.constrained_sampling
    if not config or getattr(config, "type", None) != "grammar":
        return None

    if not supports_openai_grammar_tools:
        return None

    lark_definition = config.variants.get("openai_lark")
    regex_definition = config.variants.get("openai_regex")
    has_lark = isinstance(lark_definition, str) and len(lark_definition.strip()) > 0
    has_regex = isinstance(regex_definition, str) and len(regex_definition.strip()) > 0
    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.'
        )

    try:
        return GrammarConstrainedSampling(
            format="lark" if has_lark else "regex",
            definition=lark_definition if has_lark else regex_definition,
            input_property=_infer_grammar_input_property(tool),
        )
    except ValueError as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.') from error


def create_grammar_tool_input_properties(
    tools: Optional[List[Tool]],
    supports_openai_grammar_tools: bool,
) -> Dict[str, str]:
    properties: Dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar:
            properties[tool.name] = grammar.input_property
    return properties
