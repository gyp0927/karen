"""Tool-call argument validation, mirroring pi-ai's `utils/validation.ts`.

pi-ai validates against TypeBox schemas; in Python every tool schema is a plain
JSON Schema dict, so `jsonschema` plays the role of TypeBox's `Compile` and the
"serialized plain JSON schema" coercion path is the only one that exists here
(the `TypeBox.Kind` branch has no Python analogue).
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from jsonschema import ValidationError, validators

from ..types import Tool, ToolCall

JsonSchemaObject = Mapping[str, Any]

_validator_cache: Dict[str, Any] = {}


def _get_schema_types(schema: JsonSchemaObject) -> List[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [t for t in schema_type if isinstance(t, str)]
    return []


def _matches_json_type(value: Any, json_type: str) -> bool:
    # `type(value) is ...` mirrors JavaScript's `typeof`, not Python's type hierarchy
    # (bool is an int subclass here but a distinct typeof there).
    if json_type == "number":
        return type(value) in (int, float)
    if json_type == "integer":
        return type(value) is int or (type(value) is float and value.is_integer())
    if json_type == "boolean":
        return type(value) is bool
    if json_type == "string":
        return type(value) is str
    if json_type == "null":
        return value is None
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return False


def _get_sub_schema_validator(schema: JsonSchemaObject) -> Optional[Any]:
    try:
        return _get_validator(schema)
    except Exception:
        return None


def _js_number(text: str) -> Optional[float]:
    """JavaScript's `Number(text)` for non-empty trimmed strings (hex/octal forms aside)."""
    s = text.strip()
    if not s or "_" in s:  # Python accepts "1_0"; JavaScript does not.
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return None


def _js_string(value: Any) -> str:
    """JavaScript's `String(value)` for scalars."""
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is float and value.is_integer():
        return str(int(value))
    return str(value)


def _coerce_primitive_by_type(value: Any, json_type: str) -> Any:
    if json_type == "number":
        if value is None:
            return 0
        if type(value) is str and value.strip() != "":
            parsed = _js_number(value)
            if parsed is not None and math.isfinite(parsed):
                return parsed
        if type(value) is bool:
            return 1 if value else 0
        return value
    if json_type == "integer":
        if value is None:
            return 0
        if type(value) is str and value.strip() != "":
            parsed = _js_number(value)
            if parsed is not None and float(parsed).is_integer():
                return int(parsed)
        if type(value) is bool:
            return 1 if value else 0
        return value
    if json_type == "boolean":
        if value is None:
            return False
        if type(value) is str:
            if value == "true":
                return True
            if value == "false":
                return False
        if type(value) in (int, float):
            if value == 1:
                return True
            if value == 0:
                return False
        return value
    if json_type == "string":
        if value is None:
            return ""
        if type(value) in (int, float) or type(value) is bool:
            return _js_string(value)
        return value
    if json_type == "null":
        if (type(value) is str and value == "") or (type(value) in (int, float) and value == 0) or value is False:
            return None
        return value
    return value


def _js_strict_equal(left: Any, right: Any) -> bool:
    """JavaScript `===` for scalars: numbers compare by value across int/float, everything
    else by type and value."""
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    return type(left) is type(right) and left == right


def _apply_schema_object_coercion(value: Dict[str, Any], schema: JsonSchemaObject) -> None:
    properties = schema.get("properties")
    defined_keys = set(properties) if properties else set()

    if properties:
        for key, property_schema in properties.items():
            if key not in value:
                continue
            value[key] = _coerce_with_json_schema(value[key], property_schema)

    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        for key, property_value in list(value.items()):
            if key in defined_keys:
                continue
            value[key] = _coerce_with_json_schema(property_value, additional)


def _apply_schema_array_coercion(value: List[Any], schema: JsonSchemaObject) -> None:
    items = schema.get("items")
    if isinstance(items, list):
        for index in range(len(value)):
            if index >= len(items):
                continue
            value[index] = _coerce_with_json_schema(value[index], items[index])
        return
    if isinstance(items, Mapping):
        for index in range(len(value)):
            value[index] = _coerce_with_json_schema(value[index], items)


def _coerce_with_union_schema(value: Any, schemas: Sequence[JsonSchemaObject]) -> Any:
    for schema in schemas:
        validator = _get_sub_schema_validator(schema)
        if validator is not None and validator.is_valid(value):
            return value

    for schema in schemas:
        candidate = copy.deepcopy(value)
        coerced = _coerce_with_json_schema(candidate, schema)
        validator = _get_sub_schema_validator(schema)
        if validator is not None and validator.is_valid(coerced):
            return coerced
    return value


def _coerce_with_json_schema(value: Any, schema: JsonSchemaObject) -> Any:
    next_value = value

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            next_value = _coerce_with_json_schema(next_value, nested)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        next_value = _coerce_with_union_schema(next_value, any_of)

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        next_value = _coerce_with_union_schema(next_value, one_of)

    schema_types = _get_schema_types(schema)
    matches_union_member = len(schema_types) > 1 and any(
        _matches_json_type(next_value, schema_type) for schema_type in schema_types
    )
    if schema_types and not matches_union_member:
        for schema_type in schema_types:
            candidate = _coerce_primitive_by_type(next_value, schema_type)
            if not _js_strict_equal(candidate, next_value):
                next_value = candidate
                break

    if "object" in schema_types and isinstance(next_value, dict):
        _apply_schema_object_coercion(next_value, schema)

    if "array" in schema_types and isinstance(next_value, list):
        _apply_schema_array_coercion(next_value, schema)

    return next_value


def _normalize_optional_nulls(value: Any, schema: JsonSchemaObject) -> None:
    """Delete null values of optional properties whose schema rejects null."""
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index in range(len(value)):
                if index < len(items):
                    _normalize_optional_nulls(value[index], items[index])
        elif isinstance(items, Mapping):
            for item in value:
                _normalize_optional_nulls(item, items)
        return
    if not isinstance(value, dict) or not isinstance(schema.get("properties"), Mapping):
        return

    required = set(schema.get("required") or ())
    for key, property_schema in schema["properties"].items():
        if key not in value:
            continue
        if (
            value[key] is None
            and key not in required
            and not isinstance(property_schema.get("$ref"), str)
            and (_get_sub_schema_validator(property_schema) is None
                 or _get_sub_schema_validator(property_schema).is_valid(None) is False)
        ):
            del value[key]
        else:
            _normalize_optional_nulls(value[key], property_schema)


def _get_validator(schema: JsonSchemaObject) -> Any:
    try:
        key = json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr)
    except (TypeError, ValueError):
        key = repr(sorted(schema.items()))
    cached = _validator_cache.get(key)
    if cached is not None:
        return cached
    validator = validators.validator_for(dict(schema))(dict(schema))
    _validator_cache[key] = validator
    return validator


def _format_validation_path(error: ValidationError) -> str:
    base_path = ".".join(str(part) for part in error.absolute_path)
    if error.validator == "required":
        # jsonschema phrases it as "'name' is a required property".
        message = error.message
        if message.startswith("'") and "' is a required property" in message:
            required_property = message[1 : message.index("' is a required property")]
            return f"{base_path}.{required_property}" if base_path else required_property
    return base_path or "root"


def validate_tool_call(tools: Sequence[Tool], tool_call: ToolCall) -> Any:
    """Find a tool by name and validate the call's arguments against its schema."""
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    """Validate (and coerce, where the schema allows) a call's arguments.

    Raises ValueError with the same formatted message pi-ai produces on failure.
    """
    args = copy.deepcopy(tool_call.arguments)
    _normalize_optional_nulls(args, tool.parameters)

    validator = _get_validator(tool.parameters)
    coerced = _coerce_with_json_schema(args, tool.parameters)
    if coerced is not args:
        if isinstance(args, dict) and isinstance(coerced, dict):
            args.clear()
            args.update(coerced)
        elif validator.is_valid(coerced):
            return coerced

    if validator.is_valid(args):
        return args

    errors = "\n".join(f"  - {_format_validation_path(error)}: {error.message}" for error in validator.iter_errors(args))
    if not errors:
        errors = "Unknown validation error"

    received = json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)
    raise ValueError(f'Validation failed for tool "{tool_call.name}":\n{errors}\n\nReceived arguments:\n{received}')


__all__ = ["validate_tool_arguments", "validate_tool_call"]
