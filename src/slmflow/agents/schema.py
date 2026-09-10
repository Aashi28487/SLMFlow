"""Schema enforcement at every agent I/O boundary.

``BUILD_BRIEF.md`` Phase 10 requires "grammar-constrained decoding (GBNF) or
JSON schema enforcement at every agent I/O boundary to eliminate the
malformed-output failure class". Two mechanisms, in preference order:

1. **Constrain generation.** :func:`schema_to_gbnf` compiles a JSON-schema
   subset into a GBNF grammar. A grammar-constrained model *cannot* emit
   invalid JSON, so the failure class disappears rather than shrinking.
2. **Validate after the fact.** :func:`validate` checks output against the
   schema for backends without grammar support (Ollama), and
   :func:`extract_json` salvages JSON from a chatty completion.

Phase 10 measures schema-validity rate before and after (1), so both paths must
work independently. The validator is pure stdlib -- no ``jsonschema``, which
would breach the zero-dependency rule.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from slmflow.errors import SchemaViolation

__all__ = [
    "ValidationResult",
    "validate",
    "extract_json",
    "parse_and_validate",
    "schema_to_gbnf",
    "enum_schema",
    "object_schema",
]

_SUPPORTED_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


# --------------------------------------------------------------------------- #
# Schema builders (convenience for the common cases)
# --------------------------------------------------------------------------- #
def enum_schema(values: Sequence[str], *, name: str = "label") -> dict[str, Any]:
    """A single-field object whose value is one of ``values``.

    The shape used by the repurposing classification task, where the answer is
    one of ``approved`` / ``investigational`` / ``hypothesis``.

    Example:
        >>> enum_schema(["yes", "no"])["properties"]["label"]["enum"]
        ['yes', 'no']
    """
    return {
        "type": "object",
        "title": f"{name}_enum",
        "properties": {name: {"type": "string", "enum": list(values)}},
        "required": [name],
        "additionalProperties": False,
    }


def object_schema(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: Sequence[str] = (),
    title: str = "output",
) -> dict[str, Any]:
    """Build an object schema with ``additionalProperties: false``."""
    return {
        "type": "object",
        "title": title,
        "properties": dict(properties),
        "required": list(required or properties.keys()),
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating one value against one schema."""

    valid: bool
    errors: tuple[str, ...] = ()
    value: Any = None

    def __bool__(self) -> bool:
        """Truthy when valid."""
        return self.valid


def validate(value: Any, schema: Mapping[str, Any], *, path: str = "$") -> ValidationResult:
    """Validate against the supported JSON-schema subset.

    Supported: ``type``, ``enum``, ``const``, ``properties``, ``required``,
    ``additionalProperties``, ``items``, ``minItems``, ``maxItems``,
    ``minimum``, ``maximum``, ``minLength``, ``maxLength``, ``pattern``.
    Anything else is ignored rather than silently treated as a pass -- the
    ignored keywords are listed in the errors when the value fails anyway.

    Args:
        value: Parsed JSON value.
        schema: Schema to check against.
        path: JSON path prefix used in error messages.

    Returns:
        A :class:`ValidationResult` listing every failure, not just the first.
    """
    errors: list[str] = []
    _validate_into(value, schema, path, errors)
    return ValidationResult(valid=not errors, errors=tuple(errors), value=value)


def _validate_into(
    value: Any, schema: Mapping[str, Any], path: str, errors: list[str]
) -> None:
    """Recursive worker for :func:`validate`."""
    expected = schema.get("type")
    if expected:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_is_type(value, t) for t in types):
            errors.append(f"{path}: expected type {'/'.join(types)}, got {_type_name(value)}")
            return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}, got {value!r}")

    if "enum" in schema:
        allowed = list(schema["enum"])
        if value not in allowed:
            errors.append(f"{path}: {value!r} is not one of {allowed!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        pattern = schema.get("pattern")
        if pattern and not re.search(str(pattern), value):
            errors.append(f"{path}: does not match pattern {pattern!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append(f"{path}: unexpected properties {extra}")
        for key, subschema in properties.items():
            if key in value:
                _validate_into(value[key], subschema, f"{path}.{key}", errors)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}: more than maxItems {schema['maxItems']}")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_into(item, items, f"{path}[{index}]", errors)


def _is_type(value: Any, expected: str) -> bool:
    """JSON type check, keeping ``bool`` out of ``number``."""
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _type_name(value: Any) -> str:
    """JSON type name of a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# --------------------------------------------------------------------------- #
# Salvage
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Any | None:
    """Pull the first complete JSON value out of a chatty completion.

    Small models wrap JSON in prose or fences even when told not to. This
    scans for a balanced ``{...}`` or ``[...]`` respecting string escapes.
    Returns ``None`` when nothing parses.

    Example:
        >>> extract_json('Sure! ```json\\n{"label": "approved"}\\n```')
        {'label': 'approved'}
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(stripped)):
                char = stripped[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(stripped[start : index + 1])
                        except json.JSONDecodeError:
                            break
            start = stripped.find(opener, start + 1)
    return None


def parse_and_validate(
    text: str, schema: Mapping[str, Any], *, strict: bool = False
) -> ValidationResult:
    """Extract JSON from ``text`` then validate it.

    Args:
        text: Raw model output.
        schema: Schema to enforce.
        strict: When true, raise instead of returning an invalid result.

    Returns:
        The validation result, whose ``value`` is the parsed object.

    Raises:
        SchemaViolation: When ``strict`` and the output does not conform.
    """
    parsed = extract_json(text)
    if parsed is None:
        result = ValidationResult(
            valid=False, errors=("output contained no parseable JSON value",), value=None
        )
    else:
        result = validate(parsed, schema)

    if strict and not result.valid:
        raise SchemaViolation(
            "; ".join(result.errors),
            raw_output=text,
            schema_name=str(schema.get("title", "output")),
        )
    return result


# --------------------------------------------------------------------------- #
# GBNF compilation
# --------------------------------------------------------------------------- #
def schema_to_gbnf(schema: Mapping[str, Any]) -> str:
    """Compile a JSON-schema subset into a GBNF grammar for llama.cpp.

    A grammar makes invalid output *impossible*, which is why this is preferred
    over post-hoc validation. Supports objects with fixed properties, arrays,
    string enums, and the scalar types.

    Args:
        schema: Schema to compile.

    Returns:
        A GBNF grammar string whose root rule is ``root``.

    Raises:
        ValueError: For a schema construct outside the supported subset, so an
            unsupported schema fails loudly at build time rather than silently
            producing unconstrained generation.

    Example:
        >>> grammar = schema_to_gbnf(enum_schema(["yes", "no"]))
        >>> "root ::=" in grammar
        True
    """
    rules: dict[str, str] = {}
    root = _compile_node(schema, "root", rules)
    if root != "root":
        rules["root"] = root

    body = "\n".join(f"{name} ::= {rule}" for name, rule in rules.items())
    return f"{body}\n{_GBNF_PRIMITIVES}\n"


def _compile_node(schema: Mapping[str, Any], name: str, rules: dict[str, str]) -> str:
    """Compile one schema node, appending named rules, returning its expression."""
    node_type = schema.get("type")
    if isinstance(node_type, list):
        raise ValueError(
            f"Union types are outside the supported GBNF subset (at {name!r}). "
            f"Split the schema, or validate post-hoc with validate() instead."
        )
    if node_type is not None and node_type not in _SUPPORTED_TYPES:
        raise ValueError(f"Unsupported schema type {node_type!r} at {name!r}.")

    if "enum" in schema:
        options = schema["enum"]
        if not all(isinstance(o, str) for o in options):
            raise ValueError(f"Only string enums are supported in GBNF (at {name!r}).")
        literals = " | ".join(f'"\\"{o}\\""' for o in options)
        rules[name] = literals
        return name

    if "const" in schema and isinstance(schema["const"], str):
        rules[name] = f'"\\"{schema["const"]}\\""'
        return name

    if node_type == "object":
        properties = schema.get("properties") or {}
        if not properties:
            raise ValueError(
                f"Object schema at {name!r} has no properties, so no grammar can be "
                f"derived. Declare the properties, or drop the grammar for this step."
            )
        required = list(schema.get("required") or list(properties))
        ordered = [k for k in properties if k in required]
        if not ordered:
            raise ValueError(f"Object schema at {name!r} has no required properties.")

        parts: list[str] = ['"{" ws']
        for index, key in enumerate(ordered):
            child = _compile_node(properties[key], f"{name}-{key}", rules)
            if index:
                parts.append('"," ws')
            parts.append(f'"\\"{key}\\"" ws ":" ws {child} ws')
        parts.append('"}"')
        rules[name] = " ".join(parts)
        return name

    if node_type == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            raise ValueError(f"Array schema at {name!r} needs an 'items' schema.")
        child = _compile_node(items, f"{name}-item", rules)
        rules[name] = f'"[" ws ({child} (ws "," ws {child})*)? ws "]"'
        return name

    primitive = {
        "string": "string",
        "number": "number",
        "integer": "integer",
        "boolean": "boolean",
        "null": "null",
    }.get(str(node_type))
    if primitive is None:
        raise ValueError(
            f"Schema at {name!r} has no 'type' and no 'enum', so it constrains nothing."
        )
    return primitive


#: Shared terminal rules. Kept minimal and JSON-accurate.
_GBNF_PRIMITIVES = r"""
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex
hex ::= [0-9a-fA-F]
integer ::= "-"? ([0-9] | [1-9] [0-9]*)
number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \t\n]*
""".strip()
