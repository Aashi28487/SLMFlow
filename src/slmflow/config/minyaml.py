"""A YAML subset parser, so the base install stays dependency-free.

The brief wants config-driven YAML pipelines *and* a core with zero mandatory
native dependencies. PyYAML is pure Python in its fallback path but is still a
third-party dependency, so:

* If PyYAML is installed (``slmflow[yaml]`` or ``[bench]``), it is used.
* Otherwise this parser handles the subset a pipeline config needs.

Supported: nested block mappings, block sequences, inline ``[a, b]`` and
``{k: v}`` flow collections, ``#`` comments, quoted and bare scalars, multi-line
``|`` and ``>`` blocks, ``null``/``~``, booleans, ints, floats, and anchors are
explicitly *rejected* rather than silently mishandled.

Deliberately strict: an unsupported construct raises :class:`ConfigError` with a
line number. A config parser that quietly misreads its input is worse than one
that refuses, because the misreading surfaces as a wrong experiment.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slmflow.errors import ConfigError

__all__ = ["parse_yaml", "load_yaml_file", "using_pyyaml"]

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NULL = {"null", "~", ""}


def using_pyyaml() -> bool:
    """True when PyYAML is available and will be preferred."""
    try:
        import yaml  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def load_yaml_file(path: str | Path) -> Any:
    """Load a YAML file, preferring PyYAML when installed.

    Raises:
        ConfigError: When the file is missing or cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}", path=str(p))
    text = p.read_text(encoding="utf-8")
    try:
        return parse_yaml(text)
    except ConfigError as exc:
        raise ConfigError(f"{exc} [in {p}]", path=str(p)) from exc


def parse_yaml(text: str) -> Any:
    """Parse a YAML subset (or full YAML via PyYAML when installed).

    Args:
        text: Document source.

    Returns:
        The parsed structure: dict, list or scalar.

    Raises:
        ConfigError: On an unsupported construct or a structural error.

    Example:
        >>> parse_yaml("name: demo\\nsteps:\\n  - a\\n  - b")
        {'name': 'demo', 'steps': ['a', 'b']}
    """
    try:
        import yaml  # noqa: PLC0415

        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:  # pragma: no cover - PyYAML's own message
            raise ConfigError(f"YAML parse error: {exc}") from exc
    except ImportError:
        pass

    lines = _tokenise(text)
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ConfigError(
            f"Line {lines[index][2]}: unexpected content at indentation "
            f"{lines[index][0]}; check that nesting is consistent."
        )
    return value


def _tokenise(text: str) -> list[tuple[int, str, int]]:
    """Split into ``(indent, content, line_number)``, dropping blanks/comments."""
    out: list[tuple[int, str, int]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ConfigError(
                f"Line {number}: tab used for indentation. YAML forbids tabs; use spaces."
            )
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("---") or stripped.startswith("..."):
            continue
        if stripped.startswith("&") or stripped.startswith("*") or "<<:" in stripped:
            raise ConfigError(
                f"Line {number}: anchors, aliases and merge keys are not supported by the "
                f"built-in parser. Install PyYAML (`pip install 'slmflow[yaml]'`) or inline "
                f"the value."
            )
        indent = len(raw) - len(raw.lstrip(" "))
        out.append((indent, _strip_comment(stripped), number))
    return out


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment, respecting quotes."""
    in_single = in_double = False
    for i, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1] in " \t":
                return line[:i].rstrip()
    return line


def _parse_block(
    lines: Sequence[tuple[int, str, int]], start: int, indent: int
) -> tuple[Any, int]:
    """Parse a mapping or sequence block at ``indent``, from ``start``."""
    if start >= len(lines):
        return None, start
    if lines[start][1].startswith("- "):
        return _parse_sequence(lines, start, indent)
    if lines[start][1] == "-":
        return _parse_sequence(lines, start, indent)
    return _parse_mapping(lines, start, indent)


def _parse_sequence(
    lines: Sequence[tuple[int, str, int]], start: int, indent: int
) -> tuple[list[Any], int]:
    """Parse a block sequence."""
    items: list[Any] = []
    index = start
    while index < len(lines):
        line_indent, content, number = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ConfigError(f"Line {number}: over-indented sequence item.")
        if not (content == "-" or content.startswith("- ")):
            break

        remainder = content[1:].strip()
        index += 1
        if not remainder:
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_block(lines, index, lines[index][0])
                items.append(value)
            else:
                items.append(None)
            continue

        # "- key: value" starts an inline mapping owned by this item.
        if _looks_like_mapping(remainder):
            child = [(indent + 2, remainder, number)]
            while index < len(lines) and lines[index][0] > indent:
                child.append(lines[index])
                index += 1
            value, consumed = _parse_mapping(child, 0, indent + 2)
            if consumed != len(child):
                raise ConfigError(f"Line {number}: could not parse sequence item mapping.")
            items.append(value)
        else:
            items.append(_scalar(remainder, number))
    return items, index


def _parse_mapping(
    lines: Sequence[tuple[int, str, int]], start: int, indent: int
) -> tuple[dict[str, Any], int]:
    """Parse a block mapping."""
    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line_indent, content, number = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ConfigError(
                f"Line {number}: unexpected indentation {line_indent}, expected {indent}."
            )
        if content.startswith("- "):
            break
        if not _looks_like_mapping(content):
            raise ConfigError(
                f"Line {number}: expected 'key: value', got {content!r}."
            )

        key, _, rest = content.partition(":")
        key = _unquote(key.strip())
        rest = rest.strip()
        if key in mapping:
            raise ConfigError(f"Line {number}: duplicate key {key!r}.")
        index += 1

        if rest in {"|", ">", "|-", ">-"}:
            block_lines: list[str] = []
            block_indent = lines[index][0] if index < len(lines) else indent + 2
            while index < len(lines) and lines[index][0] >= block_indent > indent:
                block_lines.append(lines[index][1])
                index += 1
            joiner = "\n" if rest.startswith("|") else " "
            mapping[key] = joiner.join(block_lines)
            continue

        if rest:
            mapping[key] = _scalar(rest, number)
            continue

        if index < len(lines) and lines[index][0] > indent:
            mapping[key], index = _parse_block(lines, index, lines[index][0])
        elif index < len(lines) and lines[index][1].startswith("- ") and lines[index][0] == indent:
            # A sequence at the same indentation as its key is legal YAML.
            mapping[key], index = _parse_sequence(lines, index, indent)
        else:
            mapping[key] = None
    return mapping, index


def _looks_like_mapping(content: str) -> bool:
    """True when ``content`` is a ``key: value`` pair rather than a scalar."""
    in_single = in_double = False
    depth = 0
    for i, char in enumerate(content):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif in_single or in_double:
            continue
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == ":" and depth == 0:
            return i + 1 == len(content) or content[i + 1] in " \t"
    return False


def _unquote(text: str) -> str:
    """Strip matching surrounding quotes."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _scalar(text: str, line_number: int) -> Any:
    """Coerce a scalar or flow collection."""
    text = text.strip()
    if not text:
        return None

    if text[0] in "[{":
        # Flow collections are JSON-compatible once bare keys are quoted.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return json.loads(_quote_bare_tokens(text))
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"Line {line_number}: could not parse inline collection {text!r}. "
                    f"Use JSON syntax, e.g. [\"a\", \"b\"] or {{\"k\": 1}}."
                ) from exc

    if text[0] in "\"'":
        return _unquote(text)

    lowered = text.lower()
    if lowered in _NULL:
        return None
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _quote_bare_tokens(text: str) -> str:
    """Quote bare words inside a flow collection so ``json`` accepts it."""
    out: list[str] = []
    token: list[str] = []
    in_string = False

    def flush() -> None:
        word = "".join(token).strip()
        token.clear()
        if not word:
            return
        lowered = word.lower()
        if lowered in _TRUE:
            out.append("true")
        elif lowered in _FALSE:
            out.append("false")
        elif lowered in _NULL:
            out.append("null")
        else:
            try:
                float(word)
                out.append(word)
            except ValueError:
                out.append(json.dumps(word))

    for char in text:
        if char == '"':
            in_string = not in_string
            token.append(char)
        elif in_string:
            token.append(char)
        elif char in "[]{},:":
            flush()
            out.append(char)
        else:
            token.append(char)
    flush()
    return "".join(out)
