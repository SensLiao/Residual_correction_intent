"""JSON configuration files: ``_base_`` inheritance, ``key=value`` overrides and ``${env:VAR}`` values.

A configuration is one JSON object. On top of plain JSON:

* ``"_base_"`` names a file, or a list of files, relative to the file that holds it. The bases are loaded first
  (recursively) and merged in order, then the file's own keys are merged on top. Objects merge key by key; lists and
  scalars are replaced. A file that is its own base, directly or through others, is an error.
* Overrides ``a.b.c=value`` (from the command line) are applied next. The value is parsed as JSON and kept as the
  raw string when it is not valid JSON (``name=sirb`` gives ``"sirb"``, ``name='"12"'`` gives ``"12"``). A plain
  key must already exist, which catches misspelt keys; ``+a.b.c=value`` may add a new key.
* ``${env:VAR}`` and ``${env:VAR,default}`` inside strings are resolved last. A variable that is unset or empty
  takes the default; without a default it is an error that names the configuration key.

This is the only module of the ``sirb`` package that reads environment variables. Duplicate keys and the non-finite
constants ``NaN`` and ``Infinity`` are refused in files and in overrides.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["BASE_KEY", "ConfigError", "add_config_arguments", "apply_overrides", "get", "load_config",
           "merge_configs", "save_config"]

#: Key that names the base file(s) of a configuration file.
BASE_KEY = "_base_"

_ENV_OPENING = "${env:"
_ENV_REFERENCE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")
_MISSING = object()


class ConfigError(ValueError):
    """A configuration file, an override or an environment reference is invalid."""


def load_config(path: str | os.PathLike, overrides: Sequence[str] = ()) -> dict:
    """Load a configuration file with its bases, apply ``overrides`` and resolve ``${env:...}`` (module doc).

    Returns a new plain ``dict``; neither the files nor ``overrides`` are modified.
    """
    config = _load_file(Path(path).resolve(), chain=())
    config = apply_overrides(config, overrides)
    return _resolve_env(config, key_path=())


def merge_configs(base: Mapping, override: Mapping) -> dict:
    """Deep merge: ``override`` wins, objects merge key by key, lists and scalars are replaced.

    Returns a new ``dict`` that shares no object with the inputs.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: Mapping, overrides: Sequence[str]) -> dict:
    """Apply ``key=value`` / ``+key=value`` overrides in order (module doc) to a copy of ``config``."""
    if isinstance(overrides, (str, bytes)):
        raise TypeError("overrides must be a sequence of 'key=value' strings, not a single string")
    result = copy.deepcopy(dict(config))
    for item in overrides:
        segments, may_add, value = _parse_override(item)
        _set_key(result, segments, value, may_add=may_add, item=item)
    return result


def save_config(config: Mapping, path: str | os.PathLike) -> None:
    """Write ``config`` as JSON: sorted keys, two-space indentation, UTF-8, LF line endings, final newline."""
    text = json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def get(config: Mapping, key: str, default: Any = _MISSING) -> Any:
    """Value at the dotted ``key`` (``"train.iterations"``); ``default`` when it is missing, else ``KeyError``."""
    node: Any = config
    for segment in key.split("."):
        if isinstance(node, Mapping) and segment in node:
            node = node[segment]
        elif default is _MISSING:
            raise KeyError(f"configuration has no key {key!r}")
        else:
            return default
    return node


def add_config_arguments(parser: argparse.ArgumentParser, *, default: str | None = None) -> argparse.ArgumentParser:
    """Add ``--config`` (required unless ``default`` is given) and trailing ``key=value`` overrides to ``parser``.

    The overrides are positional: write them after the options, e.g.
    ``--config sirb/configs/paths.json paths.results=runs +train.extra=1``. Pass ``args.config`` and
    ``args.overrides`` to :func:`load_config`.
    """
    parser.add_argument("--config", required=default is None, default=default,
                        help="JSON configuration file" + ("" if default is None else f" (default: {default})"))
    parser.add_argument("overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="configuration overrides after the options, e.g. train.iterations=1000; the value is "
                             "parsed as JSON; a leading + (+key=value) may add a key that does not exist yet")
    return parser


# ------------------------------------------------------------------------------------------------ internals
def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    document: dict = {}
    for key, value in pairs:
        if key in document:
            raise ConfigError(f"duplicate key {key!r}")
        document[key] = value
    return document


def _reject_constant(token: str) -> float:
    raise ConfigError(f"non-finite number {token} is not allowed")


def _read_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file not found: {path}") from error
    try:
        document = json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: invalid JSON: {error}") from error
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from error
    if not isinstance(document, dict):
        raise ConfigError(f"{path}: a configuration must be a JSON object, not {type(document).__name__}")
    return document


def _load_file(path: Path, chain: tuple[Path, ...]) -> dict:
    if path in chain:
        raise ConfigError(f"{BASE_KEY} cycle: " + " -> ".join(str(p) for p in (*chain, path)))
    document = _read_json(path)
    bases = document.pop(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(base, str) and base for base in bases):
        raise ConfigError(f"{path}: {BASE_KEY} must be a file name or a list of file names, not {bases!r}")
    merged: dict = {}
    for base in bases:
        merged = merge_configs(merged, _load_file((path.parent / base).resolve(), (*chain, path)))
    return merge_configs(merged, document)


def _parse_override(item: str) -> tuple[list[str], bool, Any]:
    if not isinstance(item, str) or "=" not in item:
        raise ConfigError(f"an override must look like key=value, not {item!r}")
    key, raw = item.split("=", 1)
    key = key.strip()
    may_add = key.startswith("+")
    if may_add:
        key = key[1:]
    segments = key.split(".")
    if not all(segments):
        raise ConfigError(f"override {item!r}: the key has an empty part")
    return segments, may_add, _parse_value(raw, item)


def _parse_value(raw: str, item: str) -> Any:
    try:
        return json.loads(raw, parse_constant=_reject_constant)
    except json.JSONDecodeError:
        return raw
    except ConfigError as error:
        raise ConfigError(f"override {item!r}: {error}") from error


def _set_key(config: dict, segments: list[str], value: Any, *, may_add: bool, item: str) -> None:
    node = config
    for depth, segment in enumerate(segments):
        name = ".".join(segments[:depth + 1])
        last = depth == len(segments) - 1
        if segment not in node and not may_add:
            raise ConfigError(f"override {item!r}: unknown key {name!r} (write +{'.'.join(segments)}=... to add it)")
        if last:
            node[segment] = value
            return
        child = node.setdefault(segment, {})
        if not isinstance(child, dict):
            raise ConfigError(f"override {item!r}: {name!r} is a {type(child).__name__}, not an object")
        node = child


def _format_key(key_path: Iterable[str | int]) -> str:
    text = ""
    for part in key_path:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text or "<root>"


def _resolve_env(value: Any, key_path: tuple[str | int, ...]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item, (*key_path, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item, (*key_path, index)) for index, item in enumerate(value)]
    if isinstance(value, str) and _ENV_OPENING in value:
        return _substitute(value, _format_key(key_path))
    return value


def _substitute(text: str, name: str) -> str:
    if _ENV_OPENING in _ENV_REFERENCE.sub("", text):
        raise ConfigError(f"{name}: malformed environment reference in {text!r}; "
                          "write ${env:VAR} or ${env:VAR,default}")

    def replace(match: re.Match) -> str:
        variable, default = match.group(1), match.group(2)
        value = os.environ.get(variable, "")
        if value:
            return value
        if default is not None:
            return default
        raise ConfigError(f"{name}: environment variable {variable} is not set and has no default "
                          f"(set it, or write ${{env:{variable},default}})")

    return _ENV_REFERENCE.sub(replace, text)
