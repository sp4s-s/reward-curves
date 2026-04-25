"""YAML/JSON config I/O and config-merging helpers."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=float)


def deep_merge(base: Dict[str, Any], *overlays: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge. Right-most overlay wins for scalar collisions."""
    out = deepcopy(base)
    for overlay in overlays:
        _merge_into(out, overlay)
    return out


def _merge_into(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _merge_into(dst[k], v)
        else:
            dst[k] = deepcopy(v)


def load_layered_yaml(paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Load a list of YAML files and deep-merge them in order."""
    out: Dict[str, Any] = {}
    for p in paths:
        out = deep_merge(out, load_yaml(p))
    return out


__all__ = [
    "load_yaml",
    "dump_yaml",
    "load_json",
    "dump_json",
    "deep_merge",
    "load_layered_yaml",
]
