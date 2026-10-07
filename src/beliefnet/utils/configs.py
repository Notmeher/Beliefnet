"""Tiny YAML config loader.

Reads a YAML file into a dict; supports flat dot-style overrides on the
command line (e.g. ``--override seed=1 perception.lr=5e-4``). Intentionally
not Hydra: the synthetic pipeline only needs a handful of knobs and a flat
loader keeps debugging trivial. If the SciFact / FEVER work later needs
multi-config composition, Hydra can be swapped in without changing the
config files' content.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.subkey=value`` overrides to a config dict in-place and return it."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got: {ov!r}")
        key, raw = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def _coerce(raw: str) -> Any:
    """Best-effort YAML-style scalar coercion for override values."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
