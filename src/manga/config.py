"""YAML config loading with single-level inheritance via an `extends:` key."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """Dict with attribute access, so cfg.generation.cn_scale works."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load(name: str = "base.yaml") -> Config:
    """Load a config by file name, resolving one level of `extends:`."""
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / name

    data = yaml.safe_load(path.read_text())
    parent = data.pop("extends", None)
    if parent:
        data = _deep_merge(load(parent), data)
    return Config(data)
