"""Which series exist, which LoRA checkpoints each has, and which one is promoted.

The model registry is `configs/registry.yaml`, tracked by git:

    bungo:
      lora: models/loras/bungo/bungo_style_v1.safetensors
      scale: 1.0
      promoted: 2026-11-05T10:30:00
      git: 7e118a7

Promoting a checkpoint is therefore a one-line git diff, and the weights themselves
stay in DVC. The API serves the promoted LoRA when a request names only the series.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from .config import ROOT


def _git(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ series

def series(root: Path = ROOT) -> dict[str, dict]:
    """Every config with a top-level `series:` block -> {name: {config, trigger}}.

    Read raw, not through config.load, so eval configs (which only `extends:` a
    series config's parent) are not mistaken for series.
    """
    out = {}
    for p in sorted((Path(root) / "configs").glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        s = data.get("series")
        if isinstance(s, dict) and s.get("name"):
            out[s["name"]] = {"config": p.name, "trigger": s.get("trigger", "")}
    return out


def checkpoints(name: str, root: Path = ROOT) -> list[str]:
    """LoRA files for a series, repo-relative, in step order (zero-padded names sort right)."""
    d = Path(root) / "models" / "loras" / name
    files = sorted(d.glob("*.safetensors"))
    return [p.relative_to(root).as_posix() for p in files]


def safe_lora_path(rel: str, root: Path = ROOT) -> Path:
    """Resolve a LoRA path and refuse anything outside models/loras (API input)."""
    base = (Path(root) / "models" / "loras").resolve()
    p = (Path(root) / rel).resolve()
    if base not in p.parents or p.suffix != ".safetensors":
        raise ValueError(f"not a LoRA under models/loras: {rel}")
    if not p.exists():
        raise FileNotFoundError(rel)
    return p


# ---------------------------------------------------------------- registry

def _file(root: Path) -> Path:
    return Path(root) / "configs" / "registry.yaml"


def load(root: Path = ROOT) -> dict[str, dict]:
    f = _file(root)
    return (yaml.safe_load(f.read_text(encoding="utf-8")) or {}) if f.exists() else {}


def promoted(name: str, root: Path = ROOT) -> dict | None:
    return load(root).get(name)


def promote(name: str, lora: str, scale: float = 1.0, root: Path = ROOT) -> dict:
    """Make `lora` the checkpoint served for `name`."""
    safe_lora_path(lora, root)
    reg = load(root)
    reg[name] = {"lora": lora, "scale": float(scale),
                 "promoted": datetime.now().isoformat(timespec="seconds"),
                 "git": _git(root)}
    header = ("# Model registry: the LoRA served per series. Written by the app's\n"
              "# 'Promote' button; weights live in DVC, this file lives in git.\n")
    _file(root).write_text(header + yaml.safe_dump(reg, sort_keys=True), encoding="utf-8")
    return reg[name]
