"""Prompt assembly.

Two rules learned the hard way during the POC:

1. Weights like (monochrome:1.6) do nothing in plain diffusers. They only work
   because every prompt goes through compel.
2. No color words in a monochrome prompt. "red ribbon" drags the model back
   toward color - describe shape instead.
"""
from __future__ import annotations

from .config import Config


def build(cfg: Config, pose: str, description: str, extra: str = "") -> str:
    """style + quality + who + what they are doing + background."""
    p = cfg.prompts
    parts = [p.style, p.quality, description, pose]
    if extra:
        parts.append(extra)
    parts.append(p.background)
    return ", ".join(x.strip().rstrip(",") for x in parts if x)


def negative(cfg: Config, extra: str = "") -> str:
    neg = " ".join(cfg.prompts.negative.split())
    return f"{neg}, {extra}" if extra else neg


def character_prompt(cfg: Config, pose: str) -> str:
    """Prompt for the character defined in a per-artist config."""
    return build(cfg, pose, cfg.character.description)


def character_negative(cfg: Config) -> str:
    """Negative prompt, with any props the reference carries but the outline does not."""
    props = cfg.character.get("negative_props", "")
    return negative(cfg, props)
