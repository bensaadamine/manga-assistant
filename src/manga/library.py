"""Saved references and outlines, so an artist can reuse them instead of re-uploading.

    data/library/refs/<hash>.png
    data/library/outlines/<hash>.png

Files are named by a hash of their pixels, so uploading the same image twice keeps
one copy. Each kind is capped at `max_items` (configs/base.yaml, `library:`); past
that the least recently USED file goes, not the oldest: picking an item from the
library touches it, so favourites survive.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from PIL import Image

from .config import ROOT

KINDS = ("refs", "outlines")
DEFAULT_MAX = 50


def folder(kind: str, root: Path = ROOT) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    d = Path(root) / "data" / "library" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _flatten(img: Image.Image) -> Image.Image:
    """Transparent PNGs get a white background, everything ends up RGB."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img.convert("RGB")


def add(image: str | Path | Image.Image, kind: str, root: Path = ROOT,
        max_items: int = DEFAULT_MAX) -> Path:
    """Save an image into the library (or touch it if already there), then prune."""
    img = _flatten(Image.open(image) if not isinstance(image, Image.Image) else image)
    digest = hashlib.sha1(img.tobytes() + str(img.size).encode()).hexdigest()[:12]
    path = folder(kind, root) / f"{digest}.png"
    if path.exists():
        touch(path)
    else:
        img.save(path)
    prune(kind, root, max_items)
    return path


def touch(path: str | Path) -> None:
    """Mark as just used: the modification time is the LRU clock."""
    now = time.time()
    os.utime(path, (now, now))


def items(kind: str, root: Path = ROOT) -> list[Path]:
    """Most recently used first."""
    return sorted(folder(kind, root).glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


def prune(kind: str, root: Path = ROOT, max_items: int = DEFAULT_MAX) -> list[Path]:
    """Delete the least recently used files beyond the cap. Returns what was removed."""
    extra = items(kind, root)[max_items:]
    for p in extra:
        p.unlink(missing_ok=True)
    return extra
