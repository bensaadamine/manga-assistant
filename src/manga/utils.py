"""Small helpers shared across modules."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


def seeds(seed: int, n: int, device: str = "cuda") -> list[torch.Generator]:
    """n generators seeded seed, seed+1, ... so every image is reproducible."""
    return [torch.Generator(device).manual_seed(seed + i) for i in range(n)]


def tag_seeds(images: Sequence[Image.Image], seed: int) -> list[Image.Image]:
    for i, im in enumerate(images):
        im.info["seed"] = seed + i
    return list(images)


def grid(images: Sequence[Image.Image], cols: int = 2) -> Image.Image:
    """Tile images into one picture for side-by-side comparison."""
    images = [im.convert("RGB") for im in images]
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        canvas.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return canvas


def save(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
