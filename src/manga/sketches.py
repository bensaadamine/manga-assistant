"""Turning drawings into ControlNet inputs, and cleaning reference images.

Polarity: ControlNet wants WHITE lines on a BLACK background. Paper sketches are
black on white, so everything here normalises to the former.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

SIZE = (512, 768)


def load_sketch(path: str | Path, size: tuple[int, int] = SIZE) -> Image.Image:
    """A clean digital outline. Auto-inverts if it is black-on-white."""
    img = Image.open(path).convert("L").resize(size, Image.LANCZOS)
    if np.array(img).mean() > 127:
        img = Image.eval(img, lambda p: 255 - p)
    return img.convert("RGB")


def load_photo_sketch(
    path: str | Path,
    size: tuple[int, int] = SIZE,
    threshold: int = 150,
    thicken: bool = True,
) -> Image.Image:
    """A PHOTO of a paper sketch: kill paper tone and uneven lighting, then binarize.

    threshold too high -> paper shows through as grey blotches, lower it
    threshold too low  -> thin strokes break up, raise it
    """
    img = Image.open(path).convert("L").resize(size, Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    if thicken:
        img = img.filter(ImageFilter.MinFilter(3))
    img = img.point(lambda p: 255 if p < threshold else 0)
    return img.convert("RGB")


def outline_from_image(lineart_detector, image: Image.Image, size: tuple[int, int] = SIZE):
    """Extract an anime-lineart outline from an existing finished drawing."""
    return lineart_detector(
        image.convert("RGB").resize(size), detect_resolution=512, image_resolution=768
    ).resize(size)


def whiten_background(image: Image.Image, threshold: int = 40, denoise: bool = False) -> Image.Image:
    """Flood-fill the background to pure white, starting from the borders.

    Only touches pixels connected to the edge, so greys inside the character survive.
    IP-Adapter copies EVERYTHING in a reference - background, props, ink splatter -
    so this is not optional.

    denoise=True first applies a median filter, which helps on grainy renders but
    SMEARS SCREENTONE. Leave it off for real manga art.
    """
    img = image.convert("RGB").copy()
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(3))
    w, h = img.size
    for xy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1)]:
        ImageDraw.floodfill(img, xy, (255, 255, 255), thresh=threshold)
    return img


def load_reference(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
