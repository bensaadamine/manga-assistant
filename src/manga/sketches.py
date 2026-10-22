"""Turning drawings into ControlNet inputs, and cleaning reference images.

Polarity: ControlNet wants WHITE lines on a BLACK background. Paper sketches are
black on white, so everything here normalises to the former.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

SIZE = (512, 768)



def fit_to(img: Image.Image, size: tuple[int, int] = SIZE, fill=(255, 255, 255)) -> Image.Image:
    """Scale to FIT inside `size`, keeping aspect, then centre on `fill`.

    Never squash. A plain .resize() on a landscape outline compresses the figure
    horizontally, ControlNet traces that squashed silhouette literally, and every
    generation comes out elongated.
    """
    img = img.copy()
    img.thumbnail(size, Image.LANCZOS)
    mode = img.mode if img.mode in ("RGB", "L") else "RGB"
    if mode == "L":
        fill = fill if isinstance(fill, int) else 255
    out = Image.new(mode, size, fill)
    out.paste(img, ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2))
    return out


def load_sketch(path: str | Path, size: tuple[int, int] = SIZE) -> Image.Image:
    """A clean digital outline. Auto-inverts if it is black-on-white."""
    img = fit_to(Image.open(path).convert("L"), size)
    if np.array(img).mean() > 127:
        img = Image.eval(img, lambda p: 255 - p)
    return img.convert("RGB")


def load_photo_sketch(
    path: str | Path,
    size: tuple[int, int] = SIZE,
    block: int = 51,
    C: int = 8,
    min_area: int = 150,
    max_gray: int = 125,
) -> Image.Image:
    """A PHOTO of a paper sketch: pull the pen strokes off creased, unevenly lit paper.

    A single global threshold cannot do this. Where a crease shadow crosses the
    page, one cutoff either swallows the whole sheet as ink or loses the lines in
    the lit half. So threshold LOCALLY, then keep only what looks like drawing:

      * the largest component is the figure's outline, always;
      * anything else must be dark like pen rather than faint like a crease,
        big enough not to be speckle, and lying beside the figure.

    This matters more than it sounds: ControlNet traces every line it is given,
    so a surviving crease becomes a real feature in the output.
    """
    import cv2

    g = np.array(Image.open(path).convert("L"))
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(g, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)

    n, lbl, st, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        return Image.fromarray(binary).convert("RGB")
    main = 1 + int(np.argmax(st[1:, 4]))

    reach = max(8, int(0.06 * max(g.shape)))
    near = cv2.dilate((lbl == main).astype(np.uint8),
                      np.ones((reach, reach), np.uint8)) > 0

    keep = np.zeros(n, bool)
    keep[main] = True
    for i in range(1, n):
        if i == main or st[i, 4] < min_area:
            continue
        m = lbl == i
        if float(g[m].mean()) < max_gray and near[m].mean() > 0.3:
            keep[i] = True

    sel = np.flatnonzero(keep)
    x0 = max(0, int(st[sel, 0].min()) - 12)
    y0 = max(0, int(st[sel, 1].min()) - 12)
    x1 = min(g.shape[1], int((st[sel, 0] + st[sel, 2]).max()) + 12)
    y1 = min(g.shape[0], int((st[sel, 1] + st[sel, 3]).max()) + 12)

    clean = (keep[lbl] * 255).astype(np.uint8)[y0:y1, x0:x1]
    # already white-on-black, which is the polarity ControlNet wants
    return fit_to(Image.fromarray(clean), size, fill=0).convert("RGB")


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


def load_reference(path: str | Path, size: tuple[int, int] = SIZE) -> Image.Image:
    """Reference for IP-Adapter, fitted without squashing."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return fit_to(img.convert("RGB"), size)
