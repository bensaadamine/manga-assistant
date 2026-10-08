"""Stage 4: erase lettering from a kept panel, or refuse the panel.

Whitening is only safe where the type sits on light ground — inside a bubble,
or set on open white. Where it is laid straight over artwork, painting it out
would gouge a white hole in the drawing, so the panel is dropped instead.
"""
import numpy as np
import cv2
from . import text

def clean(gray, page_h, min_ground=0.45, pad_frac=0.012, ring_light=0.80):
    """Return (cleaned, ok). ok=False means the lettering sits on artwork.

    Rather than painting out the block's bounding box, this paints out the pale
    ground the type is sitting on — the inside of its bubble, or the open white
    around it. A bubble drawn over a drawing is therefore emptied without
    touching the drawing around it, and only type laid directly on inked art
    has no pale ground to erase, which is what disqualifies the panel.
    """
    h, w = gray.shape
    out = gray.copy()
    pad = max(3, int(pad_frac * page_h))
    blocks = text.blocks_both_polarities(gray, page_h)
    if not blocks:
        return out, True

    masks, grounds = {}, {}
    for dark in (False, True):
        src = (255 - gray) if dark else gray
        masks[dark] = text.char_mask(src, page_h)[0].astype(bool)
        grounds[dark] = cv2.connectedComponentsWithStats(
            (src >= 200).astype(np.uint8), 8)[1]

    for (x0, y0, x1, y1), dark in blocks:
        src = (255 - gray) if dark else gray
        fill = 0 if dark else 255
        cm, ll = masks[dark], grounds[dark]
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(w, x1 + pad), min(h, y1 + pad)
        glyph = (src[ry0:ry1, rx0:rx1] < 200).astype(np.uint8)
        near = cv2.dilate(glyph, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        labs = np.unique(ll[ry0:ry1, rx0:rx1][(near > 0) & (glyph == 0)])
        labs = labs[labs > 0]
        if labs.size == 0:
            return gray, False

        ground = np.isin(ll[ry0:ry1, rx0:rx1], labs)
        box = np.zeros_like(ground)
        box[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = True
        if (ground & box).sum() < min_ground * box.sum():
            return gray, False                       # no pale ground: it is on art

        # Where the collar around the block is pale too — inside a bubble, or
        # set on open white — the whole block can go, which is the only way to
        # take every stroke rather than most of them.
        ring = ~box
        if ring.sum() and (src[ry0:ry1, rx0:rx1][ring] >= 200).mean() >= ring_light:
            out[ry0:ry1, rx0:rx1][box] = fill
            continue

        # Otherwise erase the strokes themselves, not the block they sit in. A
        # block mis-detected on a face then costs a few white specks instead of
        # punching a hole through the drawing.
        strokes = cm[ry0:ry1, rx0:rx1] & box
        grow = cv2.dilate(strokes.astype(np.uint8),
                          np.ones((5, 5), np.uint8), iterations=1) > 0
        near_ground = cv2.dilate(ground.astype(np.uint8),
                                 np.ones((7, 7), np.uint8), iterations=1) > 0
        out[ry0:ry1, rx0:rx1][grow & near_ground] = fill
    return out, True


def fit(gray, min_side=512, max_side=1024):
    """Scale to a short side of `min_side`, then cap the long side by centre crop.

    No letterboxing: white bars would be a feature of the data and the model
    would learn to draw them. kohya's bucketing handles the mixed aspects.
    """
    h, w = gray.shape
    s = min_side / float(min(h, w))
    nh, nw = int(round(h * s)), int(round(w * s))
    g = cv2.resize(gray, (nw, nh),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if g.shape[0] > max_side:
        o = (g.shape[0] - max_side) // 2
        g = g[o:o + max_side]
    if g.shape[1] > max_side:
        o = (g.shape[1] - max_side) // 2
        g = g[:, o:o + max_side]
    return g
