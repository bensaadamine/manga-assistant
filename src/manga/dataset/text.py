"""Lettering detection on a manga panel.

Bubbles turn out to be the wrong thing to look for: in these scans a large part
of the dialogue is set straight onto white with no outline around it at all, and
where there *is* an outline it often opens into the page background, so the
bubble interior is not a separate white region. Letters, on the other hand, are
always letters: small ink blobs of near-uniform height that cluster into lines.
So find the type first, then grow it into the block it occupies.
"""
import numpy as np
import cv2


def char_mask(gray, page_h, ink_thr=200, hmin=0.006, hmax=0.032):
    """Mask of ink components whose size and shape are letter-like."""
    ink = (gray < ink_thr).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    lo, hi = hmin * page_h, hmax * page_h
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if not (lo <= bh <= hi):
            continue
        if bw > 3.5 * bh or bw < 0.06 * bh:
            continue
        if a < 0.08 * bw * bh:                  # hollow: outline, not a glyph
            continue
        keep[i] = True
    return keep[lbl].astype(np.uint8), st, lbl, keep


def text_blocks(gray, page_h, min_chars=5):
    """Bounding boxes of lettering blocks (bubble contents or bare captions)."""
    h, w = gray.shape
    cm, st, lbl, keep = char_mask(gray, page_h)
    if cm.sum() == 0:
        return []
    k = max(3, int(0.018 * page_h))
    glue = cv2.dilate(cm, np.ones((k, k), np.uint8), iterations=1)
    n, blbl, bst, _ = cv2.connectedComponentsWithStats(glue, 8)

    # count letter components per block
    idx = np.flatnonzero(keep)
    counts = {}
    for i in idx:
        ys, xs = int(st[i, 1] + st[i, 3] // 2), int(st[i, 0] + st[i, 2] // 2)
        b = blbl[min(ys, h - 1), min(xs, w - 1)]
        if b:
            counts[b] = counts.get(b, 0) + 1

    out = []
    for b in range(1, n):
        if counts.get(b, 0) < min_chars:
            continue
        x, y, bw, bh, a = bst[b]
        if bw * bh > 0.75 * h * w:              # a whole-panel blob is not type
            continue
        pad = k // 2
        out.append((max(0, x - pad), max(0, y - pad),
                    min(w, x + bw + pad), min(h, y + bh + pad)))
    return out


def big_lettering(gray, page_h, ink_thr=200, hmin=0.035, hmax=0.16):
    """Oversized type: chapter logos, title cards, and large drawn SFX.

    These are the marks a style LoRA must never see, because they are the ones
    it would learn to scrawl across every output. They are too big to pass the
    ordinary letter test, so they get their own pass.
    """
    h, w = gray.shape
    ink = (gray < ink_thr).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    lo, hi = hmin * page_h, hmax * page_h
    picks = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if not (lo <= bh <= hi):
            continue
        if bw > 2.5 * bh or bw < 0.15 * bh:
            continue
        d = a / float(bw * bh)
        if not (0.12 <= d <= 0.78):            # hollow outline or solid blob
            continue
        picks.append((x, y, bw, bh))
    if len(picks) < 3:
        return 0.0, []
    hs = np.array([p[3] for p in picks], float)
    if hs.std() / hs.mean() > 0.55:            # a run of type is one size
        return 0.0, []
    cov = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in picks:
        cov[y:y + bh, x:x + bw] = 1
    return float(cov.mean()), picks


def blocks_both_polarities(gray, page_h, min_chars=5):
    """Lettering blocks as (box, dark_ground) pairs.

    Narration boxes are often knocked out white on black. Inverted, they are
    ordinary type, so the same detector finds them on `255 - gray`; the flag
    records which way round the block was, because erasing it means filling
    with its own ground, not always with white.
    """
    out = [(b, False) for b in text_blocks(gray, page_h, min_chars)]
    inv = 255 - gray
    for b in text_blocks(inv, page_h, min_chars):
        x0, y0, x1, y1 = b
        if gray[y0:y1, x0:x1].mean() < 128:      # genuinely a dark-ground block
            out.append((b, True))
    return out


def burden(gray, page_h):
    """Fraction of the panel covered by lettering blocks, either polarity."""
    bl = blocks_both_polarities(gray, page_h)
    cov = np.zeros(gray.shape, np.uint8)
    for (x0, y0, x1, y1), _ in bl:
        cov[y0:y1, x0:x1] = 1
    return float(cov.mean()), [b for b, _ in bl]
