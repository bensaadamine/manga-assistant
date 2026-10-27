"""magi.json -> one training image per character. CPU only.

Ported unchanged from scripts/crops_from_magi.py, which built dataset v2. Checked
26 Oct on magi_v3.json: 708 crops, a superset of the 647 that were trained on
(the tag filter removes the other 61), 40/40 sampled pixel-identical.
Rules, in the order they fire:

  size     character box short side < min_px            -> blur, dropped
  aspect   box aspect outside `aspect`                   -> eye-slits, strips
  overlap  another character covers > `overlap` of it    -> two-character crops
  text     Magi text boxes cover > max_text of the crop  -> bubble-dominated
  then     margin added (no square pad, no centre crop: kohya buckets handle
           aspect, and centre-cropping cut heads and feet), lettering erased,
           long side downscaled to max_side.

Lettering: erase only ink blobs that lie ENTIRELY inside a text box, found on
the whole page. Letters fit inside and go; hair or a coat crossing the box sticks
out and survives. Filling the box with white punched holes through ~258 faces.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .magi import key

RULES = ("small", "aspect", "overlap", "text")


def cover(a, b):
    """Share of box a covered by box b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0) / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def erase_text(a: np.ndarray, texts, c) -> np.ndarray:
    if not texts:
        return a
    H, _ = a.shape
    ink = (a < c["ink"]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    bx0, by0 = st[:, cv2.CC_STAT_LEFT], st[:, cv2.CC_STAT_TOP]
    bx1, by1 = bx0 + st[:, cv2.CC_STAT_WIDTH], by0 + st[:, cv2.CC_STAT_HEIGHT]
    erase = np.zeros(n, bool)
    p = c["pad_px"]
    for t in texts:
        erase |= (bx0 >= t[0] - p) & (by0 >= t[1] - p) & (bx1 <= t[2] + p) & (by1 <= t[3] + p)
    erase[0] = False                                        # label 0 = paper
    erase &= st[:, cv2.CC_STAT_HEIGHT] < c["max_glyph"] * H  # never a huge blob
    mask = cv2.dilate(erase[lab].astype(np.uint8), np.ones((3, 3), np.uint8))
    out = a.copy()
    out[mask > 0] = 255
    return erase_bubbles(out, texts, c) if c.get("erase_bubbles") else out


def erase_bubbles(a: np.ndarray, texts, c) -> np.ndarray:
    """Whiten the speech bubble each text box sits in, outline included.

    Letters already gone, the bubble interior is the white region under the text
    box. It counts as a bubble only if it stays close to the box (`bubble_grow`
    of the box size on each side): text set on open paper or a background whose
    outline is broken spreads far beyond it and is left alone. The region is then
    grown by `bubble_ring` of the page height so the outline ring goes with it.
    Empty bubbles left in v2 came back in B4 as ghost circles.
    """
    H, W = a.shape
    white = (a >= c["ink"]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    g, keep = c["bubble_grow"], np.zeros(n, bool)
    for t in texts:
        x0, y0, x1, y1 = (int(v) for v in t)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        ids, cnt = np.unique(lab[y0:y1, x0:x1], return_counts=True)
        cnt[ids == 0] = 0                               # 0 = ink
        if cnt.max() == 0:
            continue
        k = ids[cnt.argmax()]
        bw, bh = x1 - x0, y1 - y0
        L, T = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP]
        R, B = L + st[k, cv2.CC_STAT_WIDTH], T + st[k, cv2.CC_STAT_HEIGHT]
        if (L >= x0 - g * bw and R <= x1 + g * bw and T >= y0 - g * bh and B <= y1 + g * bh
                and L > 0 and T > 0 and R < W and B < H):
            keep[k] = True
    if not keep.any():
        return a
    r = max(3, int(c["bubble_ring"] * H)) | 1
    mask = cv2.dilate(keep[lab].astype(np.uint8), np.ones((r, r), np.uint8))
    out = a.copy()
    out[mask > 0] = 255
    return out


def build(cfg, magi: dict, pages: dict, out: Path) -> dict:
    """Write crops (placeholder captions) into `out`. Returns kept + reject counts."""
    c = dict(cfg.dataset.crops)
    out.mkdir(parents=True, exist_ok=True)
    caption = ", ".join([cfg.series.trigger] + list(cfg.dataset.emit["base_tags"]))
    why = dict.fromkeys(RULES, 0)
    kept = missing = 0
    jobs = [(v, p) for v, ps in pages.items() for p in ps]
    for n, (vol, path) in enumerate(jobs, 1):
        rec = magi.get(key(vol, path))
        if rec is None:
            missing += 1
            continue
        page = np.array(Image.open(path).convert("L"))
        H, W = page.shape
        chars, texts, clean = rec["characters"], rec["texts"], None
        for i, box in enumerate(chars):
            x0, y0, x1, y1 = box
            w, h = x1 - x0, y1 - y0
            if min(w, h) < c["min_px"]:
                why["small"] += 1; continue
            if not c["aspect"][0] <= w / max(h, 1) <= c["aspect"][1]:
                why["aspect"] += 1; continue
            if any(cover(box, o) > c["overlap"] for j, o in enumerate(chars) if j != i):
                why["overlap"] += 1; continue
            mx, my = w * c["margin"], h * c["margin"]
            b = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))
            if sum(cover(b, t) for t in texts) > c["max_text"]:
                why["text"] += 1; continue
            if clean is None:
                clean = erase_text(page, texts, c)
            im = Image.fromarray(clean[int(b[1]):int(b[3]), int(b[0]):int(b[2])])
            s = c["max_side"] / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            stem = f"{cfg.series.name}_{vol}_{Path(path).stem}_k{i:02d}"
            im.save(out / f"{stem}.png")
            (out / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
            kept += 1
        if n % 200 == 0:
            print(f"  crops {n}/{len(jobs)} pages, {kept} kept", flush=True)
    if missing:
        print(f"  WARNING: {missing} pages have no Magi record — rerun the magi stage")
    return {"kept": kept, "rejected": why, "pages_without_magi": missing}
