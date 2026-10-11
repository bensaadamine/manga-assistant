#!/usr/bin/env python
"""Rebuild character crops from a saved Magi JSON. CPU only, no GPU, no Kaggle.

    python scripts/crops_from_magi.py
    python scripts/crops_from_magi.py --out data/datasets/spyxfamily_v2/5_sxfstyle

Same 6 rules as the Kaggle cell, plus two fixes:
  - text erase removes only ink blobs that fit entirely inside a Magi text box
    (letters), never blobs that cross it (hair, black clothes) -> no more
    white rectangles on faces
  - a crop is dropped if Magi text boxes cover > MAX_TEXT of it -> drops most
    crops dominated by speech bubbles
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MIN_PX = 256          # rule 4: shortest side of the character box
MARGIN = 0.05         # rule 1
OVERLAP = 0.15        # rule 3: max share of the box covered by another character
ASPECT = (0.5, 2.0)   # rule 5
MAX_SIDE = 1024       # rule 6: = max_bucket_reso, downscale only
MAX_TEXT = 0.10       # new: max share of the crop covered by text boxes
INK = 200             # darker than this = ink
PAD_PX = 4            # text boxes are tight: allow a few px of slack
MAX_GLYPH = 0.08      # a letter is never taller than 8% of the page


def cover(a, b):
    """Share of box a covered by box b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def erase_text(a: np.ndarray, texts) -> np.ndarray:
    """Whiten every ink blob that lies ENTIRELY inside a Magi text box.

    Blobs are found on the whole page, so hair or a black coat that crosses a
    text box sticks out of it and survives. Letters fit inside and are erased.
    """
    if not texts:
        return a
    H, W = a.shape
    ink = (a < INK).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    bx0 = st[:, cv2.CC_STAT_LEFT]; by0 = st[:, cv2.CC_STAT_TOP]
    bx1 = bx0 + st[:, cv2.CC_STAT_WIDTH]; by1 = by0 + st[:, cv2.CC_STAT_HEIGHT]
    erase = np.zeros(n, bool)
    for t in texts:
        x0, y0, x1, y1 = t[0] - PAD_PX, t[1] - PAD_PX, t[2] + PAD_PX, t[3] + PAD_PX
        erase |= (bx0 >= x0) & (by0 >= y0) & (bx1 <= x1) & (by1 <= y1)
    erase[0] = False                                   # label 0 = paper
    erase &= st[:, cv2.CC_STAT_HEIGHT] < MAX_GLYPH * H  # safety: never a huge blob
    mask = cv2.dilate(erase[lab].astype(np.uint8), np.ones((3, 3), np.uint8))
    out = a.copy()
    out[mask > 0] = 255
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/datasets/magi_v3.json")
    ap.add_argument("--pages", default="data/pages/spyxfamily")
    ap.add_argument("--out", default="data/datasets/spyxfamily_v2/5_sxfstyle")
    a = ap.parse_args()

    magi = json.loads((ROOT / a.json).read_text())
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    kept = 0
    why = {"small": 0, "aspect": 0, "overlap": 0, "text": 0}
    for key, rec in sorted(magi.items()):
        vol, fname = key.split("/")
        page = np.array(Image.open(ROOT / a.pages / vol / fname).convert("L"))
        H, W = page.shape
        chars, texts = rec["characters"], rec["texts"]
        clean = None

        for i, box in enumerate(chars):
            x0, y0, x1, y1 = box
            w, h = x1 - x0, y1 - y0
            if min(w, h) < MIN_PX:
                why["small"] += 1; continue
            if not ASPECT[0] <= w / max(h, 1) <= ASPECT[1]:
                why["aspect"] += 1; continue
            if any(cover(box, o) > OVERLAP for j, o in enumerate(chars) if j != i):
                why["overlap"] += 1; continue

            mx, my = w * MARGIN, h * MARGIN
            c = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))
            if sum(cover(c, t) for t in texts) > MAX_TEXT:
                why["text"] += 1; continue

            if clean is None:
                clean = erase_text(page, texts)
            crop = Image.fromarray(clean[int(c[1]):int(c[3]), int(c[0]):int(c[2])])
            s = MAX_SIDE / max(crop.size)
            if s < 1:
                crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.LANCZOS)
            crop.save(out / f"sxf_{vol}_{Path(fname).stem}_ch{i:02d}.png")
            kept += 1

    print(f"{kept} crops -> {out}")
    print("dropped:", why)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
