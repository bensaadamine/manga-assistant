"""magi.json -> one training image per character.

No torch here: scripts/magi_detect.py already did the detection and cached the
boxes, so this stage is cv2 and numpy only and can be re-swept in seconds.

Three rules, in the order they fire:

  size   a crop smaller than `min_side` on its short side would be upscaled into
         blur, and blur is what the LoRA would then learn. Dropped, not rescued.
  text   a text box overlapping the crop is painted white. Lettering is the one
         defect that teaches the model to draw pseudo-Japanese everywhere.
  ink    an all-white or all-black crop is a detection error, not a character.

Framing is deliberately NOT a rule. Style lives in line weight, screentone and
how a face is drawn, all of which a shoulders-up crop carries. The full body
comes from ControlNet at generation time, never from this dataset.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import clean as cleaning

RULES = ("size", "ink", "blank")


def _pad(box, w, h, frac):
    """Grow the box by `frac` of its own size, clipped to the page.

    Magi boxes hug the character. A little air around them matters: a crop that
    cuts exactly at the shoulder line teaches the model to cut at the shoulder.
    """
    x0, y0, x1, y1 = box
    dx, dy = int((x1 - x0) * frac), int((y1 - y0) * frac)
    return (max(0, x0 - dx), max(0, y0 - dy), min(w, x1 + dx), min(h, y1 + dy))


def _overlap(a, b):
    """Intersection area of two boxes, 0 when they miss."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _whiten(crop, box, texts, min_cover=0.25):
    """Paint out every text box that meaningfully overlaps this crop.

    A bubble clipped at the very edge of the crop is left alone - erasing a sliver
    leaves a white notch that reads as a drawing element. `min_cover` is the
    fraction of the TEXT box that must fall inside the crop before we act.
    """
    x0, y0 = box[0], box[1]
    out = crop.copy()
    n = 0
    for t in texts:
        area = (t[2] - t[0]) * (t[3] - t[1])
        if area <= 0 or _overlap(box, t) / area < min_cover:
            continue
        tx0, ty0 = max(t[0] - x0, 0), max(t[1] - y0, 0)
        tx1, ty1 = min(t[2] - x0, out.shape[1]), min(t[3] - y0, out.shape[0])
        if tx1 > tx0 and ty1 > ty0:
            out[ty0:ty1, tx0:tx1] = 255
            n += 1
    return out, n


def judge(crop, c):
    """First rule the crop breaks, or None."""
    h, w = crop.shape
    if min(h, w) < c["min_side"]:
        return "size"
    ink = float((crop < 200).mean())
    if ink < c["ink"][0]:
        return "blank"
    if ink > c["ink"][1]:
        return "ink"
    return None


def collect(magi: dict, c: dict, pages: dict) -> list[dict]:
    """One row per character box, with the reject reason when it fails.

    Pages are located through `pages` ({vol: [path, ...]}) rather than the path
    stored in magi.json, because detection may have run on another machine -
    Kaggle, say - whose paths mean nothing here.

    Pixels are read because the rules need them, but nothing is written: the
    thresholds can be swept over the returned table without re-reading.
    """
    rows = []
    for key in sorted(magi):
        p = magi[key]
        g = cv2.imread(pages[p["vol"]][p["page"]], cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ph, pw = g.shape
        for j, box in enumerate(p["characters"]):
            b = _pad(box, pw, ph, c["pad_frac"])
            crop = g[b[1]:b[3], b[0]:b[2]]
            if crop.size == 0:
                continue
            crop, n_text = _whiten(crop, b, p["texts"], c["text_cover"])
            rows.append(dict(vol=p["vol"], page=p["page"], char=j, box=list(b),
                             w=b[2] - b[0], h=b[3] - b[1], texts_erased=n_text,
                             reject=judge(crop, c)))
    return rows


def survivors(rows, c):
    """Re-apply thresholds to an already-collected table."""
    return [r for r in rows
            if min(r["w"], r["h"]) >= c["min_side"] and r["reject"] != "blank"]


def emit(cfg, rows, magi, pages, root=".") -> dict:
    """Write the character crops and placeholder captions kohya reads.

    Same folder contract as pipeline.emit: "<repeats>_<trigger>" for training and
    a sibling "_holdout" that must stay OUTSIDE it, because kohya treats every
    subfolder of train_data_dir as a concept.
    """
    c, e = dict(cfg.dataset.characters), cfg.dataset.emit
    out = Path(root) / cfg.paths.dataset
    dirs = {"train": out / f"{e['repeats']}_{cfg.series.trigger}",
            "holdout": out.parent / f"{out.name}_holdout"}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    rows = sorted(rows, key=lambda r: (r["vol"], r["page"], r["char"]))
    shuffled = list(rows)
    random.Random(0).shuffle(shuffled)
    hold = {id(r) for r in shuffled[:e["holdout"]]}

    caption = ", ".join([cfg.series.trigger] + list(e["base_tags"]))
    by_page = {f"{p['vol']}/{p['page']}": p for p in magi.values()}
    cache, n, total = {}, 0, len(rows)

    for k, r in enumerate(rows):
        if k % 100 == 0:
            print(f"  emit {k}/{total}", flush=True)
        page = by_page[f"{r['vol']}/{r['page']}"]
        path = pages[r["vol"]][r["page"]]
        if path not in cache:
            cache = {path: cv2.imread(path, cv2.IMREAD_GRAYSCALE)}
        g = cache[path]
        b = r["box"]
        crop, _ = _whiten(g[b[1]:b[3], b[0]:b[2]], b, page["texts"], c["text_cover"])
        img = cleaning.fit(crop, e["min_side"], e["max_side"])
        stem = f"{cfg.series.name}_{r['vol']}_p{r['page']:03d}_k{r['char']:02d}"
        d = dirs["holdout" if id(r) in hold else "train"]
        cv2.imwrite(str(d / f"{stem}.png"), img)
        (d / f"{stem}.txt").write_text(caption + "\n")
        n += 1
    print(f"  emit {total}/{total}", flush=True)
    return {"written": n,
            "train": len(list(dirs["train"].glob("*.png"))),
            "holdout": len(list(dirs["holdout"].glob("*.png")))}
