"""Pages -> panels -> filtered, de-lettered, kohya-ready training folder.

Four stages, each resumable on its own:

    extract_pages   PDF            -> data/pages/<series>/<vol>/p-*.jpg
    scan            pages          -> panels.json  (every candidate, with scores)
    emit            panels.json    -> data/datasets/<series>/<repeats>_<trigger>/

`scan` scores every candidate and records *why* each one failed, so the
thresholds can be re-swept from the JSON without re-reading a single page.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from . import clean as cleaning
from . import panels as panel_split
from . import text as lettering

RULES = ("size", "aspect", "ink", "logo", "text")


# --------------------------------------------------------------------- stage 0

def _pages_from_pdf(pdf: Path, out_dir: Path, dpi=300):
    """Write one image per page, preferring the original embedded bitmap.

    Every page in these scans is a single embedded JPEG, so pulling that bitmap
    out gives back the exact bytes the scanner produced. Rendering the page
    would resample it for nothing. Pages that are not one plain image (a cover
    with overlaid vector text, say) fall back to a render.

    PyMuPDF rather than poppler's `pdfimages`, which is not installed on Windows.
    """
    import pymupdf

    doc = pymupdf.open(pdf)
    try:
        for i, page in enumerate(doc):
            imgs = page.get_images(full=True)
            stem = out_dir / f"p-{i:04d}"
            if len(imgs) == 1:
                d = doc.extract_image(imgs[0][0])
                ext = (d or {}).get("ext")
                if d and d.get("image") and ext in ("jpg", "jpeg", "png"):
                    stem.with_suffix("." + ext).write_bytes(d["image"])
                    continue
            page.get_pixmap(dpi=dpi).save(str(stem.with_suffix(".png")))
    finally:
        doc.close()


PAGE_GLOB = "p-[0-9]*.*"


def extract_pages(cfg, root="."):
    """One image per page, straight out of the PDF with no re-rendering."""
    out_root = Path(root) / cfg.paths.pages
    made = {}
    for vol in cfg.series.volumes:
        d = out_root / vol["id"]
        d.mkdir(parents=True, exist_ok=True)
        if not list(d.glob(PAGE_GLOB)):
            pdf = Path(root) / vol["pdf"]
            if not pdf.exists():
                raise FileNotFoundError(
                    f"{pdf} not found — `pdf:` in the series config is relative "
                    f"to the repo root ({Path(root).resolve()})")
            _pages_from_pdf(pdf, d)
        made[vol["id"]] = sorted(str(p) for p in d.glob(PAGE_GLOB))
    return made


# --------------------------------------------------------------------- stage 1

def judge(crop, page_h, f):
    """First rule the panel breaks, plus every score worth keeping."""
    h, w = crop.shape
    if min(h, w) < f["min_side"]:
        return "size", {}
    ar = w / float(h)
    if not (f["aspect"][0] <= ar <= f["aspect"][1]):
        return "aspect", {}
    ink = float((crop < 200).mean())
    if not (f["ink"][0] <= ink <= f["ink"][1]):
        return "ink", {"ink": ink}
    big, _ = lettering.big_lettering(crop, page_h)
    tb, _ = lettering.burden(crop, page_h)
    d = {"ink": ink, "text": tb, "big": big}
    if big > f["max_big"]:
        return "logo", d
    if tb > f["max_text"]:
        return "text", d
    return None, d


def _scan_page(args):
    path, vol, idx, pcfg, fcfg = args
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return []
    rows = []
    for j, (x0, y0, x1, y1) in enumerate(panel_split.detect(g, **pcfg)):
        crop = g[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        why, extra = judge(crop, g.shape[0], fcfg)
        rows.append(dict(vol=vol, page=idx, panel=j,
                         box=[int(x0), int(y0), int(x1), int(y1)],
                         w=int(x1 - x0), h=int(y1 - y0), reject=why, **extra))
    return rows


def scan(cfg, pages, jobs=None, score_floor=300):
    """Score every panel candidate on every content page.

    `score_floor` is deliberately below the real min_side so that small panels
    still get a text score: the thresholds can then be swept from the JSON
    without another pass over the pages.
    """
    d = cfg.dataset
    pcfg = dict(d.panels)
    fcfg = dict(d.filters)
    fcfg["min_side"] = min(fcfg["min_side"], score_floor)

    work = []
    for vol, fs in pages.items():
        keep = fs[d.pages["head_drop"]: len(fs) - d.pages["tail_drop"]]
        work += [(f, vol, i + d.pages["head_drop"], pcfg, fcfg)
                 for i, f in enumerate(keep)]

    from multiprocessing import Pool
    rows = []
    with Pool(jobs or os.cpu_count()) as p:
        for r in p.imap_unordered(_scan_page, work, chunksize=4):
            rows += r
    return rows


def survivors(rows, f):
    """Apply the real thresholds to an already-scored table."""
    return [r for r in rows
            if "text" in r
            and min(r["w"], r["h"]) >= f["min_side"]
            and r["text"] <= f["max_text"]
            and r["big"] <= f["max_big"]]


# --------------------------------------------------------------------- stage 2

def emit(cfg, rows, pages, root="."):
    """Write the de-lettered crops and placeholder captions kohya reads.

    Captions are a trigger word plus the fixed style tags; scripts/tag_dataset.sh
    replaces them with real WD14 tags, which needs a GPU and a model download.
    """
    e, c = cfg.dataset.emit, cfg.dataset.clean
    out = Path(root) / cfg.paths.dataset
    # kohya reads every subfolder of train_data_dir as a "<repeats>_<name>"
    # concept, so the held-out panels cannot live inside it.
    dirs = {"train": out / f"{e['repeats']}_{cfg.series.trigger}",
            "holdout": out.parent / f"{out.name}_holdout"}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    rows = sorted(rows, key=lambda r: (r["vol"], r["page"], r["panel"]))
    shuffled = list(rows)
    random.Random(0).shuffle(shuffled)
    hold = {id(r) for r in shuffled[:e["holdout"]]}

    caption = ", ".join([cfg.series.trigger] + list(e["base_tags"]))
    cache, n, dropped = {}, 0, 0
    total = len(rows)
    for k, r in enumerate(rows):
        if k % 50 == 0:
            print(f"  emit {k}/{total}", flush=True)
        path = pages[r["vol"]][r["page"]]
        if path not in cache:
            cache = {path: cv2.imread(path, cv2.IMREAD_GRAYSCALE)}
        g = cache[path]
        x0, y0, x1, y1 = r["box"]
        img, ok = cleaning.clean(g[y0:y1, x0:x1], g.shape[0],
                                 min_ground=c["min_ground"],
                                 pad_frac=c["pad_frac"],
                                 ring_light=c["ring_light"])
        if not ok:
            dropped += 1          # lettering sits on artwork; erasing would gouge it
            continue
        img = cleaning.fit(img, e["min_side"], e["max_side"])
        stem = f"{cfg.series.name}_{r['vol']}_p{r['page']:03d}_c{r['panel']:02d}"
        d = dirs["holdout" if id(r) in hold else "train"]
        cv2.imwrite(str(d / f"{stem}.png"), img)
        (d / f"{stem}.txt").write_text(caption + "\n")
        n += 1
    print(f"  emit {total}/{total}", flush=True)
    return {"written": n, "text_on_art_dropped": dropped,
            "train": len(list(dirs["train"].glob("*.png"))),
            "holdout": len(list(dirs["holdout"].glob("*.png")))}
