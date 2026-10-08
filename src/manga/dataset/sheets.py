"""Contact sheets, so the filtering is eyeballed rather than trusted."""
import random
from pathlib import Path

import cv2
import numpy as np

from . import clean as cleaning


def _crop(r, pages):
    g = cv2.imread(pages[r["vol"]][r["page"]], cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = r["box"]
    return g, g[y0:y1, x0:x1]


def sheet(rows, pages, path, title, label, cols=6, cell=320):
    n = len(rows)
    nrow = (n + cols - 1) // cols
    pad, bar = 10, 34
    canvas = np.full((nrow * (cell + bar + pad) + pad + 46,
                      cols * (cell + pad) + pad, 3), 245, np.uint8)
    cv2.putText(canvas, title, (pad, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2)
    for i, r in enumerate(rows):
        _, c = _crop(r, pages)
        if c.size == 0:
            continue
        s = cell / max(c.shape)
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
        tile = np.full((cell, cell), 255, np.uint8)
        yo, xo = (cell - c.shape[0]) // 2, (cell - c.shape[1]) // 2
        tile[yo:yo + c.shape[0], xo:xo + c.shape[1]] = c
        gy, gx = divmod(i, cols)
        y, x = 46 + pad + gy * (cell + bar + pad), pad + gx * (cell + pad)
        canvas[y:y + cell, x:x + cell] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(canvas, (x, y), (x + cell, y + cell), (180, 180, 180), 1)
        cv2.putText(canvas, label(r), (x + 3, y + cell + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 20, 20), 1)
    cv2.imwrite(str(path), canvas)


def before_after(rows, pages, path, page_h_of, cfg, cell=380, cols=4):
    """The one sheet that matters for stage 3: what the eraser actually did."""
    nrow = (len(rows) + cols - 1) // cols
    c = cfg.dataset.clean
    canvas = np.full((nrow * (cell + 38) + 50, cols * (2 * cell + 36) + 12, 3), 245, np.uint8)
    cv2.putText(canvas, "BEFORE / AFTER lettering removal", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2)
    for i, r in enumerate(rows):
        g, crop = _crop(r, pages)
        done, _ = cleaning.clean(crop, g.shape[0], min_ground=c["min_ground"],
                                 pad_frac=c["pad_frac"], ring_light=c["ring_light"])
        gy, gx = divmod(i, cols)
        Y, X = 50 + gy * (cell + 38), 12 + gx * (2 * cell + 36)
        for k, img in enumerate((crop, done)):
            s = cell / max(img.shape)
            t = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
            tile = np.full((cell, cell), 255, np.uint8)
            yo, xo = (cell - t.shape[0]) // 2, (cell - t.shape[1]) // 2
            tile[yo:yo + t.shape[0], xo:xo + t.shape[1]] = t
            x = X + k * (cell + 12)
            canvas[Y:Y + cell, x:x + cell] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(canvas, (x, Y), (x + cell, Y + cell), (180, 180, 180), 1)
        cv2.putText(canvas, f"{r['vol']} p{r['page']}", (X + 3, Y + cell + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 20, 20), 1)
    cv2.imwrite(str(path), canvas)


def write_sheets(cfg, rows, pages, out: Path, n=36, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    f = dict(cfg.dataset.filters)
    scored = [r for r in rows if "text" in r]
    kept = [r for r in scored
            if min(r["w"], r["h"]) >= f["min_side"]
            and r["text"] <= f["max_text"] and r["big"] <= f["max_big"]]
    groups = {
        "kept": (kept, lambda r: f"{r['vol']} p{r['page']} {r['w']}x{r['h']} t{r['text']:.2f}"),
        "rej_text": ([r for r in scored if r["text"] > f["max_text"]
                      and min(r["w"], r["h"]) >= f["min_side"]],
                     lambda r: f"{r['vol']} p{r['page']} t{r['text']:.2f}"),
        "rej_logo": ([r for r in scored if r["big"] > f["max_big"]
                      and min(r["w"], r["h"]) >= f["min_side"]],
                     lambda r: f"{r['vol']} p{r['page']} big{r['big']:.3f}"),
        "rej_size": ([r for r in scored if min(r["w"], r["h"]) < f["min_side"]],
                     lambda r: f"{r['vol']} p{r['page']} {r['w']}x{r['h']}"),
    }
    for name, (rs, lab) in groups.items():
        if not rs:
            continue
        sheet(rng.sample(rs, min(n, len(rs))), pages, out / f"{name}.png",
              f"{name}  ({len(rs)} panels)", lab)
    if kept:
        before_after(rng.sample(kept, min(12, len(kept))), pages,
                     out / "cleaned.png", None, cfg)
