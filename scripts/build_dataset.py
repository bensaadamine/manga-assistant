#!/usr/bin/env python
"""Build a LoRA dataset from a series' PDFs.

    python scripts/build_dataset.py --config spyxfamily.yaml
    python scripts/build_dataset.py --config spyxfamily.yaml --sweep
    python scripts/build_dataset.py --config spyxfamily.yaml --sheets out/

--sweep re-applies thresholds to the cached scan instead of re-reading pages,
so trading count against cleanliness costs a second rather than a minute.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga import config as cfgmod            # noqa: E402
from manga import dataset as ds               # noqa: E402

SCAN_CACHE = "panels.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="spyxfamily.yaml")
    ap.add_argument("--rescan", action="store_true", help="ignore the cached scan")
    ap.add_argument("--sweep", action="store_true", help="print a threshold table and stop")
    ap.add_argument("--sheets", metavar="DIR", help="write contact sheets and stop")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the dataset already holds real captions")
    a = ap.parse_args()

    cfg = cfgmod.load(a.config)
    pages = ds.extract_pages(cfg, ROOT)
    print(f"pages: {sum(len(v) for v in pages.values())}")

    cache = ROOT / cfg.paths.pages / SCAN_CACHE
    if cache.exists() and not a.rescan:
        rows = json.loads(cache.read_text())
        print(f"scan: {len(rows)} candidates (cached; --rescan to redo)")
    else:
        rows = ds.scan(cfg, pages, jobs=a.jobs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
        print(f"scan: {len(rows)} candidates")

    f = dict(cfg.dataset.filters)

    if a.sweep:
        print(f"\n{'min_side':<9}" + "".join(f"{'text<=' + format(t, '.2f'):>11}"
                                             for t in (0.25, 0.30, 0.35)))
        for ms in (512, 480, 448, 416, 384):
            line = f"{ms:<9}"
            for mt in (0.25, 0.30, 0.35):
                g = dict(f, min_side=ms, max_text=mt)
                line += f"{len(ds.survivors(rows, g)):>11}"
            print(line)
        return 0

    kept = ds.survivors(rows, f)
    counts = {}
    for r in rows:
        counts[r["reject"]] = counts.get(r["reject"], 0) + 1
    print("\nrejects:")
    for rule in ds.RULES:
        if counts.get(rule):
            print(f"  {rule:<8} {counts[rule]:>5}")
    print(f"  {'kept':<8} {len(kept):>5}")

    if a.sheets:
        from manga.dataset.sheets import write_sheets
        write_sheets(cfg, rows, pages, Path(a.sheets))
        print(f"\ncontact sheets -> {a.sheets}")
        return 0

    # Emitting rewrites every caption as a placeholder. Once tag_dataset.py has
    # run, that would silently throw the WD14 tags away.
    e = cfg.dataset.emit
    train_dir = ROOT / cfg.paths.dataset / f"{e['repeats']}_{cfg.series.trigger}"
    placeholder = ", ".join([cfg.series.trigger] + list(e["base_tags"]))
    tagged = [c for c in train_dir.glob("*.txt")
              if c.read_text(encoding="utf-8").strip() != placeholder]
    if tagged and not a.force:
        raise SystemExit(
            f"\n{train_dir} already holds {len(tagged)} tagged captions.\n"
            f"Rebuilding would overwrite them with placeholders and you would have "
            f"to run tag_dataset.py again.\nPass --force if that is what you want.")

    stats = ds.emit(cfg, kept, pages, ROOT)
    print(f"\nemit: {stats['written']} images "
          f"({stats['train']} train / {stats['holdout']} holdout), "
          f"{stats['text_on_art_dropped']} dropped as text-on-artwork")
    print(f"  -> {cfg.paths.dataset}")
    expected = len(kept)
    if stats["written"] + stats["text_on_art_dropped"] < expected:
        print(f"\nWARNING: only {stats['written']} of {expected} panels were written. "
              f"An interrupted run leaves a partial dataset - re-run to rebuild it.")
    print(f"\nnext: python scripts/tag_dataset.py "
          f"{cfg.paths.dataset}/{cfg.dataset.emit['repeats']}_{cfg.series.trigger} "
          f"{cfg.series.trigger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
