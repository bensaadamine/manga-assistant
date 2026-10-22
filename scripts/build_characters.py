#!/usr/bin/env python
"""Build a LoRA dataset of single-character crops from the cached Magi boxes.

    python scripts/magi_detect.py     --config spyxfamily.yaml   # once, needs a GPU
    python scripts/build_characters.py --config spyxfamily.yaml --sweep
    python scripts/build_characters.py --config spyxfamily.yaml

--sweep prints how many crops survive at each min_side and stops. Run it first:
it is the number that decides whether you need another volume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga import config as cfgmod                    # noqa: E402
from manga.dataset import characters as ch            # noqa: E402

CACHE = "magi.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="spyxfamily.yaml")
    ap.add_argument("--sweep", action="store_true", help="print a threshold table and stop")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the dataset already holds real captions")
    a = ap.parse_args()

    cfg = cfgmod.load(a.config)
    magi_path = ROOT / cfg.paths.pages / cfg.series.name / CACHE
    if not magi_path.exists():
        raise SystemExit(f"{magi_path} not found — run scripts/magi_detect.py first")
    magi = json.loads(magi_path.read_text())

    c = dict(cfg.dataset.characters)
    print(f"pages: {len(magi)}   character boxes: "
          f"{sum(len(p['characters']) for p in magi.values())}")

    print("reading pages and scoring crops...")
    rows = ch.collect(magi, dict(c, min_side=0))   # score everything, filter after

    if a.sweep:
        print(f"\n{'min_side':<10}{'crops kept':>12}")
        for ms in (640, 576, 512, 448, 384, 320, 256):
            print(f"{ms:<10}{len(ch.survivors(rows, dict(c, min_side=ms))):>12}")
        print("\n300+ is enough to train. Below that, add a volume rather than "
              "lowering min_side — upscaled blur teaches blur.")
        return 0

    kept = ch.survivors(rows, c)
    counts = {}
    for r in rows:
        counts[r["reject"]] = counts.get(r["reject"], 0) + 1
    print("\nrejects:")
    for rule in ch.RULES:
        if counts.get(rule):
            print(f"  {rule:<8} {counts[rule]:>5}")
    print(f"  {'kept':<8} {len(kept):>5}")
    erased = sum(r["texts_erased"] for r in kept)
    print(f"\n{erased} text boxes whitened across {len(kept)} crops")

    if len(kept) < 200:
        print(f"\nWARNING: {len(kept)} crops is below what a style LoRA needs (~300). "
              f"Add a volume before training.")

    e = cfg.dataset.emit
    train_dir = ROOT / cfg.paths.dataset / f"{e['repeats']}_{cfg.series.trigger}"
    placeholder = ", ".join([cfg.series.trigger] + list(e["base_tags"]))
    tagged = [p for p in train_dir.glob("*.txt")
              if p.read_text(encoding="utf-8").strip() != placeholder]
    if tagged and not a.force:
        raise SystemExit(
            f"\n{train_dir} already holds {len(tagged)} tagged captions.\n"
            f"Rebuilding would overwrite them with placeholders. Pass --force if "
            f"that is what you want.")

    stats = ch.emit(cfg, kept, magi, ROOT)
    print(f"\nemit: {stats['written']} images "
          f"({stats['train']} train / {stats['holdout']} holdout)")
    print(f"  -> {cfg.paths.dataset}")
    print(f"\nnext: python scripts/tag_dataset.py "
          f"{cfg.paths.dataset}/{e['repeats']}_{cfg.series.trigger} {cfg.series.trigger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
