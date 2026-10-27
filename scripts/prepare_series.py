#!/usr/bin/env python
"""Volumes in, kohya-ready dataset out. One command per series.

    python scripts/prepare_series.py --config bungo.yaml                  # everything
    python scripts/prepare_series.py --config bungo.yaml --limit 40       # smoke test
    python scripts/prepare_series.py --config bungo.yaml --from crops     # redo from a stage
    python scripts/prepare_series.py --config bungo.yaml --src /kaggle/input/eng

Stages, each skipped when its output already exists:

    pages    PDF or image folders -> data/pages/<series>/<vol>/p-*.jpg
    panels   panel cut, text filter, lettering erase -> <dataset>/<R>_<trigger>/
             and <dataset>_holdout/ (FID only, never trained on)
    magi     character + text boxes -> data/pages/<series>/magi.json   (GPU)
    crops    one image per character -> <dataset>/<R>_<trigger>_crops/
    tag      WD14 captions, trigger first, on both folders
    filter   drop images tagged `no humans` / `close-up` / `eye focus`

Writes <dataset>_report.json: counts per stage, git commit and the resolved
dataset config, so every dataset version says how it was made.

Magi needs a GPU. On the laptop (Smart App Control blocks torch) the magi stage
is skipped when magi.json is already present, and stops with a hint otherwise.
Run the whole thing on Kaggle with notebooks/02_prepare_series_kaggle.ipynb.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga import config as cfgmod                  # noqa: E402
from manga import dataset as ds                     # noqa: E402
from manga.dataset import crops as crops_mod        # noqa: E402
from manga.dataset import magi as magi_mod          # noqa: E402

STAGES = ["pages", "panels", "magi", "crops", "tag", "filter"]
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _pngs(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in EXTS) if d.is_dir() else []


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _placeholder(cfg) -> str:
    return ", ".join([cfg.series.trigger] + list(cfg.dataset.emit["base_tags"]))


def stage_tag(cfg, dirs, report):
    """Caption every image still carrying the placeholder. Resumable."""
    from manga.dataset import tagger
    t, ph = cfg.dataset.tag, _placeholder(cfg)
    todo = [p for d in dirs for p in _pngs(d)
            if not p.with_suffix(".txt").exists()
            or p.with_suffix(".txt").read_text(encoding="utf-8").strip() == ph]
    report["tag"] = {"captioned": len(todo)}
    if not todo:
        print("tag: all captioned"); return
    print(f"tag: {len(todo)} images (WD14 download ~400 MB on first run)")
    model, tags = tagger.download("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    for path, found in tagger.tag_images(
            todo, model, tags, general_threshold=t["threshold"],
            character_threshold=t["character_threshold"],
            progress=lambda d, n: print(f"  tag {d}/{n}", flush=True) if d % 100 == 0 or d == n else None):
        found = [x for x in found if x != cfg.series.trigger]
        path.with_suffix(".txt").write_text(", ".join([cfg.series.trigger] + found) + "\n",
                                            encoding="utf-8")


def stage_filter(cfg, dirs, report):
    """Move images whose caption carries a drop tag to <dataset>_dropped/. Idempotent."""
    drop = set(cfg.dataset.tag["drop_tags"])
    out = {}
    for d in dirs:
        n = 0
        for p in _pngs(d):
            cap = p.with_suffix(".txt")
            tags = {x.strip() for x in cap.read_text(encoding="utf-8").split(",")}
            if tags & drop:
                bin_ = d.parent.parent / f"{d.parent.name}_dropped" / d.name   # outside train_data_dir
                bin_.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), bin_ / p.name)
                shutil.move(str(cap), bin_ / cap.name)
                n += 1
        out[d.name] = {"dropped": n, "kept": len(_pngs(d))}
        print(f"filter: {d.name}  -{n}  -> {out[d.name]['kept']}")
    report["filter"] = out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", help="folder holding the volumes (overrides config paths)")
    ap.add_argument("--from", dest="start", choices=STAGES,
                    help="redo this stage and every later one")
    ap.add_argument("--limit", type=int, help="first N pages per volume, into *_smoke paths")
    ap.add_argument("--until", choices=STAGES, help="stop after this stage")
    ap.add_argument("--skip-magi", action="store_true", help="panels only, no crops")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()

    cfg = cfgmod.load(a.config)
    if a.limit:                                   # never pollute the real dataset
        cfg["paths"]["pages"] += "_smoke"
        cfg["paths"]["dataset"] += "_smoke"
    force = set(STAGES[STAGES.index(a.start):]) if a.start else set()

    pages_dir = ROOT / cfg.paths.pages
    out = ROOT / cfg.paths.dataset
    e, c = cfg.dataset.emit, cfg.dataset.crops
    trig = cfg.series.trigger
    panels_dir = out / f"{e['repeats']}_{trig}"
    crops_dir = out / f"{c['repeats']}_{trig}_crops"
    if e["repeats"] == c["repeats"]:
        print("note: panels and crops share a repeat count; folders still differ by suffix")
    report = {"series": cfg.series.name, "config": a.config, "git": _git(),
              "started": datetime.now().isoformat(timespec="seconds"),
              "limit": a.limit, "timings_s": {}}
    tick = time.time

    # ---------------------------------------------------------------- pages
    t0 = tick()
    if "pages" in force:
        for v in cfg.series.volumes:
            shutil.rmtree(pages_dir / v["id"], ignore_errors=True)
    pages = ds.extract_pages(cfg, ROOT, src_root=a.src, limit=a.limit)
    n_pages = sum(len(v) for v in pages.values())
    report["pages"] = {v: len(p) for v, p in pages.items()} | {"total": n_pages}
    report["timings_s"]["pages"] = round(tick() - t0)
    print(f"pages: {n_pages}")
    if a.until == "pages":
        return 0

    # --------------------------------------------------------------- panels
    t0 = tick()
    cache = pages_dir / "panels.json"
    if cache.exists() and "panels" not in force:
        rows = json.loads(cache.read_text())
    else:
        rows = ds.scan(cfg, pages, jobs=a.jobs)
        cache.write_text(json.dumps(rows))
    kept = ds.survivors(rows, dict(cfg.dataset.filters))
    rej = {}
    for r in rows:
        rej[r["reject"] or "kept_by_score"] = rej.get(r["reject"] or "kept_by_score", 0) + 1
    if _pngs(panels_dir) and "panels" not in force:
        print(f"panels: {len(_pngs(panels_dir))} already built (--from panels to redo)")
    else:
        hold = min(e["holdout"], len(kept) // 4)
        cfg["dataset"]["emit"]["holdout"] = hold
        st = ds.emit(cfg, kept, pages, ROOT)
        report["panels"] = {"candidates": len(rows), "survived_filters": len(kept),
                            "rejects": rej, **st}
        print(f"panels: {st['train']} train / {st['holdout']} holdout")
    report["timings_s"]["panels"] = round(tick() - t0)
    if a.until == "panels":
        print(json.dumps(report.get("panels", {}), indent=1)); return 0

    # ---------------------------------------------------------- magi, crops
    dirs = [panels_dir]
    if not a.skip_magi:
        t0 = tick()
        mj = pages_dir / "magi.json"
        have = json.loads(mj.read_text()) if mj.exists() else {}
        complete = all(magi_mod.key(v, p) in have for v, ps in pages.items() for p in ps)
        if not complete or "magi" in force:
            if "magi" in force:
                mj.unlink(missing_ok=True)
            venv = next((p for p in (ROOT / ".venv-magi/Scripts/python.exe",
                                     ROOT / ".venv-magi/bin/python") if p.exists()), None)
            if venv:                       # Magi needs transformers 4.x, the main env has 5
                cmd = [str(venv), str(ROOT / "scripts/magi_detect.py"), "--config", a.config]
                cmd += (["--src", a.src] if a.src else []) + (["--limit", str(a.limit)] if a.limit else [])
                print("magi: running in .venv-magi")
                subprocess.run(cmd, check=True)
                have = json.loads(mj.read_text())
            else:
                try:
                    have = magi_mod.detect(pages, mj)
                except (ImportError, OSError, ValueError) as err:
                    raise SystemExit(f"magi failed here ({type(err).__name__}: {err}).\n"
                                     f"Create .venv-magi, see requirements-magi.txt.")
        report["magi"] = {"pages": len(have),
                          "character_boxes": sum(len(r["characters"]) for r in have.values())}
        report["timings_s"]["magi"] = round(tick() - t0)

        t0 = tick()
        if _pngs(crops_dir) and "crops" not in force:
            print(f"crops: {len(_pngs(crops_dir))} already built (--from crops to redo)")
        else:
            shutil.rmtree(crops_dir, ignore_errors=True)
            report["crops"] = crops_mod.build(cfg, have, pages, crops_dir)
            print(f"crops: {report['crops']['kept']}  rejected {report['crops']['rejected']}")
        report["timings_s"]["crops"] = round(tick() - t0)
        dirs.append(crops_dir)
    if a.until in ("magi", "crops"):
        return 0

    # ---------------------------------------------------------- tag, filter
    t0 = tick()
    stage_tag(cfg, dirs, report)
    report["timings_s"]["tag"] = round(tick() - t0)
    stage_filter(cfg, dirs, report)

    final = {d.name: len(_pngs(d)) for d in dirs}
    hold_dir = out.parent / f"{out.name}_holdout"
    report["final"] = final | {"holdout": len(_pngs(hold_dir))}
    w = {d.name: int(d.name.split("_")[0]) for d in dirs}
    report["images_per_epoch"] = sum(final[k] * w[k] for k in final)
    report["dataset_config"] = dict(cfg.dataset)
    rp = out.parent / f"{out.name}_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str))

    print("\n=== dataset ready ===")
    for k, v in report["final"].items():
        print(f"  {k:<28}{v:>6}")
    print(f"  images per epoch            {report['images_per_epoch']:>6}")
    print(f"  train_data_dir = {cfg.paths.dataset}")
    print(f"  report -> {rp.relative_to(ROOT)}")
    if sum(final.values()) < 300:
        print("\nWARNING: under 300 training images. Add chapters before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
