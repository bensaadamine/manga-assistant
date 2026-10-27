#!/usr/bin/env python
"""Magi stage on its own, so it can run in .venv-magi (see requirements-magi.txt).

    .venv-magi\\Scripts\\python scripts/magi_detect.py --config bungo.yaml

prepare_series.py calls this automatically when .venv-magi exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga import config as cfgmod            # noqa: E402
from manga.dataset import magi                # noqa: E402
from manga.dataset.pipeline import extract_pages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    if a.limit:
        cfg["paths"]["pages"] += "_smoke"
    pages = extract_pages(cfg, ROOT, src_root=a.src, limit=a.limit)
    magi.detect(pages, ROOT / cfg.paths.pages / "magi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
