#!/usr/bin/env python
"""Step 3b - versions a series' dataset, holdout and LoRA checkpoints with DVC, then pushes.

    python scripts/dvc_version.py --config bungo.yaml

Prints the git command that records the new .dvc pointers; run it yourself so
the commit message says which LoRA / dataset version this is.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga.config import load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    cfg = load(a.config)
    name = cfg.series.name
    targets = [cfg.paths.dataset, f"{cfg.paths.dataset}_holdout", f"{cfg.paths.loras}/{name}"]
    targets = [t for t in targets if (ROOT / t).exists()]
    if not targets:
        raise SystemExit("nothing to version yet - build the dataset first")

    dvc = [shutil.which("dvc")] if shutil.which("dvc") else [sys.executable, "-m", "dvc"]
    subprocess.run(dvc + ["add", *targets], cwd=ROOT, check=True)
    if not a.no_push:
        subprocess.run(dvc + ["push"], cwd=ROOT, check=True)

    pointers = " ".join(f"{t}.dvc" for t in targets)
    print(f"\nversioned: {', '.join(targets)}\nnow record it in git:\n"
          f"  git add {pointers} configs/{Path(a.config).name} configs/registry.yaml\n"
          f'  git commit -m "{name}: dataset + LoRA versions"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
