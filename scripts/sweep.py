"""Parameter sweep for the ablation figures.

    python scripts/sweep.py --config conan.yaml --param cn_scale --values 0.3 0.5 0.7 0.9
    python scripts/sweep.py --config conan.yaml --param ip_scale --values 0.4 0.6 0.8
    python scripts/sweep.py --config conan.yaml --param cn_end   --values 0.4 0.6 0.8 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from manga import generate, pipeline, prompts, sketches
from manga.config import load
from manga.utils import grid, save


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="conan.yaml")
    ap.add_argument("--param", required=True, choices=["cn_scale", "ip_scale", "cn_end"])
    ap.add_argument("--values", type=float, nargs="+", required=True)
    ap.add_argument("--pose", type=int, default=1, help="index into cfg.poses")
    args = ap.parse_args()

    cfg = load(args.config)
    models = pipeline.load(cfg)
    pose = cfg.poses[args.pose]

    control = sketches.load_sketch(Path(cfg.paths.outlines) / pose["file"])
    reference = sketches.load_reference(cfg.character.reference)

    images = generate.sweep(
        models, cfg,
        prompts.character_prompt(cfg, pose["text"]),
        prompts.character_negative(cfg),
        control, reference, args.param, args.values,
    )

    out = Path(cfg.paths.results) / f"sweep_{args.param}.png"
    save(grid([control] + images, cols=len(images) + 1), out)
    print("values:", args.values)
    print("saved ->", out)


if __name__ == "__main__":
    main()
