"""Generate every pose for one character and save a comparison grid.

    python scripts/run_poses.py --config conan.yaml
    python scripts/run_poses.py --config conan.yaml --lora models/loras/detective_conan/..safetensors
    python scripts/run_poses.py --config conan.yaml --baseline b1   # pose only, no identity
"""
from __future__ import annotations

import argparse
from pathlib import Path

from manga import generate, pipeline, prompts, sketches
from manga.config import load
from manga.utils import grid, save

BASELINES = {
    "b1": dict(use_reference=False),                 # ControlNet only
    "b3": dict(use_reference=True),                  # ControlNet + IP-Adapter  (the POC)
    "b4": dict(use_reference=True),                  # + LoRA, pass --lora as well
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="conan.yaml")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--baseline", default="b3", choices=list(BASELINES))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--photo-sketches", action="store_true",
                    help="outlines are photos of paper, not digital files")
    args = ap.parse_args()

    cfg = load(args.config)
    models = pipeline.load(cfg)
    if args.lora:
        pipeline.load_lora(models, args.lora)

    use_ref = BASELINES[args.baseline]["use_reference"]
    reference = sketches.load_reference(cfg.character.reference) if use_ref else None
    negative = prompts.character_negative(cfg)
    loader = sketches.load_photo_sketch if args.photo_sketches else sketches.load_sketch

    tiles = []
    for pose in cfg.poses:
        control = loader(Path(cfg.paths.outlines) / pose["file"])
        images = generate.outline_to_image(
            models, cfg, prompts.character_prompt(cfg, pose["text"]),
            negative, control, reference, n=args.n,
        )
        tiles += [control] + images

    out = Path(cfg.paths.results) / f"{cfg.character.id}_{args.baseline}.png"
    save(grid(tiles, cols=args.n + 1), out)
    print("saved ->", out)


if __name__ == "__main__":
    main()
