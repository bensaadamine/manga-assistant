"""Generate every pose for one character and save a comparison grid.

    python scripts/run_poses.py --config conan.yaml
    python scripts/run_poses.py --config conan.yaml --lora models/loras/detective_conan/..safetensors
    python scripts/run_poses.py --config conan.yaml --baseline b1   # pose only, no identity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    ap.add_argument("--no-pose-text", action="store_true",
                    help="ablation: drop the pose sentence, keep ControlNet only")
    args = ap.parse_args()

    cfg = load(args.config)
    models = pipeline.load(cfg)
    if args.lora:
        pipeline.load_lora(models, args.lora)

    use_ref = BASELINES[args.baseline]["use_reference"]
    loader = sketches.load_photo_sketch if args.photo_sketches else sketches.load_sketch

    # Identity lives once per character under `characters:`; a pose only names which
    # one it is. A pose may still override any field, and falls back to the single
    # `character:` block when it names none.
    tiles = []
    for pose in cfg.poses:
        char = cfg.characters[pose["character"]] if "character" in pose else cfg.character
        control = loader(Path(cfg.paths.outlines) / pose["file"])
        ref_path = pose.get("reference", char["reference"])
        reference = sketches.load_reference(ref_path) if use_ref else None
        description = pose.get("description", char["description"])
        negative = prompts.negative(cfg, char.get("negative_props", ""))
        text = "" if args.no_pose_text else pose.get("text", "")
        print(f"{pose['file']}  ref={ref_path if use_ref else 'none'}")
        images = generate.outline_to_image(
            models, cfg, prompts.build(cfg, text, description),
            negative, control, reference, n=args.n,
        )
        tiles += [control] + images

    suffix = f"{args.baseline}_nopose" if args.no_pose_text else args.baseline
    out = Path(cfg.paths.results) / f"{cfg.character.id}_{suffix}.png"
    save(grid(tiles, cols=args.n + 1), out)
    print("saved ->", out)


if __name__ == "__main__":
    main()
