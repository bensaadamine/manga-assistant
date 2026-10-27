"""Parameter sweep for the ablation figures.

    python scripts/sweep.py --config conan.yaml --param cn_scale --values 0.3 0.5 0.7 0.9
    python scripts/sweep.py --config conan.yaml --param ip_scale --values 0.4 0.6 0.8
    python scripts/sweep.py --config conan.yaml --param cn_end   --values 0.4 0.6 0.8 1.0
    python scripts/sweep.py --config spyxfamily_eval.yaml --param lora_scale \
        --values 0.4 0.6 0.8 1.0 --lora models/loras/spyxfamily/spyxfamily_style.safetensors

lora_scale is not a generation parameter: it lives on the adapter, not in
cfg.generation, so it cannot go through generate.sweep and gets its own loop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manga import generate, pipeline, prompts, sketches
from manga.config import load
from manga.utils import grid, save

GEN_PARAMS = ["cn_scale", "ip_scale", "cn_end"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="conan.yaml")
    ap.add_argument("--param", required=True, choices=GEN_PARAMS + ["lora_scale"])
    ap.add_argument("--values", type=float, nargs="+", required=True)
    ap.add_argument("--pose", type=int, default=1, help="index into cfg.poses")
    ap.add_argument("--lora", default=None, help="required when --param lora_scale")
    args = ap.parse_args()

    if args.param == "lora_scale" and not args.lora:
        ap.error("--param lora_scale needs --lora PATH")

    cfg = load(args.config)
    models = pipeline.load(cfg)
    if args.lora:
        # BUG fixed 22 Oct: values[0] was used as the LoRA scale for EVERY param,
        # so an ip_scale sweep starting at 0.4 also ran the LoRA at 0.4.
        first = args.values[0] if args.param == "lora_scale" else 1.0
        pipeline.load_lora(models, args.lora, scale=first)

    pose = cfg.poses[args.pose]
    char = cfg.characters[pose["character"]] if "character" in pose else cfg.character

    control = sketches.load_sketch(Path(cfg.paths.outlines) / pose["file"])
    reference = sketches.load_reference(pose.get("reference", char["reference"]))
    prompt = prompts.build(cfg, pose.get("text", ""),
                           pose.get("description", char["description"]))
    negative = prompts.negative(cfg, char.get("negative_props", ""))

    if args.param == "lora_scale":
        images = []
        for v in args.values:
            pipeline.set_lora_scale(models, v)
            images += generate.outline_to_image(
                models, cfg, prompt, negative, control, reference, n=1
            )
    else:
        images = generate.sweep(
            models, cfg, prompt, negative, control, reference, args.param, args.values
        )

    out = Path(cfg.paths.results) / f"sweep_{args.param}.png"
    save(grid([control] + images, cols=len(images) + 1), out)
    print("values:", args.values)
    print("saved ->", out)


if __name__ == "__main__":
    main()
