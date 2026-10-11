"""Same poses, same seed, several LoRAs side by side. One image per cell.

    python scripts/compare_loras.py --config spyxfamily_eval.yaml \
        --poses yor_seated.png loid_running.png \
        --loras none models/loras/spyxfamily/spyxfamily_style-000002.safetensors ...

Rows = poses. Columns = outline, then one per LoRA ('none' = B3, no LoRA).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manga import generate, pipeline, prompts, sketches
from manga.config import load
from manga.utils import grid, save


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="spyxfamily_eval.yaml")
    ap.add_argument("--poses", nargs="+", required=True, help="outline file names from the config")
    ap.add_argument("--loras", nargs="+", required=True, help="paths, or 'none' for no LoRA")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    models = pipeline.load(cfg)
    poses = [p for p in cfg.poses if p["file"] in args.poses]
    missing = set(args.poses) - {p["file"] for p in poses}
    if missing:
        raise SystemExit(f"not in config: {sorted(missing)}")

    # inputs once
    jobs = []
    for pose in poses:
        char = cfg.characters[pose["character"]] if "character" in pose else cfg.character
        jobs.append(dict(
            control=sketches.load_sketch(Path(cfg.paths.outlines) / pose["file"]),
            reference=sketches.load_reference(pose.get("reference", char["reference"])),
            prompt=prompts.build(cfg, pose.get("text", ""), pose.get("description", char["description"])),
            negative=prompts.negative(cfg, char.get("negative_props", "")),
        ))

    cols = {i: [] for i in range(len(jobs))}
    for lora in args.loras:
        if lora != "none":
            pipeline.load_lora(models, lora, args.scale)
        print("==>", lora)
        for i, j in enumerate(jobs):
            img = generate.outline_to_image(models, cfg, j["prompt"], j["negative"],
                                            j["control"], j["reference"], n=1)[0]
            cols[i].append(img)
        if lora != "none":
            models.pipe.unload_lora_weights()

    tiles = []
    for i, j in enumerate(jobs):
        tiles += [j["control"]] + cols[i]
    out = Path(args.out or Path(cfg.paths.results) / "compare_loras.png")
    save(grid(tiles, cols=len(args.loras) + 1), out)
    print("columns: outline |", " | ".join(Path(l).stem for l in args.loras))
    print("saved ->", out)


if __name__ == "__main__":
    main()
