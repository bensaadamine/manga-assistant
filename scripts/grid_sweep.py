"""IP x CN grid on one pose: rows = ip_scale, columns = cn_scale.

    python scripts/grid_sweep.py --config spyxfamily_eval.yaml --pose 1 \
        --lora models/loras/spyxfamily/spyxfamily_style_v3.safetensors

Same seed in every cell, so only the two scales differ. IP and CN fight each other
(finding 17), which is why they are swept together and not one at a time.
Column 0 holds the outline (row 0) and the reference (row 1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw

from manga import generate, pipeline, prompts, sketches
from manga.config import load
from manga.utils import grid, save


def label(im: Image.Image, text: str) -> Image.Image:
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 170, 22), fill="white")
    d.text((5, 5), text, fill="black")
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="spyxfamily_eval.yaml")
    ap.add_argument("--pose", type=int, required=True, help="index into cfg.poses")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--ip", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--cn", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    args = ap.parse_args()

    cfg = load(args.config)
    models = pipeline.load(cfg)
    if args.lora:
        pipeline.load_lora(models, args.lora, scale=args.lora_scale)

    pose = cfg.poses[args.pose]
    char = cfg.characters[pose["character"]]
    control = sketches.load_sketch(Path(cfg.paths.outlines) / pose["file"])
    reference = sketches.load_reference(pose.get("reference", char["reference"]))
    prompt = prompts.build(cfg, pose.get("text", ""), pose.get("description", char["description"]))
    negative = prompts.negative(cfg, char.get("negative_props", ""))

    side = [control, reference] + [Image.new("RGB", control.size, "white")] * len(args.ip)
    cells = []
    for r, ip in enumerate(args.ip):
        cells.append(side[r])
        for cn in args.cn:
            im = generate.outline_to_image(models, cfg, prompt, negative, control, reference,
                                           n=1, ip_scale=ip, cn_scale=cn)[0]
            cells.append(label(im, f"ip {ip}  cn {cn}"))
            print(f"ip {ip}  cn {cn}  done")

    tag = "b4" if args.lora else "b3"
    out = Path(cfg.paths.results) / f"grid_{Path(pose['file']).stem}_{tag}.png"
    save(grid(cells, cols=len(args.cn) + 1), out)
    print("saved ->", out)


if __name__ == "__main__":
    main()
