#!/usr/bin/env python
"""V3 step 4 - bakes a flat-manga LoRA into Counterfeit -> a new base model folder.

    python scripts/merge_base.py --lora models/base_loras/manga_lineart.safetensors --scale 0.5
    -> models/base/counterfeit-flat/   (diffusers format)

Then point `model.checkpoint` in configs/base.yaml at models/base/counterfeit-flat.
Both the pipeline (from_pretrained) and kohya (--pretrained_model_name_or_path)
read a diffusers folder, so nothing else changes. Retrain each artist LoRA on it.

Writes merge.json beside the weights: source checkpoint, LoRA file + sha256, scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402

from manga.config import load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--scale", type=float, required=True, help="the winning weight from V3 step 3")
    ap.add_argument("--out", default="models/base/counterfeit-flat")
    a = ap.parse_args()

    src = load("base.yaml").model.checkpoint
    out = ROOT / a.out
    pipe = StableDiffusionPipeline.from_pretrained(src, torch_dtype=torch.float16,
                                                   safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.load_lora_weights(a.lora, adapter_name="flat")
    pipe.fuse_lora(lora_scale=a.scale)          # bakes it into the UNet + text encoder weights
    pipe.unload_lora_weights()                  # drop the now-redundant adapter layers
    pipe.save_pretrained(out, safe_serialization=True)

    sha = hashlib.sha256(Path(a.lora).read_bytes()).hexdigest()
    (out / "merge.json").write_text(json.dumps({
        "base": src, "lora": Path(a.lora).name, "lora_sha256": sha, "scale": a.scale,
        "made": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    print(f"saved -> {out}\nset model.checkpoint: {a.out}   (configs/base.yaml)")


if __name__ == "__main__":
    main()
