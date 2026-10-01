"""Per-artist style LoRA training.

Plain LoRA, not QLoRA: QLoRA is a quantized-base trick for 7B+ language models.
SD 1.5's UNet is ~860M parameters and fits a free T4 at 512px with batch 1 and
gradient checkpointing.

This wraps kohya_ss `sd-scripts`, which is the de facto standard for SD 1.5 style
LoRAs - better bucketing and caption handling than diffusers' own training script.

    git clone https://github.com/kohya-ss/sd-scripts
    pip install -r sd-scripts/requirements.txt

    python -m manga.train_lora --config configs/conan.yaml \
        --data data/datasets/conan --scripts ./sd-scripts
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .config import load


def build_command(cfg, data_dir: Path, scripts_dir: Path, out_dir: Path) -> list[str]:
    lora = cfg.lora
    return [
        "accelerate", "launch",
        str(scripts_dir / "train_network.py"),
        "--pretrained_model_name_or_path", cfg.model.checkpoint,
        "--train_data_dir", str(data_dir),
        "--output_dir", str(out_dir),
        "--output_name", f"{cfg.artist.series}_style",
        "--network_module", "networks.lora",
        "--network_dim", str(lora.rank),
        "--network_alpha", str(lora.alpha),
        "--learning_rate", str(lora.learning_rate),
        "--max_train_steps", str(lora.max_train_steps),
        "--train_batch_size", str(lora.batch_size),
        "--resolution", str(lora.resolution),
        "--mixed_precision", "fp16",
        "--save_precision", "fp16",
        "--gradient_checkpointing",      # the T4 needs this
        "--xformers",
        "--cache_latents",
        "--enable_bucket",               # panels are not all the same aspect ratio
        "--clip_skip", str(cfg.model.clip_skip),
        "--save_every_n_epochs", "1",
        "--logging_dir", str(out_dir / "logs"),
        "--log_with", "wandb",
        "--wandb_run_name", f"{cfg.artist.series}_r{lora.rank}_lr{lora.learning_rate}",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="conan.yaml")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--scripts", type=Path, default=Path("./sd-scripts"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load(args.config)
    out = args.out or Path(cfg.paths.loras) / cfg.artist.series
    out.mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg, args.data, args.scripts, out)
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
