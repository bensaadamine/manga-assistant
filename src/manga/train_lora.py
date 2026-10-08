"""Per-artist style LoRA training.

Plain LoRA, not QLoRA: QLoRA is a quantized-base trick for 7B+ language models.
SD 1.5's UNet is ~860M parameters and fits a free T4 at 512px with batch 1 and
gradient checkpointing.

Wraps kohya_ss `sd-scripts`, the de facto standard for SD 1.5 style LoRAs —
better bucketing and caption handling than diffusers' own training script.

    git clone https://github.com/kohya-ss/sd-scripts

    python -m manga.train_lora --config configs/spyxfamily.yaml --dry-run
    python -m manga.train_lora --config configs/spyxfamily.yaml --scripts ./sd-scripts

`--dry-run` prints the command without running it, which is how to check a config
change without burning GPU minutes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load


def series_of(cfg):
    """Name and trigger, from either the `series:` or the older `artist:` block."""
    if "series" in cfg:
        return cfg.series.name, cfg.series.trigger
    return cfg.artist.series, cfg.artist.trigger_word


def build_command(cfg, data_dir: Path, scripts_dir: Path, out_dir: Path,
                  wandb: bool = False) -> list[str]:
    lora = cfg.lora
    name, trigger = series_of(cfg)
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "2",
        str(scripts_dir / "train_network.py"),
        "--pretrained_model_name_or_path", cfg.model.checkpoint,
        "--train_data_dir", str(data_dir),
        "--output_dir", str(out_dir),
        "--output_name", f"{name}_style",
        "--network_module", "networks.lora",
        "--network_dim", str(lora.rank),
        "--network_alpha", str(lora.alpha),
        "--learning_rate", str(lora.learning_rate),
        "--unet_lr", str(lora.learning_rate),
        "--text_encoder_lr", str(lora.text_encoder_lr),
        "--lr_scheduler", lora.lr_scheduler,
        "--lr_warmup_steps", str(lora.warmup_steps),
        "--max_train_steps", str(lora.max_train_steps),
        "--train_batch_size", str(lora.batch_size),
        "--resolution", f"{lora.resolution},{lora.resolution}",
        "--optimizer_type", lora.optimizer,
        "--mixed_precision", "fp16",
        "--save_precision", "fp16",
        "--gradient_checkpointing",
        "--cache_latents",
        "--caption_extension", ".txt",
        # The trigger must stay the first token; everything after it may shuffle,
        # which stops the LoRA from memorising one fixed tag order.
        "--shuffle_caption",
        "--keep_tokens", "1",
        # Panels are not all one aspect ratio, and never upscale: the min_side
        # filter already guarantees every panel is downscaled into its bucket.
        "--enable_bucket",
        "--min_bucket_reso", "320",
        "--max_bucket_reso", str(lora.max_bucket_reso),
        "--bucket_no_upscale",
        "--clip_skip", str(cfg.model.clip_skip),
        "--seed", str(cfg.generation.seed),
        "--save_every_n_epochs", "1",
        "--max_data_loader_n_workers", "2",
        "--logging_dir", str(out_dir / "logs"),
    ]
    # RTX 50 (Blackwell, sm_120) has no xformers wheels; sdpa is the way in.
    cmd.append("--sdpa" if lora.attention == "sdpa" else "--xformers")
    if wandb:
        cmd += ["--log_with", "wandb",
                "--wandb_run_name", f"{name}_r{lora.rank}_lr{lora.learning_rate}"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="spyxfamily.yaml")
    ap.add_argument("--data", type=Path, default=None,
                    help="train_data_dir; defaults to the config's dataset path")
    ap.add_argument("--scripts", type=Path, default=Path("./sd-scripts"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load(args.config)
    name, trigger = series_of(cfg)
    data = args.data or Path(cfg.paths.dataset)
    out = args.out or Path(cfg.paths.loras) / name

    concepts = sorted(p.name for p in data.glob("*_*") if p.is_dir())
    if not concepts:
        raise SystemExit(
            f"no '<repeats>_<trigger>' folder under {data} — "
            f"run scripts/build_dataset.py --config {args.config} first")
    n = sum(1 for p in (data / concepts[0]).glob("*.png"))
    print(f"{name}: {n} images in {concepts[0]}, trigger '{trigger}'")

    out.mkdir(parents=True, exist_ok=True)
    cmd = build_command(cfg, data, args.scripts, out, wandb=args.wandb)
    print(" ".join(cmd))
    if not args.dry_run:
        if not (args.scripts / "train_network.py").exists():
            raise SystemExit(f"{args.scripts}/train_network.py not found — "
                             f"git clone https://github.com/kohya-ss/sd-scripts")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
