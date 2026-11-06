"""Creates a new series from uploaded volumes: files into data/volumes/<name>/, config into configs/<name>.yaml.

One uploaded file = one volume:
    .pdf   used as is                       -> `pdf:` volume
    .zip   unpacked to a folder of images   -> `images:` volume (chapter subfolders allowed)

The written config has the same shape as bungo.yaml, so prepare_series.py and
train_lora.py need no change. LoRA settings are bungo's (the latest recipe);
the step count is set at training time from the dataset size.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from .config import ROOT

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")

TEMPLATE = """\
# Series '{name}' - created by the app on upload.
extends: dataset.yaml

series:
  name: {name}
  trigger: {trigger}
  volumes:
{volumes}

dataset:
  pages:
    head_drop: {head_drop}          # pages skipped at the start of each volume (cover, colour plates)
    tail_drop: {tail_drop}          # pages skipped at the end (afterword, ads)

paths:
  pages: data/pages/{name}
  dataset: data/datasets/{name}

# Same recipe as bungo v1. max_train_steps / save_every_steps are overridden from
# the app (Train tab) to ~2.5 epochs of whatever dataset prepare_series builds.
lora:
  rank: 32
  alpha: 32
  learning_rate: 1.0e-4
  text_encoder_lr: 5.0e-5
  lr_scheduler: cosine_with_restarts
  warmup_steps: 100
  max_train_steps: 12000
  save_every_steps: 1200
  batch_size: 1
  resolution: 512
  max_bucket_reso: 1024
  optimizer: AdamW8bit
  attention: sdpa
"""


def _unzip(src: Path, dest: Path) -> None:
    """Extract, refusing paths that would escape `dest` (zip-slip)."""
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    with zipfile.ZipFile(src) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if base != target and base not in target.parents:
                raise ValueError(f"unsafe path in zip: {m.filename}")
        z.extractall(dest)


def create(name: str, trigger: str, files: list[str | Path], head_drop: int = 0,
           tail_drop: int = 0, root: Path = ROOT, overwrite: bool = False) -> Path:
    """Copy the volumes in and write configs/<name>.yaml. Returns the config path."""
    if not NAME_RE.match(name):
        raise ValueError("series name: lowercase letters, digits, _ ; 2-31 chars, starts with a letter")
    trigger = trigger.strip() or f"{name}style"
    if not re.fullmatch(r"[a-z0-9_]+", trigger):
        raise ValueError("trigger word: one lowercase word, e.g. 'bsdstyle'")
    if not files:
        raise ValueError("upload at least one volume (.pdf or .zip)")

    root = Path(root)
    cfg_path = root / "configs" / f"{name}.yaml"
    if cfg_path.exists() and not overwrite:
        raise FileExistsError(f"configs/{name}.yaml already exists")

    vol_dir = root / "data" / "volumes" / name
    vols = []
    for i, f in enumerate(sorted(map(Path, files), key=lambda p: p.name.lower()), start=1):
        vid = f"v{i}"
        ext = f.suffix.lower()
        if ext == ".pdf":
            dest = vol_dir / f"{vid}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            vols.append(f'    - {{id: {vid}, pdf: "{dest.relative_to(root).as_posix()}"}}  # {f.name}')
        elif ext == ".zip":
            dest = vol_dir / vid
            shutil.rmtree(dest, ignore_errors=True)
            _unzip(f, dest)
            vols.append(f'    - {{id: {vid}, images: "{dest.relative_to(root).as_posix()}"}}  # {f.name}')
        else:
            raise ValueError(f"{f.name}: only .pdf or .zip volumes")

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(TEMPLATE.format(name=name, trigger=trigger, volumes="\n".join(vols),
                                        head_drop=int(head_drop), tail_drop=int(tail_drop)),
                        encoding="utf-8")
    return cfg_path


def suggested_steps(images_per_epoch: int) -> tuple[int, int]:
    """~2.5 epochs, clamped to 3000-12000, checkpoint every tenth -> (steps, save_every).

    Bungo: 4697 images/epoch -> 11700 steps; its best checkpoint was step 4800 (~1 epoch),
    so 10 checkpoints across 2.5 epochs brackets the sweet spot with room either side.
    """
    steps = int(round(min(12000, max(3000, 2.5 * images_per_epoch)), -2))
    return steps, max(100, steps // 10)
