"""The inference service shared by the web UI and the REST API: one model in VRAM, one request at a time.

    reference + outline + description (+ LoRA) -> images, each saved with a JSON record

Models load lazily on the first request (~1 min) and are freed whenever a job
(dataset build, training) needs the GPU: 8 GB cannot hold both.

Every image goes to outputs/app/<date>/ next to a .json holding everything needed
to regenerate it: git commit, LoRA file and scale, seed, prompts, parameters.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import prompts, sketches
from .config import ROOT, load
from .jobs import Busy, JobRunner


def _git(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def prepare_reference(path: str | Path, clean: bool = True) -> Image.Image:
    """Finding 6: IP-Adapter copies everything, so white background + greyscale."""
    ref = sketches.load_reference(path)
    if clean:
        ref = sketches.whiten_background(ref).convert("L").convert("RGB")
    return ref


def prepare_outline(path: str | Path, kind: str = "digital") -> Image.Image:
    """`photo` = a phone photo of a paper sketch; `digital` = a clean drawn file."""
    loader = sketches.load_photo_sketch if kind == "photo" else sketches.load_sketch
    return loader(path)


def with_trigger(description: str, trigger: str) -> str:
    """A LoRA only fires with its trigger word in the prompt; add it if missing."""
    d = description.strip()
    if trigger and trigger not in d.replace(",", " ").split():
        d = f"{trigger}, {d}" if d else trigger
    return d


class ModelService:
    def __init__(self, cfg_name: str = "base.yaml", root: Path = ROOT,
                 jobs: JobRunner | None = None, device: str | None = None):
        self.root = Path(root)
        self.cfg = load(cfg_name)
        self.device = device or os.environ.get("MANGA_DEVICE", "cuda")
        self.models = None
        self.lora: str | None = None
        self.lock = threading.Lock()
        self.jobs = jobs
        if jobs is not None:
            jobs.before_start = self.unload

    # ------------------------------------------------------------- models
    def loaded(self) -> bool:
        return self.models is not None

    def _ensure(self):
        if self.jobs is not None and self.jobs.running():
            raise Busy(f"the GPU is busy with '{self.jobs.name}' - wait for it or stop it")
        if self.models is None:
            from . import pipeline
            self.models = pipeline.load(self.cfg, device=self.device)
            self.lora = None

    def unload(self) -> None:
        with self.lock:
            if self.models is None:
                return
            self.models, self.lora = None, None
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _set_lora(self, lora: str | None, scale: float) -> None:
        """Swap the style LoRA only when it changes; a scale change needs no reload."""
        from . import pipeline
        if lora != self.lora:
            if self.lora is not None:
                pipeline.unload_lora(self.models, "style")
                self.lora = None
            if lora:
                pipeline.load_lora(self.models, str(self.root / lora), scale)
                self.lora = lora
        elif lora:
            pipeline.set_lora_scale(self.models, scale)

    # --------------------------------------------------------- generation
    def generate(self, reference: str | Path, outline: str | Path, description: str,
                 pose: str = "", negative_props: str = "", outline_kind: str = "digital",
                 lora: str | None = None, lora_scale: float = 1.0, trigger: str = "",
                 n: int = 1, seed: int | None = None, clean_reference: bool = True,
                 save: bool = True, **overrides) -> tuple[list[Image.Image], dict]:
        from . import generate as gen

        control = prepare_outline(outline, outline_kind)
        ref = prepare_reference(reference, clean_reference)
        desc = with_trigger(description, trigger if lora else "")
        prompt = prompts.build(self.cfg, pose, desc)
        negative = prompts.negative(self.cfg, negative_props)
        if seed is not None:
            overrides["seed"] = int(seed)
        params = {**self.cfg.generation, **overrides}

        with self.lock:
            self._ensure()
            self._set_lora(lora, lora_scale)
            images = gen.outline_to_image(self.models, self.cfg, prompt, negative, control,
                                          ref, n=int(n), **overrides)

        record = {"time": datetime.now().isoformat(timespec="seconds"), "git": _git(self.root),
                  "lora": lora, "lora_scale": lora_scale if lora else None,
                  "reference": str(reference), "outline": str(outline),
                  "outline_kind": outline_kind, "clean_reference": clean_reference,
                  "prompt": prompt, "negative": negative,
                  "params": dict(params), "model": dict(self.cfg.model)}
        if save:
            record["files"] = self._save(images, record)
        return images, record

    def compare(self, checkpoints: list[str | None], **kw) -> list[tuple[Image.Image, str]]:
        """Same inputs and seed through each checkpoint (None = no LoRA, the B3 baseline)."""
        out = []
        for ck in checkpoints:
            imgs, _ = self.generate(lora=ck, n=1, **kw)
            out.append((imgs[0], Path(ck).stem if ck else "no LoRA (B3)"))
        return out

    def _save(self, images: list[Image.Image], record: dict) -> list[str]:
        d = self.root / "outputs" / "app" / f"{datetime.now():%Y%m%d}"
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%H%M%S}"
        files = []
        for i, im in enumerate(images):
            p = d / f"{stem}_{i}.png"
            im.save(p)
            files.append(p.relative_to(self.root).as_posix())
            seed = im.info.get("seed", record["params"].get("seed"))
            p.with_suffix(".json").write_text(json.dumps({**record, "seed": seed}, indent=1,
                                                         default=str), encoding="utf-8")
        return files
