"""Model loading.

Four pretrained components, nothing trained here:
  base checkpoint  - the ability to draw
  CLIP skip 2      - read the text-encoder layer the anime checkpoint was trained on
  ControlNet       - pose, from the outline
  IP-Adapter Plus  - identity, from the reference

GOTCHA: never call enable_attention_slicing(). It swaps in an attention class the
IP-Adapter loader cannot re-create, and loading fails with
`SlicedAttnProcessor.__init__() missing 1 required positional argument`.
Use pipe.vae.enable_slicing() instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from compel import Compel
from controlnet_aux import LineartAnimeDetector
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel

from .config import Config


@dataclass
class Models:
    """Everything the generation functions need, loaded once."""

    pipe: StableDiffusionPipeline
    cn_pipe: StableDiffusionControlNetPipeline
    compel: Compel
    lineart: LineartAnimeDetector
    device: str = "cuda"


def load(cfg: Config, device: str = "cuda") -> Models:
    m = cfg.model
    dtype = getattr(torch, m.dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        m.checkpoint,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )

    # CLIP skip: 12 layers total, clip_skip=2 means read the output of layer 11.
    if m.clip_skip and m.clip_skip > 1:
        pipe.text_encoder = CLIPTextModel.from_pretrained(
            m.checkpoint,
            subfolder="text_encoder",
            num_hidden_layers=13 - m.clip_skip,
            torch_dtype=dtype,
        ).to(device)

    compel = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=False,
    )

    controlnet = ControlNetModel.from_pretrained(m.controlnet, torch_dtype=dtype).to(device)

    # Shares vae / unet / text_encoder with `pipe`, so no second copy in VRAM.
    # It therefore also inherits IP-Adapter automatically once that is loaded.
    cn_pipe = StableDiffusionControlNetPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipe.vae.enable_slicing()  # NOT enable_attention_slicing - see module docstring

    pipe.load_ip_adapter(
        m.ip_adapter_repo, subfolder="models", weight_name=m.ip_adapter_weight
    )
    cn_pipe.image_encoder = pipe.image_encoder
    cn_pipe.feature_extractor = pipe.feature_extractor

    lineart = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")

    return Models(pipe=pipe, cn_pipe=cn_pipe, compel=compel, lineart=lineart, device=device)


def load_lora(models: Models, path: str, scale: float = 1.0) -> None:
    """Attach a per-artist style LoRA to the shared UNet."""
    models.pipe.load_lora_weights(path)
    models.pipe.fuse_lora(lora_scale=scale)
