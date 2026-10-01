"""Generation.

GOTCHA: once IP-Adapter is loaded, the UNet demands an image on EVERY call, even a
pure text-to-image one. Skip it and you get
`TypeError: argument of type 'NoneType' is not iterable`.
`_ip_kwargs` handles that by feeding a blank white image at strength 0.
"""
from __future__ import annotations

from typing import Sequence

from PIL import Image

from .config import Config
from .pipeline import Models
from .utils import seeds, tag_seeds

BLANK = Image.new("RGB", (224, 224), "white")


def _ip_kwargs(models: Models, reference: Image.Image | None, scale: float) -> dict:
    if reference is None:
        models.pipe.set_ip_adapter_scale(0.0)
        return {"ip_adapter_image": BLANK}
    models.pipe.set_ip_adapter_scale(scale)
    return {"ip_adapter_image": reference}


def _embed(models: Models, prompt: str, negative: str, n: int):
    pos = models.compel(prompt)
    neg = models.compel(negative)
    pos, neg = models.compel.pad_conditioning_tensors_to_same_length([pos, neg])
    return pos.repeat(n, 1, 1), neg.repeat(n, 1, 1)


def text_to_image(
    models: Models,
    cfg: Config,
    prompt: str,
    negative: str,
    reference: Image.Image | None = None,
    n: int = 1,
    **overrides,
) -> list[Image.Image]:
    """No outline. Used to build reference sheets, and for the B0/B2 baselines."""
    g = {**cfg.generation, **overrides}
    pos, neg = _embed(models, prompt, negative, n)
    images = models.pipe(
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        num_inference_steps=g["steps"],
        guidance_scale=g["cfg"],
        width=g["width"],
        height=g["height"],
        generator=seeds(g["seed"], n, models.device),
        **_ip_kwargs(models, reference, g["ip_scale"]),
    ).images
    return tag_seeds(images, g["seed"])


def outline_to_image(
    models: Models,
    cfg: Config,
    prompt: str,
    negative: str,
    control: Image.Image,
    reference: Image.Image | None = None,
    n: int = 1,
    **overrides,
) -> list[Image.Image]:
    """The main entry point.

    control   - the pose outline (white lines on black)
    reference - the character reference; None gives the B1 baseline (pose, no identity)

    cn_end is as important as cn_scale: diffusion sets composition early and surface
    detail late, so stopping ControlNet at 0.70 keeps the pose while letting the model
    clean up artifacts the outline would otherwise have traced onto the clothing.
    """
    g = {**cfg.generation, **overrides}
    pos, neg = _embed(models, prompt, negative, n)
    images = models.cn_pipe(
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        image=control,
        num_inference_steps=g["steps"],
        guidance_scale=g["cfg"],
        controlnet_conditioning_scale=g["cn_scale"],
        control_guidance_start=g["cn_start"],
        control_guidance_end=g["cn_end"],
        generator=seeds(g["seed"], n, models.device),
        width=control.width,
        height=control.height,
        **_ip_kwargs(models, reference, g["ip_scale"]),
    ).images
    return tag_seeds(images, g["seed"])


def sweep(
    models: Models,
    cfg: Config,
    prompt: str,
    negative: str,
    control: Image.Image,
    reference: Image.Image,
    param: str,
    values: Sequence[float],
) -> list[Image.Image]:
    """One image per value of a single parameter. Used for the ablation figures."""
    return [
        outline_to_image(models, cfg, prompt, negative, control, reference, n=1, **{param: v})[0]
        for v in values
    ]
