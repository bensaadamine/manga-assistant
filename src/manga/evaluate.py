"""Evaluation metrics.

The headline figure for the report is B3 -> B4: ControlNet + IP-Adapter with and
without the per-artist LoRA, on the same cases. Everything here exists to produce
that one comparison honestly.

  identity  - DINOv2 cosine similarity, reference vs generated
  style     - FID against held-out panels by the same artist
  pose      - keypoint distance, input outline vs output  (needs a pose estimator)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------- identity


class IdentityScorer:
    """Cosine similarity in DINOv2 space. Higher is more similar.

    DINOv2 over CLIP-I because it is more sensitive to fine appearance detail and
    less to semantic category - we want "is this the same outfit", not "is this a boy".
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        out = self.model(**inputs).last_hidden_state[:, 0]  # CLS token
        return torch.nn.functional.normalize(out, dim=-1)

    def score(self, reference: Image.Image, generated: Image.Image) -> float:
        return float((self.embed(reference) @ self.embed(generated).T).item())


# ---------------------------------------------------------------- style


def style_fid(generated_dir: str | Path, heldout_dir: str | Path) -> float:
    """FID between generated images and held-out panels by the same artist.

    Lower is better. Expect the large drop between B3 and B4 - closing the style gap
    is the LoRA's entire job. Needs >= ~50 images per side to mean anything.
    """
    from cleanfid import fid

    return fid.compute_fid(str(generated_dir), str(heldout_dir), mode="clean")


def ink_statistics(image: Image.Image, threshold: int = 128) -> dict[str, float]:
    """Cheap proxies for 'is this actually inked, or soft greyscale rendering?'

    FID can be fooled by images that look like anime but are rendered with gradients.
    Real manga is bimodal: mostly pure black and pure white, with screentone dots.

    black_ratio  - fraction of near-black pixels
    midtone_ratio- fraction of mid-greys. HIGH means soft rendering, LOW means inked.
    edge_density - proxy for line work and screentone frequency
    """
    a = np.asarray(image.convert("L"), dtype=np.float32)
    black = float((a < threshold * 0.4).mean())
    midtone = float(((a >= 64) & (a <= 192)).mean())
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    return {
        "black_ratio": black,
        "midtone_ratio": midtone,
        "edge_density": float((gx + gy) / 2),
    }


# ---------------------------------------------------------------- pose


def pose_distance(outline: Image.Image, generated: Image.Image) -> float:
    """Mean keypoint distance between the input outline and the output, normalised
    by image diagonal. Lower is better.

    TODO: wire up controlnet_aux.OpenposeDetector on both images and compare the
    detected keypoints. Left as a stub so the metric contract is fixed first.
    """
    raise NotImplementedError("wire up OpenposeDetector from controlnet_aux")


# ---------------------------------------------------------------- reporting


def summarise(rows: list[dict]) -> dict[str, float]:
    """Mean of every numeric column. Feed straight into a W&B summary."""
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
