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


INK_KEYS = ("black_ratio", "midtone_ratio", "edge_density", "white_ratio")

# Every ink statistic is resolution-dependent (edge_density most of all), so
# generated 512x768 frames and 512x512 training panels are compared at one size.
INK_SIZE = (512, 512)


def ink_statistics(image: Image.Image, threshold: int = 128) -> dict[str, float]:
    """Cheap proxies for 'is this actually inked, or soft greyscale rendering?'

    FID can be fooled by images that look like anime but are rendered with gradients.
    Real manga is bimodal: mostly pure black and pure white, with screentone dots.

    black_ratio  - fraction of near-black pixels
    midtone_ratio- fraction of mid-greys. HIGH means soft rendering, LOW means inked.
    edge_density - proxy for line work and screentone frequency
    white_ratio  - fraction of near-white pixels. Catches the grey-field failure,
                   where nothing is black but nothing is paper-white either.
    """
    a = np.asarray(image.convert("L").resize(INK_SIZE, Image.BICUBIC), dtype=np.float32)
    black = float((a < threshold * 0.4).mean())
    midtone = float(((a >= 64) & (a <= 192)).mean())
    white = float((a > 235).mean())
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    return {
        "black_ratio": black,
        "midtone_ratio": midtone,
        "edge_density": float((gx + gy) / 2),
        "white_ratio": white,
    }


def ink_target(heldout_dir: str | Path, limit: int | None = None) -> dict[str, float]:
    """Mean ink statistics over the artist's own held-out panels.

    This is what the generated images are supposed to look like. Without it the raw
    statistics are unitless and un-trackable; against it they become a distance.
    """
    paths = sorted(Path(heldout_dir).rglob("*.png"))[:limit]
    if not paths:
        raise FileNotFoundError(f"no .png under {heldout_dir}")
    rows = [ink_statistics(Image.open(p)) for p in paths]
    return {k: float(np.mean([r[k] for r in rows])) for k in INK_KEYS}


def style_gap(stats: dict[str, float], target: dict[str, float]) -> float:
    """One number: how far this image's ink profile sits from the artist's.

    Each statistic is divided by its target before differencing, so a ratio in the
    percents and an edge density in the tens contribute comparably. 0 is perfect;
    the scale is otherwise arbitrary and only comparable between our own runs.
    """
    return float(
        np.mean([abs(stats[k] - target[k]) / max(target[k], 1e-6) for k in INK_KEYS])
    )


# ---------------------------------------------------------------- pose


_POSE_DETECTOR = None


def _detector():
    global _POSE_DETECTOR
    if _POSE_DETECTOR is None:
        from controlnet_aux import OpenposeDetector

        _POSE_DETECTOR = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return _POSE_DETECTOR


def _keypoints(image: Image.Image) -> dict[int, tuple[float, float]]:
    """Body keypoints as {index: (x, y)} in normalised 0-1 coordinates.

    controlnet_aux returns PoseResult objects whose body keypoints may be None where
    nothing was detected; only the ones present in BOTH images are ever compared.
    """
    poses = _detector().detect_poses(np.asarray(image.convert("RGB")))
    if not poses or poses[0].body is None:
        return {}
    out = {}
    for i, kp in enumerate(poses[0].body.keypoints):
        if kp is not None and getattr(kp, "score", 1.0) > 0.1:
            out[i] = (float(kp.x), float(kp.y))
    return out


def pose_distance(outline: Image.Image, generated: Image.Image) -> tuple[float, int]:
    """Mean keypoint distance between the input outline and the output, normalised
    by image diagonal. Lower is better. Returns (distance, keypoints compared).

    Caveat worth stating in the report: OpenPose was trained on photographs and on
    drawn figures that have a face. Our outlines have an empty head oval, so the
    head keypoints are frequently missed and the count matters as much as the
    distance - a very low distance over three keypoints means nothing.
    """
    a, b = _keypoints(outline), _keypoints(generated)
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan"), 0
    d = [float(np.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1])) for i in shared]
    return float(np.mean(d)), len(shared)


# ---------------------------------------------------------------- reporting


def summarise(rows: list[dict]) -> dict[str, float]:
    """Mean of every numeric column. Feed straight into a W&B summary."""
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
