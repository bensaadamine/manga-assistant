import numpy as np
from PIL import Image

from manga import sketches


def _black_on_white(size=(200, 300)):
    img = Image.new("L", size, 255)
    img.paste(0, (50, 50, 150, 250))
    return img


def test_load_sketch_inverts_black_on_white(tmp_path):
    """ControlNet wants white lines on black. Paper sketches are the opposite."""
    p = tmp_path / "s.png"
    _black_on_white().save(p)
    out = np.asarray(sketches.load_sketch(p).convert("L"))
    assert out.mean() < 127, "expected white-lines-on-black after loading"


def test_photo_sketch_binarizes(tmp_path):
    p = tmp_path / "photo.png"
    noisy = Image.fromarray(
        (np.random.default_rng(0).normal(230, 8, (300, 200))).clip(0, 255).astype("uint8")
    )
    noisy.save(p)
    out = np.asarray(sketches.load_photo_sketch(p).convert("L"))
    assert set(np.unique(out)).issubset({0, 255}), "output must be pure black and white"


def test_whiten_background_keeps_interior():
    """Flood fill must not eat greys inside the character."""
    img = Image.new("RGB", (100, 100), (200, 200, 200))
    img.paste((120, 120, 120), (30, 30, 70, 70))
    out = np.asarray(sketches.whiten_background(img, threshold=20).convert("L"))
    assert out[0, 0] == 255, "border should be white"
    assert out[50, 50] < 200, "interior grey should survive"
