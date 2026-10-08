"""WD14 tagging over onnxruntime, with no sd-scripts and no torch.

kohya's own tagger is the obvious tool, but importing it drags in
`library.dataset` -> diffusers -> torch._dynamo -> optree, and on a machine with
Windows Smart App Control that chain dies on a blocked DLL. None of it is needed
to run a 400MB ONNX classifier: the model wants a square RGB image and gives back
one probability per tag.

Model layout (SmilingWolf's WD14 taggers):
  model.onnx          input NHWC uint8-range float32, BGR
  selected_tags.csv   one row per output column: tag_id, name, category
                      category 9 = rating, 4 = character, everything else general
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RATING, CHARACTER = 9, 4


def download(repo_id: str, cache_dir: str | None = None):
    """Fetch model.onnx and selected_tags.csv from the Hub."""
    from huggingface_hub import hf_hub_download
    model = hf_hub_download(repo_id, "model.onnx", cache_dir=cache_dir)
    tags = hf_hub_download(repo_id, "selected_tags.csv", cache_dir=cache_dir)
    return model, tags


def load_tags(csv_path):
    names, cats = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            cats.append(int(row["category"]))
    return names, np.array(cats)


def make_session(model_path, providers=None):
    import onnxruntime as ort
    if providers is None:
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in avail] or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def input_size(session, default=448):
    shape = session.get_inputs()[0].shape
    for v in shape[1:3]:
        if isinstance(v, int) and v > 0:
            return v
    return default


def fixed_batch(session):
    """The batch size the model insists on, or None when it is dynamic.

    Several WD14 ONNX exports pin the batch dimension to 1; feeding them a
    stack of 8 fails in onnxruntime rather than silently looping.
    """
    n = session.get_inputs()[0].shape[0]
    return n if isinstance(n, int) and n > 0 else None


def prepare(path, size):
    """Square-pad on white, resize, and hand back BGR floats in 0-255."""
    from PIL import Image
    im = Image.open(path)
    if im.mode != "RGB":
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA"))
        im = bg.convert("RGB")
    w, h = im.size
    s = max(w, h)
    sq = Image.new("RGB", (s, s), (255, 255, 255))
    sq.paste(im, ((s - w) // 2, (s - h) // 2))
    if s != size:
        sq = sq.resize((size, size), Image.BICUBIC)
    a = np.asarray(sq, dtype=np.float32)
    return a[:, :, ::-1]                      # RGB -> BGR


def tag_images(paths, model_path, tags_csv, batch_size=8,
               general_threshold=0.35, character_threshold=0.85,
               remove_underscore=True, progress=None):
    """Yield (path, [tags]) for each image."""
    names, cats = load_tags(tags_csv)
    session = make_session(model_path)
    size = input_size(session)
    in_name = session.get_inputs()[0].name
    pinned = fixed_batch(session)
    if pinned is not None and pinned != batch_size:
        batch_size = pinned

    keep_general = cats != RATING
    thr = np.where(cats == CHARACTER, character_threshold, general_threshold)

    paths = list(paths)
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        batch = np.stack([prepare(p, size) for p in chunk])
        probs = session.run(None, {in_name: batch})[0]
        for p, row in zip(chunk, probs):
            if len(row) != len(names):        # a model whose csv does not line up
                raise RuntimeError(
                    f"model returned {len(row)} scores but selected_tags.csv has "
                    f"{len(names)} rows — mismatched tagger files")
            picked = [names[j] for j in np.flatnonzero((row >= thr) & keep_general)]
            if remove_underscore:
                picked = [t.replace("_", " ") if len(t) > 3 else t for t in picked]
            yield Path(p), picked
        if progress:
            progress(min(i + batch_size, len(paths)), len(paths))
