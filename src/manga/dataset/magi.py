"""Pages -> magi.json: character and text boxes per page. Needs a GPU (Kaggle).

Everything after this stage is CPU: crops.py re-cuts from the JSON in minutes,
so the JSON is the expensive artefact and is written as it goes. A Kaggle
session that dies at page 1800 resumes at page 1800.

Keys are "<vol>/<page file>": two volumes both have a p-0005.jpg, and keying by
file name alone once silently overwrote 152 pages.
"""
from __future__ import annotations

import json
from pathlib import Path

MODEL = "ragavsachdeva/magiv2"
SAVE_EVERY = 25


def key(vol: str, path: str) -> str:
    return f"{vol}/{Path(path).name}"


def detect(pages: dict, out_json: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise SystemExit("Magi needs a GPU. Run notebooks/02_prepare_series_kaggle.ipynb, "
                         "or copy an existing magi.json into the pages folder.")

    done = json.loads(out_json.read_text()) if out_json.exists() else {}
    todo = [(v, p) for v, ps in pages.items() for p in ps if key(v, p) not in done]
    print(f"magi: {len(done)} pages cached, {len(todo)} to detect")
    if not todo:
        return done

    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).cuda().eval()
    for n, (vol, path) in enumerate(todo, 1):
        rgb = np.array(Image.open(path).convert("L").convert("RGB"))
        with torch.no_grad():
            r = model.predict_detections_and_associations([rgb])[0]
        done[key(vol, path)] = {
            "characters": [[float(x) for x in b] for b in r["characters"]],
            "texts": [[float(x) for x in b] for b in r["texts"]],
        }
        if n % SAVE_EVERY == 0 or n == len(todo):
            out_json.write_text(json.dumps(done))
            print(f"  magi {n}/{len(todo)}", flush=True)
    return done
