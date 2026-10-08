#!/usr/bin/env python
"""Replace the placeholder captions with WD14 tags, trigger word first.

    python scripts/tag_dataset.py data/datasets/spyxfamily/10_sxfstyle sxfstyle

Runs the tagger directly on onnxruntime. kohya's own tagger does the same job but
imports `library.dataset` -> diffusers -> torch, which is a lot of machinery for a
classifier and, on Windows with Smart App Control, a blocked DLL. Needs only
onnxruntime, huggingface_hub, pillow and numpy.

Re-running is safe: the trigger word is not stacked twice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga.dataset import tagger  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".webp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("trigger")
    ap.add_argument("--repo-id", default="SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--character-threshold", type=float, default=0.85)
    ap.add_argument("--keep-underscores", action="store_true")
    ap.add_argument("--model", help="local model.onnx, skips the download")
    ap.add_argument("--tags", help="local selected_tags.csv, skips the download")
    a = ap.parse_args()

    data = Path(a.data_dir).resolve()
    if not data.is_dir():
        raise SystemExit(f"{data} is not a directory")
    images = sorted(p for p in data.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"no images in {data} — run scripts/build_dataset.py first")

    if a.model and a.tags:
        model, tags = a.model, a.tags
    else:
        print(f"fetching {a.repo_id} (~400MB on first run)")
        try:
            model, tags = tagger.download(a.repo_id)
        except ImportError:
            raise SystemExit("pip install huggingface_hub onnxruntime pillow")

    print(f"tagging {len(images)} images")
    n = 0
    for path, tags_found in tagger.tag_images(
            images, model, tags,
            batch_size=a.batch_size,
            general_threshold=a.threshold,
            character_threshold=a.character_threshold,
            remove_underscore=not a.keep_underscores,
            progress=lambda d, t: print(f"  {d}/{t}", flush=True)):
        tags_found = [t for t in tags_found if t != a.trigger]
        path.with_suffix(".txt").write_text(
            ", ".join([a.trigger] + tags_found) + "\n", encoding="utf-8")
        n += 1

    print(f"wrote {n} captions, each starting with '{a.trigger}'")
    print("sample:", images[0].with_suffix(".txt").read_text(encoding="utf-8")[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
