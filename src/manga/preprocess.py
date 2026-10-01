"""Manga109 -> single-character training crops.

Manga109 ships per-page XML annotations with four box types: frame, body, face, text.
That is what makes this tractable:

  multiple characters per panel -> crop one `body` box at a time
  speech bubbles               -> drop crops that overlap a `text` box
  backgrounds                  -> keep them; a style LoRA learns screentone from them
  waist-up framing             -> fine, a STYLE LoRA does not need full bodies

Usage:
    python -m manga.preprocess --root data/Manga109 --book DetectiveConan \
        --out data/datasets/conan --trigger aoymstyle
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Box:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def w(self) -> int:
        return self.xmax - self.xmin

    @property
    def h(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def overlap(self, other: "Box") -> int:
        dx = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        dy = min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        return dx * dy if dx > 0 and dy > 0 else 0

    def pad(self, ratio: float, limit: tuple[int, int]) -> "Box":
        px, py = int(self.w * ratio), int(self.h * ratio)
        return Box(
            max(0, self.xmin - px),
            max(0, self.ymin - py),
            min(limit[0], self.xmax + px),
            min(limit[1], self.ymax + py),
        )


def _box(elem) -> Box:
    return Box(*(int(elem.get(k)) for k in ("xmin", "ymin", "xmax", "ymax")))


def parse_page(page) -> tuple[list[Box], list[Box]]:
    bodies = [_box(e) for e in page.findall("body")]
    texts = [_box(e) for e in page.findall("text")]
    return bodies, texts


def extract(
    root: Path,
    book: str,
    out_dir: Path,
    trigger: str,
    min_size: int = 256,
    max_text_overlap: float = 0.02,
    pad: float = 0.08,
) -> int:
    """Write one crop + one .txt caption per usable character box. Returns the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(root / "annotations" / f"{book}.xml")
    kept = 0

    for page in tree.getroot().iter("page"):
        index = int(page.get("index"))
        img_path = root / "images" / book / f"{index:03d}.jpg"
        if not img_path.exists():
            continue

        bodies, texts = parse_page(page)
        if not bodies:
            continue

        page_img = Image.open(img_path).convert("L")
        limit = page_img.size

        for i, body in enumerate(bodies):
            if body.w < min_size or body.h < min_size:
                continue  # too small to carry style information
            if body.area and sum(body.overlap(t) for t in texts) / body.area > max_text_overlap:
                continue  # a speech bubble sits on the character

            b = body.pad(pad, limit)
            crop = page_img.crop((b.xmin, b.ymin, b.xmax, b.ymax))
            stem = f"{book}_{index:03d}_{i}"
            crop.save(out_dir / f"{stem}.png")
            # Caption: trigger word first, then a generic tag. Replace the tail with
            # WD14 tagger output once you wire it in - see scripts/tag_dataset.sh.
            (out_dir / f"{stem}.txt").write_text(
                f"{trigger}, monochrome, greyscale, manga, 1character, solo"
            )
            kept += 1

    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="Manga109 root directory")
    ap.add_argument("--book", required=True, help="e.g. DetectiveConan")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trigger", required=True, help="LoRA activation token")
    ap.add_argument("--min-size", type=int, default=256)
    args = ap.parse_args()

    n = extract(args.root, args.book, args.out, args.trigger, min_size=args.min_size)
    print(f"wrote {n} crops to {args.out}")


if __name__ == "__main__":
    main()
