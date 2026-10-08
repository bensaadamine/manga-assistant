"""Unit tests for the panel/lettering pipeline on synthetic pages.

Synthetic rather than real pages so the tests run without the PDFs present.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manga.dataset import clean, panels, text  # noqa: E402

PAGE_H = 1200


def blank(h=PAGE_H, w=850):
    return np.full((h, w), 255, np.uint8)


def box(img, x0, y0, x1, y1, t=4):
    img[y0:y0 + t, x0:x1] = 0
    img[y1 - t:y1, x0:x1] = 0
    img[y0:y1, x0:x0 + t] = 0
    img[y0:y1, x1 - t:x1] = 0


def scribble(img, x0, y0, x1, y1, step=9):
    """Something that reads as artwork: enough ink, no letter-sized blobs."""
    img[y0:y1:step, x0:x1] = 40


def letters(img, x0, y0, cols, rows, h=22, w=14, gap=9):
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * (w + gap)
            y = y0 + r * (h + gap)
            img[y:y + h, x:x + w] = 0
            img[y + 4:y + h - 4, x + 4:x + w - 4] = 255     # hollow, like a glyph


def test_splits_a_two_by_two_grid():
    p = blank()
    for (a, b) in ((40, 40), (450, 40), (40, 620), (450, 620)):
        box(p, a, b, a + 360, b + 540)
        scribble(p, a + 20, b + 20, a + 340, b + 520)
    found = [b for b in panels.detect(p) if min(b[2] - b[0], b[3] - b[1]) > 200]
    assert len(found) == 4


def test_does_not_shatter_a_sparse_borderless_panel():
    """Open white with two small figures must not become two panels."""
    p = blank()
    scribble(p, 60, 60, 160, 200, step=3)
    scribble(p, 600, 700, 700, 840, step=3)
    big = [b for b in panels.detect(p) if (b[2] - b[0]) * (b[3] - b[1]) > 0.2 * p.size]
    assert len(big) <= 1


def test_finds_a_block_of_lettering():
    p = blank(400, 400)
    letters(p, 60, 60, cols=5, rows=3)
    blocks = text.text_blocks(p, PAGE_H)
    assert len(blocks) == 1
    x0, y0, x1, y1 = blocks[0]
    assert x0 <= 60 and y0 <= 60 and x1 >= 170 and y1 >= 130


def test_artwork_alone_is_not_lettering():
    p = blank(400, 400)
    scribble(p, 40, 40, 360, 360, step=5)
    assert text.burden(p, PAGE_H)[0] < 0.05


def test_oversized_type_is_flagged_separately():
    p = blank(500, 500)
    letters(p, 40, 40, cols=4, rows=1, h=70, w=46, gap=18)
    assert text.big_lettering(p, PAGE_H)[0] > 0.02
    p2 = blank(500, 500)
    letters(p2, 40, 40, cols=4, rows=1)
    assert text.big_lettering(p2, PAGE_H)[0] == 0.0


def test_lettering_on_pale_ground_is_erased():
    p = blank(400, 400)
    letters(p, 80, 80, cols=5, rows=3)
    before = (p < 200).sum()
    out, ok = clean.clean(p, PAGE_H)
    assert ok
    assert (out < 200).sum() < 0.1 * before


def test_lettering_over_artwork_disqualifies_the_panel():
    """Type set straight on inked artwork has no pale ground to erase.

    Each glyph carries only a hairline halo, enough to keep it separable but
    nowhere near enough pale ground to paint the block out without gouging
    the drawing underneath.
    """
    p = blank(400, 400)
    p[40:360, 40:360] = 70                      # inked artwork
    for r in range(3):
        for c in range(5):
            x, y = 80 + c * 23, 80 + r * 31
            p[y - 2:y + 24, x - 2:x + 16] = 235
            p[y:y + 22, x:x + 14] = 0
            p[y + 4:y + 18, x + 4:x + 10] = 235
    assert len(text.text_blocks(p, PAGE_H)) == 1
    _, ok = clean.clean(p, PAGE_H)
    assert not ok


@pytest.mark.parametrize("shape,expect", [((900, 600), (768, 512)),
                                          ((600, 900), (512, 768)),
                                          ((400, 4000), (512, 1024))])
def test_fit_scales_short_side_and_caps_long_side(shape, expect):
    out = clean.fit(np.full(shape, 255, np.uint8))
    assert out.shape == expect
