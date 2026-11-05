"""Turns manga volumes into a kohya LoRA dataset. Driven by scripts/prepare_series.py.

    pipeline  pages out of PDFs/folders, panel scan, cleaned panels + holdout
    panels    page  -> panel boxes, by recursive gutter / border cut
    text      panel -> lettering blocks and oversized SFX
    clean     panel -> the same panel with the lettering erased
    magi      pages -> magi.json, character + text boxes (GPU)
    crops     magi.json -> one crop per character
    tagger    WD14 captions over onnxruntime
    sheets    contact sheets, to check the filters by eye
"""
from .pipeline import extract_pages, scan, survivors, emit, RULES  # noqa: F401
