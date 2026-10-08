"""Turn bought manga PDFs into a kohya LoRA dataset.

    panels    page  -> panel boxes, by recursive gutter / border cut
    text      panel -> lettering blocks and oversized SFX
    clean     panel -> the same panel with the lettering erased
    pipeline  the four stages, driven from configs/<series>.yaml
"""
from .pipeline import extract_pages, scan, survivors, emit, RULES  # noqa: F401
