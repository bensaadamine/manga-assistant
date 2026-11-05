#!/usr/bin/env python
"""Step 2 - trains the style LoRA: thin CLI around manga.train_lora (kohya sd-scripts), no PYTHONPATH needed.

    python scripts/train_lora.py --config spyxfamily.yaml --dry-run
    python scripts/train_lora.py --config spyxfamily.yaml --scripts ./sd-scripts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manga.train_lora import main  # noqa: E402

if __name__ == "__main__":
    main()
