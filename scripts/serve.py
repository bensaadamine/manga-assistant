#!/usr/bin/env python
"""Step 6 - the app: web UI at http://localhost:8000/ui, REST API docs at /docs.

    python scripts/serve.py
    python scripts/serve.py --port 7860 --host 0.0.0.0     # reachable from the LAN / Docker
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from manga.api import create_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-ui", action="store_true", help="API only")
    a = ap.parse_args()
    print(f"UI  -> http://localhost:{a.port}/ui\nAPI -> http://localhost:{a.port}/docs")
    uvicorn.run(create_app(with_ui=not a.no_ui), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
