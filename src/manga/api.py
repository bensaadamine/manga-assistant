"""REST API + web UI in one process, so the models sit in VRAM once.

    GET  /health     models loaded? job running?
    GET  /series     series, trigger words, checkpoints, promoted LoRA
    POST /generate   reference + outline (+ series) -> PNG (n images side by side)
    /ui              the Gradio app (src/manga/ui.py)
    /docs            interactive API docs (FastAPI)

/generate serves the series' PROMOTED LoRA from configs/registry.yaml unless the
request names a checkpoint, so "what is in production" is one file in git.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image

from . import library, registry
from .config import ROOT
from .jobs import Busy, JobRunner
from .service import ModelService


def _strip(images: list[Image.Image]) -> bytes:
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h), "white")
    for i, im in enumerate(images):
        canvas.paste(im.convert("RGB"), (i * w, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def create_app(service: ModelService | None = None, jobs: JobRunner | None = None,
               root: Path = ROOT, with_ui: bool = True) -> FastAPI:
    root = Path(root)
    jobs = jobs or JobRunner(root)
    service = service or ModelService(root=root, jobs=jobs)
    app = FastAPI(title="Manga Panel Assistant", version="0.2.0")

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/ui" if with_ui else "/docs")

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": service.loaded(), "job": jobs.status()}

    @app.get("/series")
    def series():
        return [{"name": n, **info, "checkpoints": registry.checkpoints(n, root),
                 "promoted": registry.promoted(n, root)}
                for n, info in registry.series(root).items()]

    @app.post("/generate", responses={200: {"content": {"image/png": {}}}})
    def generate(reference: UploadFile = File(...), outline: UploadFile = File(...),
                 description: str = Form(...), series: str | None = Form(None),
                 lora: str | None = Form(None, description="checkpoint path; overrides the registry"),
                 lora_scale: float | None = Form(None), pose: str = Form(""),
                 negative_props: str = Form(""), outline_kind: str = Form("digital"),
                 n: int = Form(1, ge=1, le=4), seed: int | None = Form(None)):
        info = registry.series(root).get(series) if series else None
        if series and info is None:
            raise HTTPException(404, f"unknown series '{series}'")
        if lora:
            try:
                registry.safe_lora_path(lora, root)
            except (ValueError, FileNotFoundError) as e:
                raise HTTPException(400, str(e))
        elif series:
            p = registry.promoted(series, root)
            if not p:
                raise HTTPException(409, f"no promoted LoRA for '{series}' - promote one in the UI")
            lora, lora_scale = p["lora"], lora_scale if lora_scale is not None else p["scale"]
        if outline_kind not in ("digital", "photo"):
            raise HTTPException(422, "outline_kind must be 'digital' or 'photo'")

        with tempfile.TemporaryDirectory() as tmp:
            saved = []
            for up, kind in ((reference, "refs"), (outline, "outlines")):
                tmp_path = Path(tmp) / f"{kind}.png"
                tmp_path.write_bytes(up.file.read())
                try:
                    saved.append(library.add(tmp_path, kind, root))
                except Exception:
                    raise HTTPException(422, f"{kind[:-1]}: not a readable image")
        try:
            images, record = service.generate(
                reference=saved[0], outline=saved[1], description=description, pose=pose,
                negative_props=negative_props, outline_kind=outline_kind, lora=lora,
                lora_scale=1.0 if lora_scale is None else lora_scale,
                trigger=info["trigger"] if info else "", n=n, seed=seed)
        except Busy as e:
            raise HTTPException(409, str(e))
        return Response(_strip(images), media_type="image/png",
                        headers={"X-Lora": str(record["lora"]),
                                 "X-Seed": str(record["params"]["seed"]),
                                 "X-Record": record.get("files", [""])[0].replace(".png", ".json")})

    if with_ui:
        import gradio as gr

        from . import ui
        app = gr.mount_gradio_app(app, ui.build(service, jobs, root), path="/ui",
                                  allowed_paths=[str(root / "data" / "library"),
                                                 str(root / "outputs")])
    return app
