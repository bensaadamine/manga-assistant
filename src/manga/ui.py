"""The web UI: volumes in -> dataset -> LoRA -> drawing, without touching a terminal.

    1 Volumes   drop PDFs / zips, name the series      -> configs/<series>.yaml
    2 Dataset   scripts/prepare_series.py, live log     -> data/datasets/<series>/
    3 Train     scripts/train_lora.py (kohya), live log -> models/loras/<series>/*.safetensors
    4 Draw      reference + outline + checkpoint         -> outputs/app/<date>/

Every button calls the same scripts as the command line, so the UI adds no second
way of doing things: it fills in the flags. Served by scripts/serve.py at /ui.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

from . import library, registry
from . import series as series_mod
from .config import ROOT, load
from .jobs import Busy, JobRunner
from .service import ModelService

STAGES = ["(resume)", "pages", "panels", "magi", "crops", "tag", "filter"]
NO_LORA = "none (B3 baseline)"


def _report(name: str, root: Path) -> dict:
    info = registry.series(root).get(name)
    if not info:
        return {}
    cfg = load(str(root / "configs" / info["config"]))
    p = root / f"{cfg.paths.dataset}_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build(service: ModelService, jobs: JobRunner, root: Path = ROOT) -> gr.Blocks:
    root = Path(root)
    lib_max = int(service.cfg.get("library", {}).get("max_items", library.DEFAULT_MAX))

    # ------------------------------------------------------------ helpers
    def series_names():
        return sorted(registry.series(root))

    def ckpt_choices(name):
        return [NO_LORA] + (registry.checkpoints(name, root) if name else [])

    def default_ckpt(name):
        p = registry.promoted(name, root) if name else None
        cks = ckpt_choices(name)
        return p["lora"] if p and p["lora"] in cks else (cks[1] if len(cks) > 1 else NO_LORA)

    def header():
        models = "loaded" if service.loaded() else "not loaded (first drawing takes ~1 min)"
        return f"**Job:** {jobs.status()}  ·  **Models:** {models}"

    def on_series(name):
        rep = _report(name, root)
        steps, every = series_mod.suggested_steps(rep.get("images_per_epoch", 4700))
        p = registry.promoted(name, root) if name else None
        return (gr.Dropdown(choices=ckpt_choices(name), value=default_ckpt(name)),
                rep or {"note": "no dataset yet - run tab 2"},
                steps, every,
                f"Promoted: `{p['lora']}` (scale {p['scale']})" if p else "Promoted: none")

    def config_of(name):
        if not name:
            raise gr.Error("pick a series first")
        return registry.series(root)[name]["config"]

    def start_job(label, cmd):
        try:
            jobs.start(label, cmd)
        except Busy as e:
            raise gr.Error(str(e))
        return header()

    def poll():
        log = jobs.tail()
        return header(), log, log

    # ---------------------------------------------------------------- tabs
    with gr.Blocks(title="Manga Panel Assistant") as demo:
        gr.Markdown("# Manga Panel Assistant\nOne artist's volumes in, that artist's style out.")
        with gr.Row():
            names = series_names()
            series_dd = gr.Dropdown(names, value=names[0] if names else None,
                                    label="Series", scale=3)
            refresh_btn = gr.Button("↻ Refresh", scale=0)
        status_md = gr.Markdown(header())

        # 1 ------------------------------------------------------- volumes
        with gr.Tab("1 · Volumes"):
            gr.Markdown("Drop the series' volumes: one **PDF** or one **zip of page images** "
                        "per volume (chapter folders inside a zip are fine, read in natural order).")
            with gr.Row():
                new_name = gr.Textbox(label="Series name", placeholder="vagabond")
                new_trigger = gr.Textbox(label="Trigger word", placeholder="vgbstyle",
                                         info="One made-up word; the LoRA learns the style under it.")
                head_drop = gr.Number(0, precision=0, label="Skip first pages / volume",
                                      info="cover, colour plates")
                tail_drop = gr.Number(0, precision=0, label="Skip last pages / volume",
                                      info="afterword, ads")
            vols = gr.File(file_count="multiple", file_types=[".pdf", ".zip"], label="Volumes")
            create_btn = gr.Button("Create series", variant="primary")
            create_msg = gr.Markdown()

        # 2 ------------------------------------------------------- dataset
        with gr.Tab("2 · Dataset"):
            gr.Markdown("Pages → panels → Magi character crops → WD14 captions → tag filter. "
                        "Each stage is skipped if already done. Magi uses the GPU.")
            with gr.Row():
                limit = gr.Number(0, precision=0, label="Smoke test: first N pages / volume (0 = all)")
                redo = gr.Dropdown(STAGES, value=STAGES[0], label="Redo from stage")
            with gr.Row():
                prep_btn = gr.Button("Build dataset", variant="primary")
                stop1 = gr.Button("Stop job", variant="stop")
            log1 = gr.Textbox(label="Log", lines=18, max_lines=18, autoscroll=True)
            report = gr.JSON(label="Dataset report")

        # 3 --------------------------------------------------------- train
        with gr.Tab("3 · Train"):
            gr.Markdown("kohya sd-scripts LoRA. Steps default to ~2.5 epochs of the dataset, "
                        "with 10 checkpoints to choose from in tab 4.")
            with gr.Row():
                steps = gr.Number(12000, precision=0, label="Total steps")
                every = gr.Number(1200, precision=0, label="Save a checkpoint every")
                sd_scripts = gr.Textbox(str((root.parent / "sd-scripts")), label="sd-scripts folder")
                use_wandb = gr.Checkbox(True, label="Log to W&B")
            with gr.Row():
                train_btn = gr.Button("Train LoRA", variant="primary")
                stop2 = gr.Button("Stop job", variant="stop")
                dvc_btn = gr.Button("Version dataset + LoRAs with DVC")
            log2 = gr.Textbox(label="Log", lines=18, max_lines=18, autoscroll=True)

        # 4 ---------------------------------------------------------- draw
        with gr.Tab("4 · Draw"):
            ref_paths = gr.State([])
            out_paths = gr.State([])
            with gr.Row():
                with gr.Column():
                    ref_img = gr.Image(type="filepath", label="Character reference",
                                       sources=["upload", "clipboard"], height=300)
                    ref_gal = gr.Gallery(label=f"Saved references (last {lib_max})",
                                         columns=6, height=160, allow_preview=False,
                                         object_fit="contain")
                    clean_ref = gr.Checkbox(True, label="Whiten background + greyscale the reference")
                with gr.Column():
                    out_img = gr.Image(type="filepath", label="Pose outline",
                                       sources=["upload", "clipboard"], height=300)
                    out_gal = gr.Gallery(label=f"Saved outlines (last {lib_max})",
                                         columns=6, height=160, allow_preview=False,
                                         object_fit="contain")
                    kind = gr.Radio(["digital", "photo"], value="digital",
                                    label="Outline is", info="photo = phone picture of a paper sketch")
            with gr.Row():
                ckpt = gr.Dropdown([NO_LORA], value=NO_LORA, label="LoRA checkpoint", scale=3)
                lora_scale = gr.Slider(0, 1.5, 1.0, step=0.05, label="LoRA strength")
            promoted_md = gr.Markdown()
            description = gr.Textbox(label="Character (clothes, body - not the head)",
                                     placeholder="1boy, solo, white collared shirt, black necktie, "
                                                 "suspenders, black trousers, black boots")
            with gr.Row():
                pose = gr.Textbox(label="Pose (optional)", placeholder="standing, arms crossed")
                neg_props = gr.Textbox(label="Never draw (optional)", placeholder="bag, weapon")
            with gr.Accordion("Advanced (frozen POC values)", open=False):
                g = service.cfg.generation
                with gr.Row():
                    seed = gr.Number(g.seed, precision=0, label="Seed")
                    n = gr.Slider(1, 4, 2, step=1, label="Images")
                    ip = gr.Slider(0, 1, g.ip_scale, step=0.05, label="ip_scale (reference)")
                    cn = gr.Slider(0, 1.5, g.cn_scale, step=0.05, label="cn_scale (outline)")
            with gr.Row():
                draw_btn = gr.Button("Draw", variant="primary")
                cmp_btn = gr.Button("Compare all checkpoints (1 image each)")
                promote_btn = gr.Button("Promote this checkpoint")
            result = gr.Gallery(label="Result", columns=4, height=520, object_fit="contain")
            record = gr.JSON(label="Run record (saved next to each image)")

        # ---------------------------------------------------------- events
        def refresh_lib():
            r, o = library.items("refs", root), library.items("outlines", root)
            return [str(p) for p in r], [str(p) for p in r], [str(p) for p in o], [str(p) for p in o]

        def pick(paths, evt: gr.SelectData):
            p = paths[evt.index]
            library.touch(p)
            return p

        def create(name, trig, hd, td, files):
            try:
                p = series_mod.create(name.strip().lower(), trig.strip().lower(),
                                      files or [], hd or 0, td or 0, root)
            except (ValueError, FileExistsError) as e:
                raise gr.Error(str(e))
            names = series_names()
            return (gr.Dropdown(choices=names, value=name.strip().lower()),
                    f"Created `{p.relative_to(root).as_posix()}`. Next: tab 2.")

        def prepare(name, lim, stage):
            cmd = [sys.executable, "scripts/prepare_series.py", "--config", config_of(name)]
            if lim:
                cmd += ["--limit", int(lim)]
            if stage != STAGES[0]:
                cmd += ["--from", stage]
            return start_job(f"prepare_{name}", cmd)

        def train(name, st, ev, sd, wb):
            if not (Path(sd) / "train_network.py").exists():
                raise gr.Error(f"train_network.py not found in {sd}")
            cmd = [sys.executable, "scripts/train_lora.py", "--config", config_of(name),
                   "--scripts", sd, "--steps", int(st), "--save-every", int(ev)]
            return start_job(f"train_{name}", cmd + (["--wandb"] if wb else []))

        def version(name):
            return start_job(f"dvc_{name}", [sys.executable, "scripts/dvc_version.py",
                                             "--config", config_of(name)])

        def stop():
            jobs.stop()
            return header()

        def inputs_ok(ref, out):
            if not ref or not out:
                raise gr.Error("need a reference and an outline")
            return (library.add(ref, "refs", root, lib_max),
                    library.add(out, "outlines", root, lib_max))

        def kwargs(name, ref, out, clean, k, desc, ps, neg, sd, ipv, cnv):
            r, o = inputs_ok(ref, out)
            return dict(reference=r, outline=o, clean_reference=clean, outline_kind=k,
                        description=desc or "", pose=ps or "", negative_props=neg or "",
                        trigger=registry.series(root).get(name, {}).get("trigger", ""),
                        seed=int(sd), ip_scale=ipv, cn_scale=cnv)

        common = [series_dd, ref_img, out_img, clean_ref, kind, description, pose, neg_props,
                  seed, ip, cn]

        def draw(name, ref, out, clean, k, desc, ps, neg, sd, ipv, cnv, ck, ls, num):
            kw = kwargs(name, ref, out, clean, k, desc, ps, neg, sd, ipv, cnv)
            try:
                imgs, rec = service.generate(lora=None if ck == NO_LORA else ck,
                                             lora_scale=ls, n=int(num), **kw)
            except Busy as e:
                raise gr.Error(str(e))
            return ([(im, f"seed {im.info.get('seed')}") for im in imgs], rec, header(),
                    *refresh_lib())

        def compare(name, ref, out, clean, k, desc, ps, neg, sd, ipv, cnv, ls):
            kw = kwargs(name, ref, out, clean, k, desc, ps, neg, sd, ipv, cnv)
            cks = [None] + sorted(registry.checkpoints(name, root))
            try:
                pairs = service.compare(cks, lora_scale=ls, **kw)
            except Busy as e:
                raise gr.Error(str(e))
            return pairs, {"compared": [c or "none" for c in cks], "seed": int(sd)}, header(), \
                *refresh_lib()

        def promote(name, ck, ls):
            if not name or ck == NO_LORA:
                raise gr.Error("pick a series and a checkpoint")
            p = registry.promote(name, ck, ls, root)
            return f"Promoted: `{p['lora']}` (scale {p['scale']}) - commit configs/registry.yaml"

        lib_outputs = [ref_paths, ref_gal, out_paths, out_gal]
        series_out = [ckpt, report, steps, every, promoted_md]

        demo.load(on_series, series_dd, series_out).then(refresh_lib, None, lib_outputs)
        series_dd.change(on_series, series_dd, series_out)
        refresh_btn.click(lambda: gr.Dropdown(choices=series_names()), None, series_dd) \
            .then(on_series, series_dd, series_out).then(refresh_lib, None, lib_outputs)
        create_btn.click(create, [new_name, new_trigger, head_drop, tail_drop, vols],
                         [series_dd, create_msg])
        prep_btn.click(prepare, [series_dd, limit, redo], status_md)
        train_btn.click(train, [series_dd, steps, every, sd_scripts, use_wandb], status_md)
        dvc_btn.click(version, series_dd, status_md)
        stop1.click(stop, None, status_md)
        stop2.click(stop, None, status_md)
        ref_gal.select(pick, ref_paths, ref_img)
        out_gal.select(pick, out_paths, out_img)
        draw_btn.click(draw, common + [ckpt, lora_scale, n],
                       [result, record, status_md] + lib_outputs)
        cmp_btn.click(compare, common + [lora_scale], [result, record, status_md] + lib_outputs)
        promote_btn.click(promote, [series_dd, ckpt, lora_scale], promoted_md)
        gr.Timer(2.0).tick(poll, None, [status_md, log1, log2])

    return demo
