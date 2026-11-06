# Manga Panel Assistant

An assistant for mangaka. The artist supplies a **character reference** and a **rough
pose outline**; the system draws that character in that pose, in that artist's style.

Not a universal model. A **reproducible method for adapting the system to one artist**,
proven across three visually different series.

**Boundary:** the system draws the body, the artist draws the head — face *and* hair.
The outline supplies head shape, size and orientation so the artist's drawing lands
in the right place.

---

## Status

The proof of concept is complete and **passed** — see `POC_VERDICT.md` in the project
docs. 11 of 15 planned cases run, all 11 passed on the first attempt, against a bar of
≥60% within three attempts. Parameters tuned once on character 1 transferred to
characters 2 and 3 untouched.

Phase 2 is per-artist LoRA training, which is what this repository is scaffolded for.

## How it works

```
   pose outline ───► ControlNet lineart ──┐
                                           ├──► SD 1.5 UNet ──► image
   text prompt  ─────────────────────────┤
                                           │
   character ref ──► IP-Adapter Plus ─────┘

   per-artist LoRA ──► injected into the UNet's attention layers  (phase 2)
```

| Component | Supplies | Trained by us |
|---|---|---|
| `gsdf/Counterfeit-V2.5` (SD 1.5) | the ability to draw | no |
| CLIP skip 2 + `compel` | weighted prompt handling | no |
| ControlNet lineart-anime | pose, from the outline | no |
| IP-Adapter Plus | identity, from the reference | no |
| **Style LoRA** | **one artist's line work and screentone** | **yes** |

## Install

```bash
git clone <your-repo-url> && cd manga-assistant
pip install -e .
```

No local GPU: `notebooks/02_prepare_series_kaggle.ipynb` builds a dataset and
`notebooks/01_train_lora_kaggle.ipynb` trains the LoRA, both on Kaggle.

## Run it end to end (one series)

```bash
# 1. dataset: pages -> panels -> Magi -> crops -> WD14 tags -> filter
#    (Magi runs in its own venv, see requirements-magi.txt)
python scripts/prepare_series.py --config bungo.yaml

# 2. train the style LoRA (kohya sd-scripts)
python scripts/train_lora.py --config bungo.yaml --scripts <path/to/sd-scripts>

# 3. pick a checkpoint: same poses, same seed, one column per LoRA
python scripts/compare_loras.py --config bungo_eval.yaml --poses akiko2.png atsushi2.png \
    --loras none models/loras/bungo/bungo_style-step00004800.safetensors

# 4. measure B3 vs B4
python scripts/evaluate.py --config bungo_eval.yaml --baseline b3
python scripts/evaluate.py --config bungo_eval.yaml --baseline b4 --lora models/loras/bungo/bungo_style_v1.safetensors

# 5. draw every pose in the config
python scripts/run_poses.py --config bungo_eval.yaml --baseline b4 --lora models/loras/bungo/bungo_style_v1.safetensors
```

Steps 1-2 run once per artist; step 5 is the everyday one. Optional research tools:
`scripts/sweep.py` (one parameter) and `scripts/grid_sweep.py` (ip_scale x cn_scale).

Plain LoRA, not QLoRA — QLoRA is a quantized-base trick for 7B+ language models, and
SD 1.5's UNet is ~860M parameters, which fits a T4 directly.

## The app (same steps, no terminal)

```bash
pip install -e .
python scripts/serve.py          # UI -> http://localhost:8000/ui   API -> http://localhost:8000/docs
```

| Tab | Does | Calls |
|---|---|---|
| 1 Volumes | drop PDFs / zips, name the series and trigger | writes `configs/<series>.yaml` |
| 2 Dataset | build the dataset, live log | `scripts/prepare_series.py` |
| 3 Train | train the LoRA (steps ~2.5 epochs), version with DVC | `scripts/train_lora.py`, `scripts/dvc_version.py` |
| 4 Draw | reference + outline + checkpoint; compare all checkpoints; promote one | `manga.service` |

- One job at a time (Magi and kohya each need the whole GPU); drawing models are freed while it runs.
  Logs stay in `outputs/jobs/`.
- References and outlines are kept in `data/library/` for reuse, 50 of each
  (`library.max_items` in `base.yaml`), least recently used deleted first.
- Every image is saved in `outputs/app/<date>/` with a `.json`: git commit, LoRA, seed, prompts, parameters.
- **Promote** writes the chosen checkpoint to `configs/registry.yaml` (git). The API's
  `POST /generate` with only `series=` serves that one.

```bash
curl -F series=bungo -F description="1boy, solo, white shirt, black trousers" \
     -F reference=@atsushi.png -F outline=@pose.png http://localhost:8000/generate -o out.png
```

Docker (serving only; training and Magi stay outside):

```bash
docker build -t manga-assistant .
docker run --gpus all -p 8000:8000 -v %cd%/models:/app/models -v %cd%/data:/app/data ^
  -v %cd%/outputs:/app/outputs -v %USERPROFILE%/.cache/huggingface:/cache/hf manga-assistant
```

## Configuration

Everything lives in `configs/`. No magic numbers in code.

- `base.yaml` — models, the three frozen generation parameters, style and negative prompts
- `<artist>.yaml` — `extends: base.yaml`, adds the character, the poses, the LoRA hyperparameters

The three numbers that matter, frozen at the end of the POC:

```yaml
cn_scale: 0.50   # outline strength
cn_end:   0.70   # ControlNet steers only the first 70% of denoising steps
ip_scale: 0.80   # reference strength
```

## Evaluation

The headline figure is **B3 → B4**: ControlNet + IP-Adapter with and without the
per-artist LoRA, on the same cases.

| # | Setup | Purpose |
|---|---|---|
| B0 | prompt only | floor |
| B1 | ControlNet only | pose without identity |
| B2 | IP-Adapter only | identity without pose |
| B3 | ControlNet + IP-Adapter | the POC — the number to beat |
| B4 | B3 + per-artist LoRA | the proposed system |

`manga.evaluate` provides DINOv2 identity similarity, FID against held-out panels,
and ink statistics that catch "looks like anime but is not actually inked."

## Versioning

| What | Tool |
|---|---|
| code | git |
| datasets, LoRA weights | DVC, Google Drive remote |
| runs, metrics, sample grids | Weights & Biases |
| which LoRA is served per series | `configs/registry.yaml` (git) |
| tests on every push | GitHub Actions (`.github/workflows/ci.yml`) |
| serving environment | Docker (`Dockerfile`) |

`data/` and `outputs/` are gitignored on purpose — they are DVC's.

## Findings worth not rediscovering

1. **Never call `enable_attention_slicing()`** — it breaks IP-Adapter loading. Use
   `pipe.vae.enable_slicing()`.
2. **Once IP-Adapter is loaded the UNet demands an image on every call**, even pure
   text-to-image. Handled by `_ip_kwargs`.
3. **IP-Adapter copies everything in the reference** — background, props, ink splatter.
   Always whiten the background, and negative-prompt any prop the outline lacks.
4. **The outline must describe the *clothed* silhouette.** With a nude mannequin
   outline, ControlNet's body edge beats IP-Adapter's fabric: coat tails go
   transparent, knee socks vanish.
5. **ControlNet traces every line literally.** Joint circles become rips in the jeans;
   a line across a head oval becomes a blindfold. Contour lines only.
6. **No color words in a monochrome prompt.** Describe shape.
7. **Prompt weights do nothing without `compel`.**

## Layout

```
configs/      base.yaml + one per artist
src/manga/    config, pipeline, prompts, sketches, generate, train_lora, evaluate, dataset/
scripts/      runnable entry points for figures and datasets
notebooks/    Kaggle runners for dataset prep and training
tests/        config, dataset and sketch invariants
```
