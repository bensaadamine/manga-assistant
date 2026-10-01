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

On Colab, use `notebooks/00_colab_runner.ipynb` — it clones the repo, installs, and
points the Hugging Face cache at Drive so an 8 GB download survives a disconnect.

## Use

```bash
# generate every pose for a character
python scripts/run_poses.py --config conan.yaml

# the same, with photographed paper sketches as outlines
python scripts/run_poses.py --config conan.yaml --photo-sketches

# baselines for the report
python scripts/run_poses.py --config conan.yaml --baseline b1            # pose only
python scripts/run_poses.py --config conan.yaml --baseline b4 --lora <path>

# ablation figures
python scripts/sweep.py --config conan.yaml --param ip_scale --values 0.4 0.6 0.8
```

## Training a per-artist LoRA

```bash
# 1. Manga109 annotations -> single-character crops, speech bubbles filtered out
python -m manga.preprocess --root data/Manga109 --book DetectiveConan \
    --out data/datasets/conan --trigger aoymstyle

# 2. real tags instead of the placeholder captions
bash scripts/tag_dataset.sh data/datasets/conan aoymstyle

# 3. train  (kohya_ss sd-scripts, ~1-2 h on a free T4)
git clone https://github.com/kohya-ss/sd-scripts
python -m manga.train_lora --config conan.yaml --data data/datasets/conan
```

Plain LoRA, not QLoRA — QLoRA is a quantized-base trick for 7B+ language models, and
SD 1.5's UNet is ~860M parameters, which fits a T4 directly.

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
src/manga/    config, pipeline, prompts, sketches, generate, preprocess, train_lora, evaluate
scripts/      runnable entry points for figures and datasets
notebooks/    thin Colab runner - clone, install, call the modules
tests/        config, prompt and sketch invariants
```
