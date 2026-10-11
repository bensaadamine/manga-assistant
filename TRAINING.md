# LoRA #1 — Spy × Family style
 
Status: **set up, not yet run.** Dataset ready, config frozen, runner tested dry.
 
Phase 2 step C: validate the chain, not the quality. The point of this first run is
that data → kohya → a `.safetensors` that loads into the POC pipeline works end to
end. Whether the style gap actually closes is step D's question (B3 vs B4).
 
---
 
## Where it runs
 
**Kaggle.** Windows Smart App Control blocks `optree/_C.pyd`, which every sd-scripts
entry point reaches through `library.dataset` → diffusers → `torch._dynamo`. Kaggle
was already the reference environment per CLAUDE.md, so this changes nothing about
the plan — it just removes a question.
 
`notebooks/01_train_lora_kaggle.ipynb` is the runner: clone, install, dry-run, train.
No logic in the notebook; it calls `scripts/train_lora.py`, which builds the command
from `configs/spyxfamily.yaml`.
 
Settings → Accelerator **GPU P100**, Internet **On**. Upload
`data/datasets/spyxfamily/` (the folder *containing* `10_sxfstyle/`) as a Kaggle
Dataset called `spyxfamily-panels`.
 
## The parameters
 
```yaml
lora:
  rank: 32              # usual floor for a style
  alpha: 16             # alpha/rank = 0.5
  learning_rate: 1.0e-4
  text_encoder_lr: 5.0e-5   # half the UNet's: the text side overfits faster
  lr_scheduler: cosine_with_restarts
  warmup_steps: 100
  max_train_steps: 1500 # ~5.5 epochs over 270 images at batch 1
  batch_size: 1
  resolution: 512
  max_bucket_reso: 1024
  optimizer: AdamW8bit
  attention: sdpa       # RTX 50 is sm_120: no xformers wheels exist
```
 
Three flags that are not defaults and matter:
 
- **`--shuffle_caption --keep_tokens 1`** — the trigger stays the first token while
  everything after it shuffles, so the LoRA binds to `sxfstyle` rather than to one
  fixed tag order.
- **`--bucket_no_upscale`** — the dataset's `min_side` filter already guarantees every
  panel is downscaled into its bucket. Upscaling here would undo that care.
- **`--sdpa`, never `--xformers`** — no sm_120 wheels exist. Harmless on Kaggle's
  P100 too, so one flag covers both machines.
## Check before spending GPU minutes
 
```
python scripts/train_lora.py --config spyxfamily.yaml --dry-run
```
 
Prints the image count, the trigger, and the full command without running it.
 
## Expected
 
~45–70 min on a P100 for 1500 steps at batch 1. A checkpoint after every epoch, so an
interrupted session still leaves usable weights. Output:
`spyxfamily_style.safetensors`, roughly 150 MB at rank 32.
 
## Reading the result
 
The first LoRA is a plumbing test, so the bar is low: load it into the POC pipeline at
weight 0.7 and check the output is still coherent. Then step D does the real work.
 
If it trains but the style barely moves, raise `max_train_steps` or `alpha` before
touching the dataset. If poses start breaking — ControlNet losing the outline — lower
`text_encoder_lr` first, since an overcooked text encoder is the usual cause.
 
A LoRA that scrawls pseudo-text is the one failure that points back at the data:
45% of training panels still carry readable lettering (see `claude/DATASET.md`). The
fix there is a stricter `max_text` threshold, not a training knob.
 
## After this
 
1. **B3 vs B4** over the POC's 15 cases — the headline figure of the report.
2. **FID** against the 77 panels in `data/datasets/spyxfamily_holdout/`, which the
   trainer has never seen.
3. Artists 2 and 3. Vagabond's config is stubbed; expect the dataset defaults to
   behave worse there, which is the point of using it.

---

## Update 11 Oct — LoRA v3, and why v1/v2 did nothing

**kohya counts steps in images x folder repeats, not images.** One epoch = sum of
(images x the `N_` prefix) over all concept folders.

| Run | Folders | Images / epoch | 1500 steps = |
|---|---|---|---|
| v1 | `10_` x 270 | 2700 | 0.55 epoch |
| v2 | `10_` x 270 + `5_` x 647 | 5935 | 0.25 epoch |
| **v3** | `2_` x 270 + `1_` x 647 | **1187** | 10 epochs saved |

v1 and v2 saw each image less than once; their B4 grids were near pixel-identical, and a
`lora_scale` sweep 0.4 → 1.0 changed nothing. That was the symptom.

**v3 settings** (now in `configs/spyxfamily.yaml`): alpha 32 (was 16), max_train_steps 6000,
dataset `data/datasets/spyxfamily_v2` with repeats 2/1, everything else unchanged.

**Picking the epoch.** `scripts/compare_loras.py` runs the same poses and seed through
several checkpoints side by side. Epoch 2 → 10: style strengthens (white background, flat
black fills, cleaner ink); from epoch 6 pose details start to fade (flexing arms). **Epoch 4
kept** → `models/loras/spyxfamily/spyxfamily_style_v3.safetensors`.

On Kaggle the repeats were set without re-uploading by copying the input folders to
`/kaggle/working/data/2_sxfstyle` and `1_sxfstyle` in cell 6.
