# V2 — B3 vs B4, measured

What per-artist adaptation actually bought, on the eval cases. 10–11 October 2026.
Gate from `DESIGN_SPACE.md` §5: *a style-gap number exists*. It does. It is negative.

> **Superseded in part, 11 Oct.** The LoRA measured here is v1/v2, which kohya ran for
> 0.55 / 0.25 of an epoch (see `claude.md` finding 18 and `TRAINING.md` update). LoRA v3
> (alpha 32, epoch 4) fixed it. §3's undertraining suspicion was right; the numbers in §1
> are v1's, not the current system's.

---

## 1. The number

Style is measured as a distance between an image's **ink profile** and the mean profile of
30 of Endo's own held-out panels. Lower is better; 0 is the artist. The scale is arbitrary
and only comparable between our own runs.

| | identity ↑ | style_gap ↓ |
|---|---|---|
| **B3** — ControlNet + IP-Adapter | 0.793 | **1.249** |
| **B4** — B3 + LoRA at scale 1.0 | 0.781 | **1.729** |

**The LoRA loses on both metrics.** It widens the style gap by 39%.

Breakdown against the target:

| | black | midtone | white | edge |
|---|---|---|---|---|
| Endo, 30 held-out panels | 0.107 | 0.120 | **0.672** | 8.75 |
| B3 | 0.043 | 0.422 | **0.069** | 3.92 |
| B4 | 0.009 | 0.682 | **0.146** | 4.69 |

`white_ratio` is the dominant failure and it is **larger than the LoRA question**. Endo's
panels are two-thirds white paper; ours are 7–15%. The LoRA mostly inflates `midtone` —
it turns the image into grey mush — while `black_ratio` collapses to near zero.

Raw rows live in `outputs/metrics.csv`, one per run, appended by `scripts/evaluate.py`.

## 2. What was ruled out

**Not the training data.** The 270 training panels were measured against the 77 held-out
ones: style_gap 0.146, effectively identical (0.123 / 0.153 / 0.627 / 9.58 vs
0.107 / 0.120 / 0.672 / 8.75). The panels are white and inked. Rebuilding the dataset for
tone reasons would fix nothing. *This reversed an earlier recommendation to bring V4
forward.*

**Not Counterfeit's gender bias.** POC character 1 was a teenage boy, 5/5 pass; case M1
reproduced it with a clean outline.

**Not `lora_scale`.** A sweep at 0.4 / 0.6 / 0.8 / 1.0 showed the grey field identical at
every value. That sweep contained **no zero point**, so it was wrongly read at the time as
"the LoRA is not the cause". B3 *is* the zero point, and it says otherwise. A sweep without
a control is not a control.

## 3. The remaining suspect

The `.safetensors` metadata:

```
ss_max_train_steps       = 1500
ss_num_batches_per_epoch = 1350
ss_network_dim = 32   ss_network_alpha = 16   ss_clip_skip = 2
ss_sd_model_name = gsdf/Counterfeit-V2.5
```

1500 / 1350 = **1.1 epochs**. `ss_num_epochs` said 2, but `max_train_steps` cut it short.
The LoRA saw the data roughly once — enough to disturb the rendering, not enough to impose
a style.

**Decision rule for the retrain** (`max_train_steps` 1500 → 4000, nothing else changed):

| new B4 style_gap | conclusion |
|---|---|
| < 1.25 | the LoRA finally helps — V2 passes with a positive number |
| 1.25 – 1.73 | better but still losing; undertraining was only part of it |
| ≥ 1.73 | undertraining was not the cause → go to V3 (`counterfeit-flat`) |

Independently of the outcome: **B3 itself sits at `white_ratio` 0.069 against 0.672.** The
grey field is Counterfeit's, not the LoRA's — exactly the weakness `DESIGN_SPACE` §2.1
attributes to it. V3 is what attacks it, and there is now a number justifying V3.

## 4. The dataset rebuild — in flight

Manual inspection found roughly **10 of 270** panels met the ideal (one character, no
lettering, light background). Filtering ten volumes to that standard is not viable.

**The move is not to filter panels but to crop characters out of them.** A panel with three
characters and two speech bubbles yields three single-character crops with no text. Magi
(`ragavsachdeva/magiv2`) gives a box per character.

Two corrections to the requirements that made this work:

- **Full body is not needed for a style LoRA.** Style is line weight, screentone, how a face
  and fabric are drawn. Waist-up teaches all of it. The full body comes from ControlNet.
- **Multi-character crops are a dead end** — they teach composition the pipeline never
  generates.

**First attempt, 11 October:** 1443 crops from 355 pages, rejecting anything under 256 px.
Visual check of 40 found roughly **1 in 3 usable**. The rest carried panel borders and
neighbouring panels, speech text and SFX, two characters, or were extreme close-ups — an
eye, a mouth, a hand. The ink metric does not filter these (85% pass), so it is no help here.

Root cause: crops were padded to 512 square around Magi's box, pulling in surroundings, and
the JSON was not saved so nothing could be recut.

**Second attempt:** `scripts/magi_crops.py`, six rules —

| | Rule | Fixes |
|---|---|---|
| 1 | 5% margin, no square padding | panel borders, neighbouring panels |
| 2 | whiten Magi's text boxes inside the crop | bubbles and SFX |
| 3 | drop a box another character covers by >15% | two-character crops |
| 4 | box short side ≥ 256 px | blur |
| 5 | aspect ratio within 0.5–2.0 | eye-slits, narrow strips |
| 6 | resize short side to 512, centre crop | uniform size |

It saves `magi.json` so thresholds can be retuned without re-running Magi. Expect 400–700
clean crops. Below 300, add the third volume.

> Shipped as `scripts/crops_from_magi.py` with rules 2 and 6 corrected: never fill a text
> box with white (it punched holes through ~258 faces — erase only ink blobs fully inside
> the box), and no square pad or centre crop (it cut heads and feet; kohya buckets handle
> aspect). Result: 647 crops. See `DATASET.md`, update 11 Oct.

## 5. Tooling added

| File | What |
|---|---|
| `src/manga/evaluate.py` | `ink_target()` computes the target profile from the held-out panels; `style_gap()` is the distance; `white_ratio` added; `pose_distance()` implemented with OpenPose (the `NotImplementedError` stub is gone); everything resized to 512² so panels and outputs are comparable |
| `scripts/evaluate.py` | generates images **individually** (the grid from `run_poses.py` is not measurable), scores them, appends one row to `outputs/metrics.csv`. `--compare` prints the history |
| `scripts/magi_crops.py` | the six crop rules above |
| `src/manga/pipeline.py` | `load_lora()` no longer calls `fuse_lora` — see §6; `set_lora_scale()` added |
| `scripts/sweep.py` | `lora_scale` added as a sweepable parameter, with `--lora` |
| `scripts/run_poses.py` | reads `cfg.characters[...]`; `--no-pose-text` ablation flag; `_nopose` output suffix |

**FID is deliberately not reported.** It needs ~50 images per side; 9 cases at n=2 gives 18.
`scripts/evaluate.py` skips it and writes the reason into the row's `note` column. For a real
FID, run with `--n 6`.

## 6. One bug worth remembering

B4 was identical to B3 for a whole morning. Cause: `models.pipe.fuse_lora(lora_scale=scale)`
called **without** `components`. In the installed diffusers version the default is an empty
list, so it fused nothing and raised nothing.

The fix was to stop fusing. `load_lora_weights` + `set_adapters` leaves the adapter live, the
scale is read inside the layer's forward pass, and that survives being called through
`cn_pipe` — a different pipeline object sharing the same UNet. It also makes a `lora_scale`
sweep cheap, since no reload is needed per value. A hard assertion on injected module count
now fails loudly instead of silently.

---

## 7. LoRA v3 measured — 22 October (Kaggle, n=6, 9 cases, 54 images per side)

| | identity ↑ | style_gap ↓ | midtone | white | FID ↓ |
|---|---|---|---|---|---|
| B3 | 0.774 | 1.356 | 0.477 | 0.001 | 283.0 |
| **B4 — LoRA v3 (epoch 4), scale 1.0** | **0.785** | **1.149** | **0.367** | **0.212** | **265.6** |
| Endo, held-out | — | 0 | 0.120 | 0.672 | — |

**V2 passes.** style_gap 1.149 < 1.25, so §3's decision rule says the LoRA helps. B4 beats B3
on every metric. v1 → v3 took B4 from 1.73 (worse than B3) to 1.15 (better), which confirms
undertraining was the cause. A repeated B4 run gave identical numbers (fixed seed).

Remaining gap: white_ratio 0.21 against 0.67. FID is meaningful only as B3 vs B4, not in
absolute terms, at 54 images per side.

**But visually the result is still bad.** The metrics measure *relative* progress over B3, not
quality: B4 is less wrong than B3, not right. The outputs still read as grey, soft renders
rather than Endo's flat ink on white paper (white 0.21 vs 0.67 says the same thing in numbers).
The headline is "the LoRA moves the system in the right direction", **not** "the style is
solved". Report both the table and the image grids side by side, so the claim is not overstated.
