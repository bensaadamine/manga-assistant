# CLAUDE.md — Manga Panel Assistant

## How to work with me

- I am a student. **No prior experience** with diffusion models. Explain any unknown term in one line.
- **BE BRIEF. This is the most important rule.** Short answers, no essays, no recap of what I already know. If a table or 5 bullet points will do, do not write 3 paragraphs.
- Give me **exact code to copy-paste**, not descriptions.
- **Answer my question first**, before acting on it.
- When several options exist, pick one and say which. Don't list.
- **Never raise anything outside this project's scope** — it causes misunderstandings.
- **Do not re-ask things I have already told you.** Read this file.

## The project

Academic project, two courses: **Generative AI** + **MLOps**.

An assistant for mangaka: the artist gives a **character reference** + a **rough pose outline**;
the system draws that character in that pose, in that artist's style.

Not a universal model — a **reproducible method for adapting the system to one artist**
(one LoRA per series), proven on three visually different series.

**Scope boundary:** the system draws the body, **the artist draws the head — face *and* hair**.
The outline supplies head shape, size and orientation so the drawing lands correctly.

**Machine:** ASUS TUF F16, i7-14650HX, RTX 5060 Laptop 8 GB, 24 GB RAM.
**Reference environment: Kaggle.** The laptop is side research — Windows Smart App
Control blocks the torch import chain kohya needs. Details in `claude/DATASET.md`.

---

# PHASE 1 — POC: DONE, VERDICT **GO**

Full write-up in `claude/POC_VERDICT.md`. Related work and evaluation plan in `claude/RELATED_WORK.md`.

11 of 15 planned cases run, **all 11 passed first attempt**. Three characters, two kinds of
reference, machine- and hand-drawn outlines. **Parameters tuned once on character 1
transferred to characters 2 and 3 untouched.**

## Stack — all pretrained in phase 1

| Layer | Component |
|---|---|
| Base | `gsdf/Counterfeit-V2.5` (SD 1.5) — **settled, do not re-debate** |
| Prompt weighting | `compel` — `(word:1.5)` syntax |
| Text encoder | CLIP skip 2 (`num_hidden_layers=11`) |
| Structure | ControlNet `lllyasviel/control_v11p_sd15s2_lineart_anime` |
| Identity | IP-Adapter Plus `h94/IP-Adapter` → `ip-adapter-plus_sd15.safetensors` |
| Background cleanup | `whiten_bg` — PIL flood fill. **`rembg` was dropped, it crashed Colab** |

**SDXL is out of scope.** The ControlNet and IP-Adapter above are SD 1.5 only.

## Frozen parameters

```python
CN_SCALE = 0.50   # outline strength
CN_END   = 0.70   # ControlNet steers only the first 70% of denoising steps
IP_SCALE = 0.80   # reference strength
```

## Findings — do not rediscover these

**Method**

1. Parameters generalise across characters. One tuning pass covered all three.
2. **The outline must describe the *clothed* silhouette, not the nude body.** ControlNet's
   body edge beats IP-Adapter's fabric: coat tails go transparent, knee socks vanish.
3. **ControlNet traces every line literally.** Contour lines only, nothing inside the body.
   Joint circles → ripped jeans. A line across a head oval → a blindfold.
4. `cn_end` matters as much as `cn_scale` — composition is set early, detail late.
5. IP-Adapter Plus carries composition, not just identity. Above ~0.5 it imposes the
   reference's pose; ControlNet wins that fight at the frozen settings.
6. **IP-Adapter copies EVERYTHING in the reference** — background, props, colour.
   Always whiten the background. **Use a greyscale reference**, or colour leaks into a
   monochrome prompt.
7. Hand-drawn pen sketches work as well as generated ones, same settings.
8. Real manga references (flat line + screentone) work — no `ip_scale` increase needed.
9. The head never transfers. Resolved by scoping the head to the artist.
10. No colour words in a monochrome prompt. Describe shape.
11. **Outline and reference must face the same way.** A back-view outline with a
    front-view reference makes the model draw two heads.

**Code**

12. Prompt weights do nothing without `compel`.
13. **Never call `enable_attention_slicing()`** — breaks IP-Adapter. Use `pipe.vae.enable_slicing()`.
14. Once IP-Adapter is loaded, the UNet **demands an image on every call**. Pass a blank
    white image at scale 0 to disable it.
15. A "naked" negative prompt is NOT needed — bare torsos came from nude test mannequins.
16. **Never `.resize()` an input image.** A landscape outline squashed into 512×768 makes
    every output elongated. Everything goes through `sketches.fit_to()`.

## Open limitations

- Style is soft greyscale, not flat ink + screentone. **Excluded from the pass criterion** — the LoRA's job.
- Head inpainting untested. Deferred by decision.
- Backgrounds occasionally creep in despite negative prompts.

---

# PHASE 2 — PER-ARTIST LoRA (current)

## Dataset — Spy × Family, built from bought volumes

Manga109 was **dropped**: pre-2000 art, too far from the checkpoint, burned-in Japanese text.
Vagabond (bought, scanned) turned out to be the same problem — it is kept as **artist #3**,
replacing `RinToSiteSippuNoNaka`.

Artist #1 is **Spy × Family** (Tatsuya Endo): thin uniform digital line, flat screentone,
anime-standard faces, rectangular panels, lots of full-body framing.

```
355 pages → 2864 panel candidates → 363 kept → 347 after lettering removal
data/datasets/spyxfamily/10_sxfstyle/   270 images + WD14 captions  (train_data_dir is the parent)
data/datasets/spyxfamily_holdout/        77 images                  (FID, never trained on)
```

Pipeline: `python scripts/build_dataset.py --config spyxfamily.yaml` then
`python scripts/tag_dataset.py data/datasets/spyxfamily/10_sxfstyle sxfstyle`.
CPU only, ~90 s. Full write-up, thresholds and known defects in **`claude/DATASET.md`**.

**Known data defect:** 45% of training panels still carry readable lettering. The WD14 tag
`english text` is in those captions, so the model attributes text to that tag rather than to
`sxfstyle`. If the LoRA scrawls pseudo-text, tighten `max_text`, don't change the prompt.

## LoRA #1 — trained

`models/loras/spyxfamily/spyxfamily_style.safetensors` — rank 32, alpha 16, lr 1e-4,
1500 steps, ~20 min on a Kaggle P100. Config and flags in **`claude/TRAINING.md`**.
Runner: `notebooks/01_train_lora_kaggle.ipynb`.

## Current state

- B3 (no LoRA) generated for Yor and Loid. Yor passes; Loid's outline is a back view
  against a front-view reference and produces two heads — input problem, see finding 11.
- **B4 has not been run yet.** That is the next action.

```
python scripts/run_poses.py --config spyxfamily_eval.yaml --baseline b4 \
    --lora models/loras/spyxfamily/spyxfamily_style.safetensors
```

## Phase 2 plan

```
A. Data card                 done
B. Filtering + build_dataset done
C. LoRA #1                   done
D. B3 vs B4 on the eval cases   ← current, headline figure of the report
E. Artists 2 and 3
F. Head inpainting            ← last step, by decision
```

Eval cases live in `configs/spyxfamily_eval.yaml`. Each pose may
carry its own `reference:` and `description:`; `character:` is the fallback.
Prompts are identical for B3 and B4 so only the LoRA differs.

## MLOps maturity ladder

| Level | What | State |
|---|---|---|
| 1 | config + scripts + git | done |
| 2 | DVC data + weights | done (local remote `C:/Users/bensa/dvcstore`; swap to Drive for teammates) |
| 3 | Dockerfile | skeleton exists |
| 4 | CI: tests on push | to do |
| 5 | FastAPI serving, loads LoRA from registry | to do |

Versioning: **git** code · **DVC** data + LoRA weights · **W&B** runs and metrics.

## Evaluation (detail in `claude/RELATED_WORK.md`)

Baselines B0 prompt-only · B1 ControlNet only · B2 IP-Adapter only · **B3 = the POC** ·
**B4 = B3 + LoRA**. B3 → B4 is the headline figure.
Metrics: DINOv2 identity similarity · FID vs the 77 held-out panels · ink statistics ·
OpenPose keypoint distance · retouch time.

## Parameter cheat sheet

| Symptom | Change |
|---|---|
| Output in colour | use a greyscale reference; raise `(monochrome:1.6)` toward 2.0 |
| Pose ignored | raise `cn_scale` |
| Sketch lines traced onto clothing | lower `cn_end` toward 0.6 |
| Character doesn't match reference | raise `ip_scale` |
| Two heads / doubled figure | outline and reference face different ways — fix the outline |
| Elongated or squashed figure | the input was resized without aspect — use `sketches.fit_to()` |
| Reference background or props appear | `whiten_background()` + negative-prompt the prop |
| Clothing missing where outline shows bare body | redraw the outline as the clothed silhouette |
| Out of memory | `n=1` — never `enable_attention_slicing()` |

## Out of scope

Backgrounds, multiple characters, non-human characters, animation, SDXL,
hunting for a better base checkpoint.