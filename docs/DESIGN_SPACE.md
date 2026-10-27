# Design space and versioning plan
 
Companion to `POC_VERDICT.md` (what was proven) and `RELATED_WORK.md` (what exists elsewhere).
Compiled 8 October 2026. Last amended 9 October 2026 (§2.6, §5).
 
Records the alternatives considered at each layer, why the current component was chosen, and
what is worth trying later. Purpose: make every choice defensible rather than accidental, and
keep rejected ideas on record with the reason, so they are not re-litigated.
 
Everything marked **[untested]** is a hypothesis, not a result.
 
---
 
## 1. Positioning against DiffSensei
 
**DiffSensei trains nothing from scratch.** Both halves start from pretrained weights. The
difference is how much is fine-tuned, and on what.
 
| Component | Initialised from | Trained on MangaZero |
|---|---|---|
| Image generator | SDXL | all U-Net + feature-extractor params, 250k steps (stage 1) |
| Feature extractor | IP-Adapter-Plus-SDXL | fully trained |
| MLLM adapter | SEED-X | LoRA (rank 64) + resampler only, 20k steps (stage 2) |
 
| | DiffSensei | Ours |
|---|---|---|
| Generator | SDXL U-Net, ~2.6B | SD 1.5 U-Net, ~0.86B |
| Extra modules | SEED-X MLLM (13B class) + resampler | ControlNet ~0.36B + IP-Adapter Plus |
| Trained params | billions (full stage 1) + LoRA on the MLLM | ~20–40M: one LoRA, ~150 MB |
| Training data | 43,264 pages / 427,147 panels, 48 series | one artist's volumes — 270 panels |
| Compute | multi-GPU, 270k steps | one P100, ~1 h |
| Control signal | text caption + bounding boxes | **the artist's own pose drawing** |
| Style | one generic manga model | **one LoRA per artist** |
| Produces | whole pages, multi-character, dialog layout | one character, one panel, body only |
 
**Framing for the report:** their advantage is scale and generality; ours is per-artist
adaptability, the drawing as control signal, and an explicit human-in-the-loop boundary.
Different claims, not a quality contest we would lose.
 
---
 
## 2. Component design space
 
### 2.1 Base checkpoint
 
- **Chosen:** `gsdf/Counterfeit-V2.5`.
- **Why:** structural criteria — clean separable lines, sound anatomy, not over-stylised. SD 1.5
  is mandatory: lineart-anime ControlNet and IP-Adapter Plus exist only for SD 1.5 in the
  versions we need.
- **Alternatives:** AnythingV5, MeinaMix, Cetus, AbyssOrange2 — all SD 1.5, all NovelAI/Danbooru
  lineage, mostly merges of each other. Lateral moves.
- **Known weakness:** soft greyscale with gradients and cast shadows, a rendered almost-3D look,
  not flat ink + screentone. Prior also skews to a narrow subject range.
- **To try:** `counterfeit-flat` — Counterfeit with a flat-manga LoRA merged in. See §4. [untested]
- **Revisit when:** step D (B3 vs B4) produces a style number. Not before.
### 2.2 Base model strategy
 
| Option | Cost | Verdict |
|---|---|---|
| Counterfeit as-is | zero | **Current.** 11/11 POC cases |
| Train a base from scratch | 10⁵–10⁷ images, 100s–1000s GPU-h | **Rejected.** Impossible on the budget; also the wrong layer |
| Full fine-tune to flat manga | 2–5k panels, 10–20k steps, ~6–15 h P100 | **Deferred.** Data, not compute, is the blocker. Risks the anatomy that motivated the checkpoint. LR 1e-6–5e-6, not 1e-4 |
| Merge a flat-manga LoRA in | ~1 h | **Recommended first move.** Same artifact, answers the question cheaply |
| Distil DiffSensei | research-scale | **Rejected.** See §3 |
 
### 2.3 Structure and identity
 
| Layer | Chosen | Alternatives / notes |
|---|---|---|
| Structure | ControlNet `lineart_anime`, CN_SCALE 0.50, CN_END 0.70 | canny, scribble, softedge, openpose. **Openpose is now a scheduled ablation, not a loose idea — see §2.6** |
| Identity | IP-Adapter Plus, IP_SCALE 0.80 | plain IP-Adapter (loses exact collar/silhouette), FaceID (head is out of scope), textual inversion / per-character LoRA (needs per-character training, defeats zero-shot) |
| Prompt | `compel`, CLIP skip 2 | weights do nothing without compel; layer 12 gives softer, less obedient output |
| Style | LoRA per artist, rank 32 / alpha 16 / lr 1e-4 / 1500 steps | DreamBooth too heavy, textual inversion too weak, QLoRA irrelevant (7B+ LLM trick). **Rank 64 and 3000 steps are the first knobs to move** |
 
### 2.4 Dataset and preprocessing
 
- **Chosen:** Spy × Family self-scanned → 270 train + 77 holdout. Vagabond as artist #3.
- **Rejected — Manga109:** pre-2000 art, burned-in Japanese text.
- **Rejected — MangaZero as images:** the HF release is 412 MB of JSON, 32,715 rows —
  annotations plus MangaDex CDN URLs, **no images**. MIT covers annotations, not artwork. URLs
  are session-scoped and a 2024 dump may be dead. Corpus spans 1974–2024 and skews pre-2000.
- **Accepted — MangaZero as annotations:** panel/character/dialog boxes, character IDs, LLaVA
  captions. Free ground truth to validate our panel-splitter and text-detector. Needs no images.
  Character `type`: 0 = clean face, 1 = vague face, 2 = exaggerated/non-standard.
- **Known defects:** ~10% under-split panels; borderless panels shatter; **45% of training
  panels still carry readable lettering**.
- **To try — Magi** (`ragavsachdeva/magi`, CVPR 2024): trained manga-understanding model giving
  panel, character and text boxes. Fixes under-splitting and lettering together. ~500M params,
  ~3–4 GB VRAM — runs on the RTX 5060, or CPU at ~30 s/page. The blocker was never the GPU, it
  was the blocked HF download in the build sandbox. Run on Kaggle, commit the JSON, pipeline
  stays torch-free. [untested]
### 2.5 Environment
 
Kaggle (P100) for training; local RTX 5060 for side research. Smart App Control blocks
`optree/_C.pyd`, which every sd-scripts entry point reaches via `library.dataset` → diffusers →
`torch._dynamo`. RTX 50 is sm_120: cu128+ and `--sdpa`, never `--xformers`. Never
`pip install -r sd-scripts/requirements.txt`.
 
### 2.6 The control-signal question — lineart outline vs OpenPose skeleton
 
**Status: decided to keep lineart for now, with a dated fallback. Not settled.**
 
Raised 9 October after the first B3 run produced a two-headed figure on a back-view pose.
 
| | Lineart outline (current) | OpenPose skeleton |
|---|---|---|
| What the artist produces | a drawing | a posed manikin in a GUI |
| Silhouette control | coat length, skirt shape, build | none |
| Garment *behaviour* in the pose | yes — flaring coat, lifted hem | **no** |
| Clothed-silhouette requirement | yes (POC finding #2) | no |
| Stray lines becoming features | real risk (POC finding #3) | none |
| Back / turned poses | ambiguous → observed failure | unambiguous |
 
**The counter-argument, recorded because it is good.** POC finding #2 ("ControlNet's body edge
beats IP-Adapter's fabric") describes a *conflict* — a nude silhouette contradicting a clothed
reference. A skeleton supplies no silhouette, so there is no conflict and IP-Adapter is
unopposed on clothing. The finding therefore does **not** show that OpenPose would lose the
clothes.
 
**What would still be lost** is garment behaviour specific to the pose. IP-Adapter carries one
static reference; it cannot know how that coat moves in a pose it has never seen. That is the
artistic intent the method claims to preserve.
 
**Decision (9 Oct):** continue with lineart for a few more days. If outputs do not improve,
switch to OpenPose and accept that the workflow changes — the artist poses rather than draws,
and the "artist's own drawing as control signal" claim in §1 has to be rewritten.
 
**Not a blind switch.** V2 runs first so the LoRA's gain lands on a clean baseline; V2b then
swaps only the ControlNet, same poses, same references, same seed. "We tested both control
signals" is a stronger report than either choice asserted up front.
 
---
 
## 3. Ideas evaluated and rejected
 
**Distilling DiffSensei into a small model — rejected.** Four independent reasons:
it cannot be queried for our task (no pose-outline input, so no teacher outputs matching our
inputs); architecture mismatch (SDXL→SD1.5 cross-architecture distillation is a research
problem); it would distil the average of 48 series when our contribution is per-artist; the
MLLM half is 13B class. *The adjacent legitimate idea — DiffSensei as a synthetic data
generator — is not distillation and is not blocked in principle, but still needs SDXL + a 13B
model running somewhere and still yields generic-manga-average. Not pursued.*
 
**Own base model from scratch — rejected.** Wrong layer, wrong budget, and it damages the
contribution: "a reproducible method for adapting an existing model" is defensible; "we trained
a base model" is not supportable on a student budget.
 
**Head inpainting — deferred by decision**, not rejected. Last step of phase 2.
 
---
 
## 4. The flat-base investigation
 
**The question:** Counterfeit renders soft and shaded; the target mangaka are flat line +
screentone. Would a one-time intermediate fine-tune — producing a new base the per-artist
pipeline then uses unchanged — close part of the gap before the artist LoRA is applied?
 
### 4.1 What exists
 
No credible flat-manga SD 1.5 **checkpoint**. What exists is LoRAs — which makes the flat base
a download and a merge rather than a training run.
 
| | Anime Lineart / Manga-like | MANGA (General) Style Collection |
|---|---|---|
| Civitai id | 16014 | 8667 |
| Trained on | Anything V4.5 | Anything V4.5, 40 images, 512² |
| Triggers | `lineart`, `monochrome` | `manga`, `greyscale`, `monochrome` |
| Weight | 0.8–1.0 | 0.40–0.60 |
| Register | clean line art, light screentone | mature — Chainsaw Man / Death Note |
 
Both trained on Anything V4.5, a close sibling of Counterfeit, so they should transfer with only
a weight adjustment.
 
### 4.2 Two warnings
 
**The style↔identity conflict is documented, not hypothetical.** The author of 16014 states it
does not always combine with character LoRAs — you must lower one or the other. A 2025
practitioner write-up ran essentially our experiment: stacking a character LoRA on a
manga-monochrome base let the character signal dominate and pushed the monochrome back toward
pale greyscale; weakening the character LoRA restored the manga look but broke the character
lock. Same author also tested SD 1.5 lineart ControlNet with CounterfeitV30. **Worth citing in
related work** as practitioner evidence.
 
**Licensing.** Several assets are NSFW-flagged, some need an account, and at least one B/W
checkpoint forbids redistribution — unusable in a method claiming reproducibility. All remain
Danbooru-lineage underneath.
 
### 4.3 Test protocol — half a day, no training
 
1. Download 16014 and 8667; check licences first.
2. Run the 15 POC cases at LoRA weight 0.3 / 0.5 / 0.8, frozen CN and IP.
3. Score flatness (ink statistics), identity (DINOv2), pose (OpenPose keypoint distance).
4. Merge the better one into Counterfeit with kohya's merge script → `counterfeit-flat`.
5. Re-run the 15 cases to confirm anatomy and prompt-following survived.
**Decision rule:** if the merge flattens output without costing identity or pose,
`counterfeit-flat` becomes the base and the full fine-tune is dead. If both LoRAs fight identity
the way the practitioner report describes, that is a finding, and the full fine-tune is unlikely
to behave differently.
 
---
 
## 5. Versioning plan
 
Each version changes **one** thing and produces **one** number. Ordering matters: B3→B4 is the
headline figure and must be measured on a clean baseline. Adding the flat base first would make
the gain unattributable.
 
| V | Change | Question | Gate |
|---|---|---|---|
| **V0** done | POC: Counterfeit + ControlNet + IP-Adapter | Can identity and pose coexist without training? | 11/11, GO |
| **V1** done | + LoRA #1 (Spy × Family), weight 0.7 | Does data → kohya → `.safetensors` → pipeline work end to end? | trained 9 Oct, 37.8 MB |
| **V2** next | B3 vs B4 on the eval cases | **What does per-artist adaptation buy?** | a style-gap number exists |
| **V2b** | Swap lineart → OpenPose, everything else frozen | Does the skeleton fix orientation without losing garment control? | §2.6; one number vs V2's |
| **V3** | + flat-manga LoRA merged (`counterfeit-flat`) | Does a flatter base help, or does the artist LoRA already do it? | identity and pose not degraded |
| **V4** | + Magi detection; rebuild the dataset | Does cutting lettering from 45% to ~0 stop pseudo-text? | lettering rate measured |
| **V5** | Artists 2 and 3 (Vagabond last) | **Does the method transfer?** | three artists, one recipe |
| **V6** | Head inpainting | Clean neck seam? | last step, by decision |
 
**Optional arm if V3 succeeds:** a shared "flat manga" LoRA across all three series under one
trigger, stacked beneath the artist LoRA (rank 8–16, weight 0.4–0.5) or merged. Second
publishable ablation — *does a shared manga base help, or is per-artist enough?* — at a fraction
of a full fine-tune. Needs artists 2 and 3 preprocessed, which V5 requires anyway.
 
## 6. Open questions
 
1. Does a LoRA close the style gap to flat ink and screentone, or only narrow it?
2. Does the clothed-silhouette requirement hold for loose clothing — capes, long hair?
3. How many reference images does a new series need before the method transfers?
4. Does head inpainting leave a clean neck seam?
5. Does the style↔identity conflict appear in our pipeline, where identity comes from
   IP-Adapter rather than a second LoRA? **Our architecture may sidestep it** — the two signals
   enter at different places. Test at V3.
6. Is the drawn outline actually the better control signal, or only the one that fits the
   story we want to tell? §2.6, decided at V2b.
## Sources
 
- DiffSensei — https://arxiv.org/html/2412.07589v1
- MangaZero dataset card — https://huggingface.co/datasets/jianzongwu/MangaZero
- Magi / The Manga Whisperer — https://arxiv.org/abs/2401.10224
- Anime Lineart / Manga-like LoRA — https://civitai.com/models/16014
- MANGA (General) Style LoRA Collection — https://civarchive.com/models/8667
- Manga BW workflow test (practitioner) — https://lilting.ch/en/articles/manga-bw-workflow-test-draft