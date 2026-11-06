# Design space and versioning plan

Companion to `V2_RESULTS.md` (what was measured) and `RELATED_WORK.md` (what exists elsewhere).
Compiled 8 October 2026. Rewritten 6 November 2026.

Why each component was chosen, what replaces it if it fails, and the order of the next
experiments. Everything marked **[untested]** is a hypothesis, not a result.

---

## 1. Components

### 1.1 Base checkpoint

- **Chosen:** `gsdf/Counterfeit-V2.5`.
- **Why:** clean separable lines, sound anatomy, not over-stylised. SD 1.5 is chosen for
  **iteration speed on an 8 GB laptop GPU (RTX 5060)**, and because the `lineart_anime` ControlNet exists
  officially only for SD 1.5. SDXL equivalents exist (IP-Adapter-Plus-SDXL is official;
  lineart ControlNets are community-made) but are several times slower to train and run on
  this hardware. Upgrading compute is out of scope.
- **Alternatives:** AnythingV5, MeinaMix, Cetus, AbyssOrange2 — all SD 1.5, all Danbooru
  lineage, mostly merges of each other. Lateral moves.
- **Known weakness:** soft greyscale with gradients and cast shadows, not flat ink +
  screentone. Measured: B4 white_ratio 0.21 against the artist's 0.67, and B3 shows the grey
  comes from Counterfeit itself (`V2_RESULTS` §7). Addressed by V3, §2.
- **Rejected:** training a base from scratch (10⁵–10⁷ images, 1000s of GPU-h, wrong layer);
  distilling DiffSensei (no outline input to query it with, SDXL→SD 1.5 mismatch, 13B MLLM);
  full fine-tune to flat manga (deferred — data, not compute, is the blocker).

### 1.2 Structure, identity, prompt, style

| Layer | Chosen | Alternatives / notes |
|---|---|---|
| Structure | ControlNet `lineart_anime`, CN_SCALE 0.50, CN_END 0.70 | canny, scribble, softedge. OpenPose rejected: the artist's own drawing is the control signal |
| Identity | IP-Adapter Plus, IP_SCALE 0.80 | plain IP-Adapter (loses exact collar/silhouette), FaceID (head is out of scope), per-character LoRA (needs per-character training, defeats zero-shot) |
| Prompt | `compel`, CLIP skip 2 | weights do nothing without compel; layer 12 gives softer, less obedient output |
| Style | LoRA per artist, rank 32 / alpha 32 / lr 1e-4, epoch 4 (`V2_RESULTS` App. A) | DreamBooth too heavy, textual inversion too weak, QLoRA irrelevant (7B+ LLM trick) |

### 1.3 Environment

Local first: RTX 5060 Laptop, 8 GB VRAM. Kaggle is the fallback when 8 GB is not enough.
RTX 50 is sm_120: cu128+ and `--sdpa`, never `--xformers`. Never
`pip install -r sd-scripts/requirements.txt`.

---

## 2. Next step — V3, `counterfeit-flat` [untested]

**What it is.** Not a fine-tune, no training: Counterfeit plus an existing flat-manga LoRA,
merged into the weights with kohya's merge script. The merged model becomes the base; the
per-artist LoRA is then retrained on top of it.

| | Anime Lineart / Manga-like | MANGA (General) Style Collection |
|---|---|---|
| Civitai id | 16014 | 8667 |
| Trained on | Anything V4.5 (close sibling of Counterfeit) | Anything V4.5, 40 images, 512² |
| Triggers | `lineart`, `monochrome` | `manga`, `greyscale`, `monochrome` |
| Weight | 0.8–1.0 | 0.40–0.60 |
| Register | clean line art, light screentone | mature — Chainsaw Man / Death Note |

**Risks.** (1) Style vs identity: a practitioner report found a manga base pushes character
identity back toward pale greyscale, and the reverse. (2) Licences: some assets forbid
redistribution — check before use, since the method claims reproducibility.

**Protocol — half a day.**
1. Download 16014 and 8667; check licences.
2. Run the 15 POC cases at LoRA weight 0.3 / 0.5 / 0.8, CN and IP frozen.
3. Score flatness (ink statistics), identity (DINOv2), pose (OpenPose keypoint distance).
4. Merge the better one into Counterfeit → `counterfeit-flat`.
5. Re-run the 15 cases to confirm anatomy and prompt-following survived.

**Decision rule.** Flatter output with identity and pose intact → `counterfeit-flat` becomes
the base. If it fights identity as the practitioner report describes, that is a finding, and a
full fine-tune is unlikely to behave differently.

---

## 3. Versioning plan

Each version changes **one** thing and produces **one** number.

| V | Change | Question | Gate |
|---|---|---|---|
| **V0** done | POC: Counterfeit + ControlNet + IP-Adapter | Can identity and pose coexist without training? | 11/11, GO |
| **V1** done | + LoRA #1 (Spy × Family) | Does data → kohya → `.safetensors` → pipeline work end to end? | trained 9 Oct |
| **V2** done | B3 vs B4 on the eval cases | **What does per-artist adaptation buy?** | style_gap 1.149 < 1.25, passed (`V2_RESULTS` §7) |
| **V4** done | Magi character crops; dataset rebuilt | Does removing lettering stop pseudo-text? | 647 crops, used by LoRA v3 (`V2_RESULTS` §4) |
| **V3** next | Flat-manga LoRA merged (`counterfeit-flat`) | Does a flatter base help, or does the artist LoRA already do it? | §2 decision rule |
| **V5** | Artists 2 and 3 (Vagabond last) | **Does the method transfer?** | three artists, one recipe |
| **V6** | Head inpainting | Clean neck seam? | last step, by decision |

**Optional arm if V3 succeeds:** a shared "flat manga" LoRA across all three series, stacked
beneath the artist LoRA (rank 8–16, weight 0.4–0.5) or merged. Ablation: *does a shared manga
base help, or is per-artist enough?* Needs artists 2 and 3 preprocessed, which V5 requires
anyway.

---

## 4. Open questions

1. Does a LoRA close the style gap to flat ink and screentone, or only narrow it?
2. Does the clothed-silhouette requirement hold for loose clothing — capes, long hair?
3. How many reference images does a new series need before the method transfers?
4. Does head inpainting leave a clean neck seam?
5. Does the style↔identity conflict appear in our pipeline, where identity comes from
   IP-Adapter rather than a second LoRA? Our architecture may sidestep it — the two signals
   enter at different places. Test at V3.

## Sources

- DiffSensei — https://arxiv.org/html/2412.07589v1
- Magi / The Manga Whisperer — https://arxiv.org/abs/2401.10224
- Anime Lineart / Manga-like LoRA — https://civitai.com/models/16014
- MANGA (General) Style LoRA Collection — https://civarchive.com/models/8667
- Manga BW workflow test (practitioner) — https://lilting.ch/en/articles/manga-bw-workflow-test-draft
