# Related work and positioning

Literature survey for the "what exists already / what is our added value" requirement.
Compiled 30 September 2026. Reworked 6 November 2026.

Each entry: what it does, its **components**, its model and cost, how it differs from us, and
whether it can be tested today.

---

## 1. Academic work

### 1.1 Closest to us

**AnimeAdapter (2026)** — An add-on that keeps an anime character's appearance consistent
across images; pose comes from a separate ControlNet.
Components: **CLIP image encoder (7 layers fused)** → **IP-Adapter-style cross-attention** +
**SAM 3 masks to drop background tokens** + **frozen OpenPose ControlNet / T2I-Adapter**.
- Model: frozen SD, version not stated (1024 px references suggest SDXL); **24.7M trainable
  params**, 1× A100.
- vs us: a stronger replacement for our IP-Adapter Plus (benchmarked against it directly).
  Pose from a skeleton, not a sketch. No style learning.
- Testable: **no** — "code and weights will be released".

**SketchKeyAnime (2026)** — Generates an animation clip from one reference image plus a few
key sketches.
Components: **SparseCtrl-style sketch ControlNet** + **CLIP sketch-context encoder (Relation
Transformer + Resampler)** + **gated sketch cross-attention** + **line-art-weighted loss**.
- Model: **Stable Video Diffusion (~1.5B)**; 4× V100; 60k Sakuga-42M clips at 256×256.
- vs us: same sketch + reference fusion, but the output is video, training is heavy, and there
  is no per-artist style.
- Testable: **no** — no code released.

**Interactive Drawing Guidance for Anime Illustrations (2025)** — Turns the artist's strokes
into an anime image in real time, then back into a cleaned sketch the artist can draw over.
Components: **StreamDiffusion** (cached real-time SD pipeline) + **style LoRA (40 images)** +
**LCM-LoRA** (few-step speed-up) + **Informative Drawings** (image → sketch) + **skip when the
sketch barely changed**.
- Model: SD, version not stated; **RTX 4090, ~1.2 s for 4 images at 512 px, ~13 GB VRAM**.
- vs us: closest on artist-in-the-loop with a LoRA, but it *guides* the artist's sketch rather
  than drawing the character. No identity reference.
- Testable: **no official code**; rebuildable from public parts (StreamDiffusion + LCM-LoRA).

### 1.2 Manga generation (DiffSensei family)

**DiffSensei — customized manga generation (CVPR 2025)** — Generates whole manga pages with
multi-character control from text plus character reference images.
Components: **SDXL generator (fully fine-tuned)** + **IP-Adapter-Plus-SDXL feature extractor** +
**multimodal LLM as a text-compatible identity adapter (SEED-X, LoRA)** + **MangaZero dataset**
(43,264 pages, 427,147 panels).
- Model: SDXL (~2.6B UNet) + 13B-class MLLM; multi-GPU, 270k steps.
- vs us: generates *pages* from *text* — it replaces the artist's composition; we execute it.
  Trained once across 48 series, not adapted per artist.
- Testable: yes — code and weights public; far beyond our hardware.

**MangaFlow (2026)** — LLM agents turn a story into finished manga pages: plot plan, layout,
panels, speech bubbles.
Components: **story planner + section memory** → **layout agent** → **panel agent** →
**reference-conditioned renderer** → **text / bubble agent**.
- Model: **FLUX.2 (9B)** or **Gemini 2.5 Flash Image (closed API)** as renderer; planner LLM not
  named.
- vs us: text → page, like DiffSensei. ~9× our model size, or a paid API.
- Testable: **no code**; only its renderers can be tried as baselines.

**Manga Generation via Layout-controllable Diffusion (2024, a.k.a. MangaDiffusion)** —
Generates a whole manga page from text, learning panel layout and reading order jointly.
Components: **intra-panel attention** (per-panel caption) + **inter-panel attention** (layout /
order) + **speech-bubble mask** + **padding mask**; **Manga109Story** dataset (21k pages).
- Model: **Latte DiT (~0.7B)** + SD 1.4 VAE + T5 text encoder; 4× A100, ~65 h, 512×384.
- vs us: text only, no character reference (listed as future work). Generic style.
- Testable: **partly** — code on GitHub; weights and dataset release not confirmed. Manga109
  needs a licence request.

### 1.3 Manga line art and screentone

**MangaNinja — line art colorization with precise reference following (CVPR 2025)** —
Colorizes finished line art from a reference, using **patch shuffling** and **point-driven
control** for large pose differences.
- vs us: takes *finished line art*; we take a *rough outline* and produce the drawing.
  Complementary — a later stage of the same production pipeline.

**Sketch2Manga** — Screentone-shaded manga from a sketch via diffusion. Closest to our style
layer, but style only, no character identity.

**Screentone-preserved manga retargeting / screentone VAE (Xie et al.)** — Generating and
preserving screentone specifically. Directly relevant to our main limitation (soft greyscale
instead of flat ink and screentone).

**Manga109 segmentation annotations (CVPR 2025)** — Segmentation masks for Manga109 beyond
the existing boxes. Possible source of cleaner character masks than bounding boxes.

### 1.4 Style personalization (our phase 2 method sources)

**ZipLoRA (2023)** — Merges a subject LoRA and a style LoRA into one by learning per-column
mixing weights.
Components: **2 separately trained LoRAs** + **learned merge coefficients** + **loss that keeps
the two apart**.
- Model: **SDXL (~2.6B UNet)**, 3× our SD 1.5 UNet (~0.86B).
- vs us: addresses the style↔identity risk; ours gets identity from an adapter, not a second
  LoRA, which may sidestep it (`DESIGN_SPACE` open question 5).
- Testable: **yes** — unofficial `mkshing/ziplora-pytorch` (SDXL only; heavy for a T4).

**Opt-In Art (NeurIPS 2025)** — Learns one artist's style from 9–50 paintings on a base model
trained with no art at all.
Components: **"Blank Canvas" base trained from scratch on 9M art-free photos** + **BERT text
encoder (no CLIP)** + **LoRA on UNet up-blocks** + **style loss + content-preservation loss**.
- Model: SD 1.4 architecture (~0.86B UNet, our size class), but the base cost **~11.4k
  A100-hours**; the LoRA itself is 1,000 steps.
- vs us: same per-artist few-shot LoRA idea, and the same "how many panels does an artist need"
  question. The art-free base is a consent argument, not a quality gain, and no existing
  ControlNet or IP-Adapter fits its UNet.
- Testable: **no** — no weights yet; dataset promised on publication.

**AC-LoRA, Meta-LoRA, LoRA.rar** — automatic component selection, meta-learned initialization,
and hypernetwork merging of subject + style LoRAs. Further method sources, not competitors.

### 1.5 Identity-preserving pose-guided animation (PHiD, AniFaceDiff, etc.)

Keeps a person's identity while changing pose. Almost all of it targets photorealistic humans
and video; the techniques transfer conceptually, but none of it works on black-and-white manga
line art with an artist in the loop.

### 1.6 Size and cost at a glance

| Work | Generator | Size vs ours (SD 1.5, ~1B) | Trainable on a T4? |
|---|---|---|---|
| **Ours** | SD 1.5 + ControlNet + IP-Adapter + LoRA | 1× | **yes** |
| AnimeAdapter | SD (version unstated) | ~1–3× | adapter likely |
| SketchKeyAnime | SVD | ~1.5× (+ video) | no |
| Interactive Guidance | SD + LCM | ~1× | LoRA yes |
| DiffSensei | SDXL + 13B MLLM | ~15× | no |
| MangaFlow | FLUX.2 9B / Gemini | ~9× / API | no |
| MangaDiffusion | Latte DiT | ~0.7× | no (A100s) |
| ZipLoRA | SDXL | ~3× | tight |
| Opt-In Art | SD 1.4, from scratch | 1× (11k GPU-h base) | LoRA yes |

Quality is reported on different metrics in each paper, so a real quality-per-size comparison
exists only for what we can run ourselves — today ZipLoRA and, partly, MangaDiffusion.

---



## 3. Our positioning

### 3.1 Added value

No surveyed work combines all four:

| | Artist's own sketch as control | Identity from reference | Per-artist manga style | Runs on 1 consumer GPU |
|---|---|---|---|---|
| **Ours** | ✓ rough outline, clothes included | ✓ IP-Adapter | ✓ LoRA, ~650 crops, ~1 h | ✓ P100 / T4 |
| AnimeAdapter | ✗ skeleton | ✓ | ✗ | ~ |
| SketchKeyAnime | ✓ | ✓ | ✗ generic, video | ✗ 4× V100 |
| Interactive Guidance | ✓ | ✗ | ~ one LoRA, not per artist | ✓ RTX 4090 |
| Opt-In Art | ✗ text | ✗ | ✓ | base ✗ (11k GPU-h) |
| DiffSensei | ✗ text | ✓ | ✗ | ✗ |
| MangaFlow | ✗ text | ✓ | ✗ | ✗ |
| MangaDiffusion | ✗ text | ✗ | ✗ | ✗ |

Plus a fifth point no one else defines: **an explicit human-in-the-loop boundary** — the system
draws the body, the artist draws the head.

> Others decide the composition for the artist; ours executes the artist's composition, in the
> artist's style, from a few hundred panels, on free hardware.

### 3.2 What we combine, and from where

| Aspect | Closest prior work | What we change |
|---|---|---|
| Sketch + reference fusion | SketchKeyAnime | still images, the artist's rough outline instead of clean key sketches, consumer GPU |
| Identity from a reference image | IP-Adapter, AnimeAdapter | used as-is, zero-shot, no per-character training |
| Few-shot per-artist style LoRA | Opt-In Art | manga ink and screentone instead of paintings, on a pretrained base |
| Artist in the loop | Interactive Drawing Guidance | the system draws the character; the artist keeps composition and the head |
| Style vs identity conflict | ZipLoRA | identity and style enter at different points (adapter vs LoRA), tested rather than merged |

### 3.3 Conclusion

**The components are not novel; the combination, the per-artist method, and the evaluation
are.** Each piece exists somewhere in the literature, but always paired with text or skeleton
control, a single generic style, or multi-GPU training. Our contribution is a reproducible
recipe that brings sketch control, reference identity and per-artist style together on one
consumer GPU, validated across three visually different series. That is a defensible
contribution for a course project and should be stated plainly rather than overclaimed.

---

## 4. Evaluation plan

Both a baseline and an ablation, measured on the same 15 cases used in the POC.

### Baselines to compare against

| # | Baseline | Purpose |
|---|---|---|
| B0 | Prompt only, no ControlNet, no IP-Adapter | floor |
| B1 | ControlNet only | pose without identity |
| B2 | IP-Adapter only | identity without pose |
| **B3** | **ControlNet + IP-Adapter, no LoRA** | **our POC — the number to beat** |
| B4 | B3 + per-artist LoRA | the full proposed system |

B3 → B4 is the headline figure: *what does per-artist adaptation actually buy?*

### Metrics

| Criterion | Metric | Notes |
|---|---|---|
| **Identity** | DINOv2 / CLIP-I cosine similarity, reference vs generated character crop | crop with the same body box logic as preprocessing |
| **Style** | FID against held-out panels by the same artist | expect the big B3 → B4 drop; this is the LoRA's whole job |
| **Style (fine)** | screentone / line-density statistics vs held-out panels | catches "looks anime but isn't inked" — FID alone may not |
| **Pose fidelity** | OpenPose keypoint distance, input outline vs output | objective, and directly measures the ControlNet half |
| **Practical** | minutes to retouch an output to publishable, with vs without the tool | the only metric an artist cares about |
| **Human** | small forced-choice study: "which of these was drawn by X?" | strongest evidence, cheapest to run badly — design carefully |

### Ablations worth reporting

- `cn_scale` × `ip_scale` grid (already have POC data)
- `cn_end` early-stop effect on traced artifacts
- number of training panels vs style FID — answers "how much does a new artist need to
  provide?", which is the most practically useful curve in the whole project

---

## Sources

**Closest to us**
- [AnimeAdapter: A Modular Adapter for Appearance-Consistent Anime Character Generation](https://arxiv.org/html/2605.20237)
- [SketchKeyAnime: Reference-anchored Sparse Key-Sketch Animation Synthesis](https://arxiv.org/pdf/2606.19958)
- [Interactive Drawing Guidance for Anime Illustrations with Diffusion Model](https://arxiv.org/html/2507.09140v1)

**Manga generation**
- [DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](https://jianzongwu.github.io/projects/diffsensei/) · [arXiv](https://arxiv.org/html/2412.07589v1) · [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_DiffSensei_Bridging_Multi-Modal_LLMs_and_Diffusion_Models_for_Customized_Manga_CVPR_2025_paper.pdf)
- [MangaFlow: An End-to-End Agentic Framework for Controllable Story to Manga Generation](https://arxiv.org/html/2605.28173)
- [Manga Generation via Layout-controllable Diffusion](https://arxiv.org/html/2412.19303v1) · [project page](https://siyuch-fdu.github.io/MangaDiffusion)

**Manga line art and screentone**
- [MangaNinja: Line Art Colorization with Precise Reference Following](https://arxiv.org/abs/2501.08332) · [project page](https://johanan528.github.io/MangaNinjia/)
- [SKETCH2MANGA: Shaded Manga Screening from Sketch with Diffusion Models](https://www.researchgate.net/publication/385286880_SKETCH2MANGA_Shaded_Manga_Screening_from_Sketch_with_Diffusion_Models)
- [Screentone-Preserved Manga Retargeting](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.70096)
- [Manga filling style conversion with screentone variational autoencoder](https://dl.acm.org/doi/abs/10.1145/3414685.3417873)
- [Advancing Manga Analysis: Comprehensive Segmentation Annotations for the Manga109 Dataset](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_Advancing_Manga_Analysis_Comprehensive_Segmentation_Annotations_for_the_Manga109_Dataset_CVPR_2025_paper.pdf)

**Style personalization**
- [ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](https://arxiv.org/pdf/2311.13600) · [ziplora-pytorch (unofficial)](https://www.sourcepulse.org/projects/2179035)
- [Opt-In Art: Learning Art Styles Only from Few Examples](https://arxiv.org/html/2412.00176v3)
- [AC-LoRA: Auto Component LoRA for Personalized Artistic Style Image Generation](https://arxiv.org/html/2504.02231)
- [Meta-LoRA: Meta-Learning LoRA Components for Domain-Aware ID Personalization](https://arxiv.org/html/2503.22352v2)
- [LoRA.rar: Learning to Merge LoRAs via Hypernetworks](https://openaccess.thecvf.com/content/ICCV2025/papers/Shenaj_LoRA.rar_Learning_to_Merge_LoRAs_via_Hypernetworks_for_Subject-Style_Conditioned_ICCV_2025_paper.pdf)

**Pose-guided identity**
- [PHiD: Preserving human identity in pose-guided character animation](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015577)
- [Identity-Preserving Pose-Guided Character Animation via Facial Landmarks Transformation](https://arxiv.org/html/2412.08976v2)

**Commercial**
- [Clip Studio Paint 3D tools for comics](https://www.clipstudio.net/en/comics-manga/tool/3d/) · [AI Pose Scanner](https://www.celsys.com/en/topic/20190228)
