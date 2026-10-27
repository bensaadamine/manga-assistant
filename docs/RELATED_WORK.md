# Related work and positioning
 
Literature survey for the "what exists already / what is our added value" requirement.
Compiled 30 September 2026.
 
---
 
## 1. The closest academic work
 
### DiffSensei — customized manga generation (CVPR 2025)
 
The single most relevant paper. Generates **whole manga pages** with multi-character control, combining a diffusion generator with a multimodal LLM acting as a "text-compatible identity adapter." Takes text plus character reference images; controls layout, character appearance and character actions. Trained on **MangaZero**, a purpose-built dataset of 43,264 pages and 427,147 annotated panels.
 
**Where it differs from us:** DiffSensei generates *pages* from *text*. The author describes a scene and the system composes it. Our system takes a **drawing** — the artist's own pose sketch — and fills it in. DiffSensei replaces the artist's composition decisions; we execute them. It is also trained once across many series, not adapted per artist.
 
### MangaNinja — line art colorization with precise reference following (CVPR 2025)
 
Colorizes line art by following a reference image precisely, using patch shuffling and point-driven control to handle large pose differences between reference and target.
 
**Where it differs:** it takes *finished line art* and adds color. We take a *rough pose outline* and produce the drawing. Complementary, not competing — arguably a later stage of the same production pipeline.
 
### Sketch2Manga — shaded manga screening from sketch
 
Generates screentone-shaded manga from a sketch using diffusion. Closest thing to our style layer.
 
**Where it differs:** style-only, no character identity. It doesn't answer "is this the same character."
 
### Screentone-preserved manga retargeting / screentone VAE (Xie et al.)
 
A line of work on generating and preserving screentone specifically. Directly relevant to our biggest open limitation — output is soft greyscale, not flat ink and screentone. **Read before designing the LoRA evaluation.**
 
### Manga109 segmentation annotations (CVPR 2025)
 
Adds segmentation masks to Manga109 beyond the existing frame/body/face/text boxes. Worth checking — if available, it makes our character-cropping preprocessing considerably cleaner than bounding boxes.
 
### Identity-preserving pose-guided character animation (PHiD, AniFaceDiff, etc.)
 
A cluster of 2024–2026 work on keeping a person's identity while changing pose. Almost all of it targets **photorealistic humans and video**. The techniques transfer conceptually; nobody in that cluster works on black-and-white manga line art with an artist in the loop.
 
### AC-LoRA, Meta-LoRA, LoRA.rar
 
Personalized-style LoRA methods: automatic component selection, meta-learned initialization, merging subject and style LoRAs via hypernetworks. Useful **method sources for our phase 2**, not competitors — we can cite them as the basis for how we train per-artist.
 
---
 
## 2. What exists commercially
 
| Tool | What it does | Gap |
|---|---|---|
| **Clip Studio Paint** 3D model posing + AI Pose Scanner | Pose a 3D mannequin, or extract a pose from a photo and apply it | Gives a *reference* to draw over. Does not draw the character. No style learning. |
| **NovelAI / niji·journey** and similar | High-quality anime generation from text | Generic anime style. No per-artist adaptation, no outline control, no identity guarantee across images. |
| **ControlNet + IP-Adapter workflows** (A1111, ComfyUI) | Exactly our POC stack, assembled by hobbyists | This is our *baseline*, not a competitor. No per-artist LoRA method, no reproducible recipe, no evaluation. |
 
---
 
## 3. Our positioning
 
Nobody in the surveyed work does all three of these at once:
 
1. **Per-artist adaptation as a reproducible method** — a documented recipe that takes one artist's pages and produces a working assistant for that artist, validated across three visually different series. The existing work trains one model for all manga.
2. **The artist's own drawing as the control signal** — a rough pose sketch, not a text prompt. The artist keeps composition and staging; the system does the labour.
3. **An explicit human-in-the-loop boundary** — the system draws the body, the artist draws the head. Nobody else defines a boundary; they all aim for full automation and accept identity drift as a quality problem.
The honest framing: **the components are not novel; the method, the per-artist framing, and the evaluation are.** That is a defensible contribution for a course project and should be stated plainly rather than overclaimed.
 
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
- number of training panels vs style FID — answers "how much does a new artist need to provide?", which is the most practically useful curve in the whole project
---
 
## Sources
 
- [DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](https://jianzongwu.github.io/projects/diffsensei/) · [arXiv](https://arxiv.org/html/2412.07589v1) · [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_DiffSensei_Bridging_Multi-Modal_LLMs_and_Diffusion_Models_for_Customized_Manga_CVPR_2025_paper.pdf)
- [MangaNinja: Line Art Colorization with Precise Reference Following](https://arxiv.org/abs/2501.08332) · [project page](https://johanan528.github.io/MangaNinjia/)
- [SKETCH2MANGA: Shaded Manga Screening from Sketch with Diffusion Models](https://www.researchgate.net/publication/385286880_SKETCH2MANGA_Shaded_Manga_Screening_from_Sketch_with_Diffusion_Models)
- [Screentone-Preserved Manga Retargeting](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.70096)
- [Manga filling style conversion with screentone variational autoencoder](https://dl.acm.org/doi/abs/10.1145/3414685.3417873)
- [Advancing Manga Analysis: Comprehensive Segmentation Annotations for the Manga109 Dataset](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_Advancing_Manga_Analysis_Comprehensive_Segmentation_Annotations_for_the_Manga109_Dataset_CVPR_2025_paper.pdf)
- [PHiD: Preserving human identity in pose-guided character animation](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015577)
- [Identity-Preserving Pose-Guided Character Animation via Facial Landmarks Transformation](https://arxiv.org/html/2412.08976v2)
- [AC-LoRA: Auto Component LoRA for Personalized Artistic Style Image Generation](https://arxiv.org/html/2504.02231)
- [Meta-LoRA: Meta-Learning LoRA Components for Domain-Aware ID Personalization](https://arxiv.org/html/2503.22352v2)
- [LoRA.rar: Learning to Merge LoRAs via Hypernetworks](https://openaccess.thecvf.com/content/ICCV2025/papers/Shenaj_LoRA.rar_Learning_to_Merge_LoRAs_via_Hypernetworks_for_Subject-Style_Conditioned_ICCV_2025_paper.pdf)
- [Clip Studio Paint 3D tools for comics](https://www.clipstudio.net/en/comics-manga/tool/3d/) · [AI Pose Scanner](https://www.celsys.com/en/topic/20190228)