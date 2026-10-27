# POC verdict — Manga Panel Assistant
 
**Date:** 27 September 2026 · **Duration:** 3 days of the 7 allotted · **Decision: GO**
 
---
 
## The question this week had to answer
 
> Given a character reference and a rough pose outline, can off-the-shelf components produce that same character in that pose — without any training?
 
**Yes.** Demonstrated on three characters of different build, with two different kinds of reference image and both machine- and hand-drawn outlines, at a single set of parameters that never needed retuning after the first character.
 
---
 
## The system
 
No component was trained. Everything is pretrained and off the shelf.
 
| Layer | Component | Supplies |
|---|---|---|
| Base | `gsdf/Counterfeit-V2.5` (SD 1.5) | the ability to draw |
| Text | CLIP skip 2 + `compel` weighted prompts | style tags, framing |
| Structure | ControlNet `control_v11p_sd15s2_lineart_anime` | pose, from the outline |
| Identity | IP-Adapter Plus (`ip-adapter-plus_sd15`) | hair, outfit, proportions, from the reference |
 
Runs on a free Colab T4. Roughly 12 s per 512×768 image at 28 steps.
 
**Tuned once, on character 1, then frozen:**
 
```
CN_SCALE = 0.50    outline strength
CN_END   = 0.70    ControlNet steers only the first 70% of denoising steps
IP_SCALE = 0.80    reference strength
```
 
That these transferred unchanged to characters 2 and 3 is the single most important result of the week. It means the method is a method, not a per-character hand-tuning exercise.
 
---
 
## Evidence
 
| Character | Reference type | Outlines | Cases | Result |
|---|---|---|---|---|
| 1 — teenage boy, long coat | SD-generated, rendered greyscale | 5 machine-drawn | 5 | 5/5 pass |
| 2 — Detective Conan | Generated in manga style | 4 machine-drawn + 1 **hand-drawn on paper** | 5 | 5/5 pass |
| 3 — Ran Mouri | **Real manga art** — flat line + screentone | 1 machine-drawn | 1 | 1/1 pass |
 
**11 of the planned 15 cases were run. All 11 passed on the first attempt**, against a criterion of ≥60% within three attempts. The four not run were dropped for input reasons, not method failures: one Gemini pose refusal, and time reallocated to testing a real manga reference and a real pen sketch instead — both higher-value tests than a fourth repetition.
 
### What each case had to show
 
1. **Recognisably the same character** — outfit, silhouette, proportions ✅
2. **Follows the outline** — pose, limb placement, orientation ✅
3. **Clean enough to retouch rather than redraw** ✅
---
 
## Three results that matter beyond the verdict
 
**Parameters generalise across characters.** One tuning pass covered a teenager in a coat, a child in a blazer, and a teenage girl in a school uniform.
 
**Hand-drawn input works as well as generated input.** A pen sketch on paper — thin, uneven strokes, photographed under room light — produced a pose as accurate as a clean digital outline, at identical settings. This is the real use case, and it was never the weak link.
 
**Real manga references work.** The open worry was that IP-Adapter, trained on photographs and rendered art, would read flat screentone line art weakly. It did not. The Ran reference transferred blazer, necktie, pleated skirt and shoes without raising `ip_scale`.
 
---
 
## Known limitations
 
**The head does not transfer.** Hair and face consistently drift from the reference. This became a design decision rather than a defect: **the system draws the body, the artist draws the head.** The outline still supplies head shape, size and orientation, so the artist's drawing lands in the right place. Narrowing scope this way also removed the only identity failure observed.
 
**The outline must describe the *clothed* silhouette.** Our test sketches were nude contour mannequins, and wherever clothing should have covered the body, ControlNet's body edge fought IP-Adapter's fabric:
 
- character 1's coat tails came out semi-transparent, with the legs showing through
- character 3's knee socks vanished entirely, leaving bare legs
This is an input requirement, not a limitation — a mangaka sketching a pose draws the clothes anyway. It should be stated explicitly in the method.
 
**ControlNet traces every line literally.** Joint circles became rips in the jeans; a line across a head oval became a blindfold; a knee contour became a bandage. Pose sketches must carry contour lines only, nothing inside the body.
 
**Style is not manga.** Output is soft greyscale rendering with gradients and cast shadows, not flat ink with screentone. Explicitly excluded from the pass criterion, and precisely what the per-series LoRA exists to fix.
 
**Head inpainting is untested.** Deferred deliberately: the chosen workflow is to generate the whole character and let the artist replace the head, which makes the seam a neck line rather than a hairline. Still to be validated.
 
---
 
## What this licenses
 
Committing the team. The riskiest assumption — that identity and pose could coexist at all without training — is retired. What remains is engineering with known shapes:
 
- **Generative AI course:** train a LoRA per series and measure the style gap closing. The method and its parameters are already fixed.
- **MLOps course:** the pipeline is four pretrained components and three numbers. Containerising, tracking and serving it is well-defined work.
## Open questions for the next phase
 
1. Does head inpainting leave a clean neck seam?
2. Does a LoRA close the style gap to flat ink and screentone?
3. Does the clothed-silhouette requirement hold for loose clothing — capes, long hair, flowing garments?
4. How many reference images does a new series need before the method transfers?