# Dataset — artist #1, Spy × Family
 
Status: **done**. 270 training panels tagged, 77 held out for FID. Ready to train.
 
---
 
## Why Spy × Family and not Manga109
 
Manga109 is pre-2000 art: the LoRA would have to drag Counterfeit-V2.5 a long way,
and the burned-in Japanese text was the biggest risk in the whole plan.
 
Vagabond, bought and scanned, turned out to be the same problem in a new wrapper —
sumi-e brush, dense crosshatch, realistic adult anatomy, tilted and page-bleeding
panels, and English SFX redrawn straight over the artwork. **It is kept as artist #3**,
replacing `RinToSiteSippuNoNaka`: it is the furthest of the three from the checkpoint,
which is exactly what makes the transfer claim worth making. `configs/vagabond.yaml`
is stubbed with the specific ways the defaults are expected to misbehave there.
 
Spy × Family is the closest fit to the checkpoint: thin uniform digital line, flat
screentone, anime-standard faces, rectangular panel grids, a lot of full-body framing.
 
## The source files
 
| File | Pages | Page size | Language | Role |
|---|---|---|---|---|
| `spy x  family 2.pdf` | 204 | 2415×3440 | Thai | primary |
| `spy x family.pdf` | 151 | 1080×1688 | Spanish | secondary |
| Missions 77–80 | 81 | 784×1145 | English | **unused** |
 
PDFs live in `dev/ENG`, beside the repo — the `pdf:` paths in the series config are
relative to the repo root, hence `../ENG/...`.
 
Every page is a single embedded JPEG, so PyMuPDF pulls out the original scan bytes —
no re-rendering, no resampling.
 
**Mixed resolutions resolve themselves.** The `min_side` filter only passes panels that
will be *downscaled* to 512, so both volumes reach the trainer at comparable sharpness
instead of averaging to a soft line. The mission files fail that gate automatically —
their panels would need upscaling — so they need no manual exclusion.
 
Mixed languages do not matter: the lettering is erased either way, and no single script
dominating the residue is mildly helpful.
 
## Running it
 
```
pip install pymupdf opencv-python-headless onnxruntime huggingface_hub
python scripts/build_dataset.py --config spyxfamily.yaml     # ~90 s, CPU
python scripts/tag_dataset.py data/datasets/spyxfamily/10_sxfstyle sxfstyle
```
 
| Stage | What | Where |
|---|---|---|
| 0 | PDF → page JPEGs | `pipeline.extract_pages` (PyMuPDF) |
| 1 | page → panel boxes | `dataset/panels.py` |
| 2 | score + filter | `dataset/text.py` |
| 3 | erase lettering | `dataset/clean.py` |
| 4 | write kohya folder | `pipeline.emit` |
| 5 | WD14 tags + trigger | `dataset/tagger.py` |
 
Nothing in the chain needs torch. `--sweep` re-applies thresholds to the cached scan
without re-reading pages; `--sheets DIR` writes contact sheets.
 
Output:
 
```
data/datasets/spyxfamily/10_sxfstyle/     270 .png + .txt   <- train_data_dir is the parent
data/datasets/spyxfamily_holdout/          77 .png + .txt   <- FID, never trained on
```
 
The holdout sits *outside* the dataset folder on purpose: kohya reads every subfolder
of `train_data_dir` as a `<repeats>_<name>` concept, so a `holdout/` inside it would
be picked up or warned about.
 
## Measured result
 
355 pages → 2864 panel candidates → **363 kept** → 347 after lettering removal
→ 270 train + 77 holdout.
 
| Rejected by | Count |
|---|---|
| smaller than 448 native | 1554 |
| too much lettering (>0.30) | 434 |
| oversized type: logos and drawn SFX (>0.02) | 247 |
| sliver aspect | 124 |
| blank or solid black | 23 |
| lettering sat on artwork, unerasable | 16 |
 
Threshold sensitivity, if more panels are needed:
 
| min_side | text≤0.25 | text≤0.30 | text≤0.35 |
|---|---|---|---|
| 512 | 246 | 323 | 382 |
| **448** | 284 | **367** | 433 |
| 416 | 299 | 388 | 464 |
| 384 | 323 | 414 | 495 |
 
## What the captions say about the cleanup
 
WD14 tags are a free audit of stage 3. Across the 270 training panels:
 
| Tag | Count | Reading |
|---|---|---|
| monochrome / greyscale | 270 / 268 | as expected |
| comic | 206 | panel borders are visible — fine |
| `english text` | 121 | **45% still carry readable lettering** |
| `speech bubble` | 118 | empty bubbles left behind, by design |
| `1girl` / `1boy` | 126 / 117 | plenty of single-character framing |
 
That 45% is the honest measure of residual lettering, and it is higher than the
contact sheets suggested. It is not fatal: the tag is *in the caption*, so the model
attributes the text to `english text` rather than to `sxfstyle`, and omitting the tag
at inference suppresses it. Character names (`anya (spy x family)`, `yor briar`) do the
same job for identity, which is what keeps a style-only LoRA from absorbing faces.
 
If the trained LoRA still scrawls pseudo-text, the fix is a stricter `max_text`
(0.25 costs ~80 panels) or a trained text detector, not a prompt change.
 
## Three findings worth not rediscovering
 
**1. Bubble detection by white-region enclosure does not work on this material.**
The obvious approach — a speech bubble is a white region fenced in by its outline —
fails twice over: in line art *every* shape is a closed white region (a face, a sleeve,
a door), and much of the dialogue here is set straight onto white with no outline at
all. Detect the *lettering* instead — small ink blobs of near-uniform height that
cluster into lines — then grow it into the block it occupies.
 
**2. Erase the strokes, not the block.** Painting out a text block's bounding box
punches white holes through faces wherever the detector misfires. The rule that works
is two-tier: where the collar around the block is pale too (inside a bubble, or set on
open white) take the whole block; otherwise take only the glyph strokes, so a false
positive costs a few white specks instead of a hole.
 
**3. Narration boxes are white-on-black.** Run the same detector on the inverted image
and fill with the block's own ground, not always with white.
 
## Environment traps (all hit once, all fixed)
 
- **No `pdfimages` on Windows.** It is poppler, a Linux toolkit. Stage 0 uses PyMuPDF,
  which extracts the same embedded bitmap and runs anywhere, Colab and Kaggle included.
- **No `bash` on Windows.** `tag_dataset.sh` is replaced by `tag_dataset.py`.
- **Windows Smart App Control blocks `optree/_C.pyd`.** Any sd-scripts entry point
  imports `library.dataset` → diffusers → `torch._dynamo` → optree, and the import dies
  on a blocked DLL. Tagging no longer touches sd-scripts at all: `dataset/tagger.py`
  runs the WD14 ONNX model on onnxruntime. **Training cannot dodge it** — see below.
- **The WD14 ONNX export pins its batch dimension to 1.** The tagger reads the batch
  size off the model rather than assuming it is dynamic.
- **Never `pip install -r sd-scripts/requirements.txt`.** It pins an older torch and
  would replace the cu128 build, the only one with sm_120 kernels for the RTX 5060.
  The same trap applies on Kaggle, where it would break the preinstalled CUDA torch.
- **An interrupted build leaves a partial dataset silently.** Once produced 102 of 363
  panels with no error. `emit` now prints progress and warns on a short count.
## The Smart App Control problem
 
Smart App Control blocks unsigned native extensions by file reputation. It stopped
`optree/_C.pyd`, which torch imports on the way into `torch._dynamo`. The dataset
pipeline sidesteps it; kohya's trainer cannot.
 
**Decision: train on Kaggle.** It was already the reference environment, so this costs
nothing and settles the question permanently. `notebooks/01_train_lora_kaggle.ipynb`
is the runner. Turning Smart App Control off is one-way (re-enabling needs a Windows
reinstall) and is not worth it for a side-research machine; `pip install
--force-reinstall --no-cache-dir optree` is worth one try if local training ever
becomes necessary.
 
## Known defects
 
- **Under-splitting.** Some panels are whole-page blocks. Border-line cutting fixed most
  of it; roughly 10% remain.
- **Borderless panels shatter.** Open panels on white with sparse figures split into
  fragments that then die on size, so the real panel is lost. Costs yield, not quality.
- **Residual lettering in ~45% of panels** (measured above). A trained detector
  (Magi, `ragavsachdeva/magi`) would fix this and the under-splitting together; it was
  not used because it needs a GPU and the HF download is blocked in the build sandbox.

---

## Update 11 Oct — dataset v2: Magi character crops

The residual lettering (45%) and empty bubbles showed up in B4 as ghost circles and
name-tag boxes on clothing. Magi (`ragavsachdeva/magiv2`, GPU, Kaggle) now supplies
character and text boxes; everything after detection runs locally on CPU.

```
Kaggle: Magi on 355 pages        -> data/datasets/magi_v3.json   (boxes only, ~40 min GPU)
local:  scripts/crops_from_magi.py -> data/datasets/spyxfamily_v2/1_sxfstyle   (~2 min CPU)
        scripts/tag_dataset.py     -> WD14 captions, trigger first
        drop captions with `no humans` / `close-up` / `eye focus`
        = 647 crops, + the 270 panels as 2_sxfstyle
```

Crop rules: character box >= 256 px, aspect 0.5-2, another character covers <= 15%,
5% margin, **no square pad and no center-crop** (kohya buckets handle aspect; center-crop
cut heads and feet), downscale so the long side <= 1024, drop the crop if text boxes cover
> 10% of it.

### Findings worth not rediscovering

1. **Key Magi output by `volume/page`.** Both volumes have `p-0005.jpeg`: keyed by page
   name alone, 50 crops overwrote each other and the JSON kept 203 of 355 pages.
2. **Never fill a text box with white.** Magi text boxes overlap hair and black clothes;
   filling them punched white rectangles through ~258 faces.
3. **Erase only ink blobs that lie entirely inside a text box**, found on the whole page.
   Letters fit inside and go; hair or a coat crossing the box sticks out and survives. An
   earlier rule (blob shorter than 60% of the box) kept whole words like "FINAL", because a
   tight box is as tall as its letters.
4. **Save the JSON.** Re-cropping from it is a 2-minute CPU job; re-running Magi is a GPU
   session.

---

## Update 26 Oct — one command per series

```
python scripts/prepare_series.py --config bungo.yaml            # Kaggle: notebooks/02
python scripts/prepare_series.py --config bungo.yaml --limit 20 --until panels   # smoke test, CPU
```

pages (PDF **or** chapter folders, natural order) → panels → Magi (GPU) → crops → WD14 tags →
drop `no humans` / `close-up` / `eye focus` → `data/datasets/<series>_report.json` (counts per
stage, git commit, resolved config). Each stage is skipped when its output exists; `--from STAGE`
redoes it and everything after. Magi writes `magi.json` every 25 pages, so a dead Kaggle session
resumes. All thresholds live in `configs/dataset.yaml` (`crops:`, `tag:`), none in scripts.

Output for a series:

```
data/datasets/<series>/2_<trigger>/          panels       (repeats 2)
data/datasets/<series>/1_<trigger>_crops/    Magi crops   (repeats 1)
data/datasets/<series>_holdout/              panels for FID, never trained on
```

**Regression check** on Spy × Family's `magi_v3.json`: 708 crops, containing all 647 of dataset v2,
40/40 sampled pixel-identical. The 891 in `sxf_crops/` predate the `max_text` rule.

Retired (in `_to_delete/scripts/`): `magi_crops.py` (Kaggle cell, box-whitening — the rule that
holed 258 faces), `crops_from_magi.py` (ported into `dataset/crops.py`), `build_characters.py` +
`dataset/characters.py` (box-whitening again, never used), `tag_dataset.sh`.

**Bungo smoke test (120 pages):** 738 candidates → 62 panels (45 train / 17 holdout). Extrapolates
to ~1400 panels from 2680 pages. Panels on this series carry visible English lettering and some
under-splitting; the Magi crops are expected to be the cleaner half.

### Bubble erase (26 Oct, `crops.erase_bubbles`)

Letters were erased but empty bubble outlines stayed — the source of B4's ghost circles. Now the
white region under each Magi text box is whitened with a ring for its outline, **only if** that
region stays within 1x the text box on each side. Text set on open panel white (common in Bungo)
spreads to the whole panel and is left alone.

Bungo smoke test: 60 of 125 crops changed, closed bubbles gone. Left: bubbles that open onto the
panel border, and bubbles whose text Magi missed. Sealing gaps by thickening ink first did not help
(the leaks are open bubbles, not JPEG gaps). Off for dataset v2 (Spy × Family), on from Bungo.

