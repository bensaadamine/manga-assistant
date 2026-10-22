# Input contract — what the system can do, and what its two inputs must look like

Companion to `README.md` (how the POC works), `POC_VERDICT.md` (what was proven) and
`DESIGN_SPACE.md` (component choices). Opened 9 October 2026, last amended 15 October.

**Append a row to §5 after each run.**

---

## 1. Capabilities and limits

**Can do**

- One character, body only, black and white
- Front or three-quarter front views
- Any framing: full body, waist-up, shoulders-up
- Pose follows the drawn outline closely
- Clothing and build transfer from the reference
- A photographed pen sketch works as well as a digital file

**Cannot do**

- Back views and turned poses — produces a two-headed figure
- More than one character
- Backgrounds, props, objects in hands
- Faces and hair — out of scope by design, the artist draws the head
- Flat ink on white paper — currently comes out grey and soft

**Known broken, 15 October**

- Grey background instead of white. The largest measured gap against real panels,
  and it is present without the LoRA, so it belongs to the base checkpoint.
- The LoRA makes style worse, not better. Undertrained — see `V2_RESULTS.md`.
- Male characters drift feminine on narrow-shouldered outlines.

## 2. Reference image

| # | Rule | Why | Source |
|---|---|---|---|
| R1 | One character, nothing else in frame | IP-Adapter copies everything — props, background, ink splatter | POC #6 |
| R2 | **Empty hands** | A held object transfers as floating artifacts. `clean_ref()` whitens the background only; it cannot remove a prop | POC #6, case L1 |
| R3 | Full body preferred, never cropped above mid-thigh | Preference, not a hard rule — M1 passed with a mid-thigh crop | case L1, M1 |
| R4 | Front or 3/4 front view | Reference orientation fights an outline of opposite orientation | case L1 |
| R5 | Flat line + screentone **or** SD-rendered both work, same `ip_scale` | Verified on real manga art | POC #8 |
| R6 | Level of line detail is not a lever | Detail is never the missing ingredient; cleanliness is | case L1 |

**Not required:** colour, high resolution, matching the target pose.

## 3. Pose outline

| # | Rule | Why | Source |
|---|---|---|---|
| O1 | **Contour lines only** — nothing inside the body | ControlNet traces every line literally: joint circle → ripped jeans, line across head oval → blindfold | POC #3 |
| O2 | No garment interior **that the reference does not have** | Refined by M1: a jean seam drawn in the outline traced correctly because the reference has that seam. The rule is "no interior line that contradicts the reference", not "no interior line ever" | cases L1, M1 |
| O3 | Draw the **clothed** silhouette, not the nude body | ControlNet's body edge beats IP-Adapter's fabric: coat tails go transparent, knee socks vanish | POC #2 |
| O4 | Head is an **empty oval** — no line across it, no features | POC #3 |
| O5 | Proportions come from the outline | Nothing downstream fixes an adult body on a child | POC |
| O6 | Separation lines only where a limb or garment panel overlaps another | Allowed exception to O1 |
| O7 | Full figure, centred, ~80% of frame height, vertical 2:3 | Cropped or off-centre framing compounds R3 | case L1 |
| O8 | **Avoid back and turned views** until V2b | Model prior and reference are both front-facing; a back silhouette collides with both | §2.6, cases B1 + L1 |

**Not required:** clean digital lines. A photographed pen sketch performed identically,
first attempt (POC #7).

## 4. Failure → cause lookup

| Symptom | Input at fault | Fix |
|---|---|---|
| Sky, scenery or a panel border in the output | R1 — reference is an uncropped manga panel | Whiten and crop the reference; negative-prompt the panel border |
| Colour tint on skin or hair | prompt, not input | `(monochrome:1.3)` plus `high contrast` |
| Flat grey field instead of white paper | prompt **and** base model | See §6; a bare `background` token in the negative cancels the positive white-background token |
| Two heads, face on a turned-away body | O8 | Redraw front or 3/4 |
| Garment shatters into flat overlapping planes | O2 | Strip interior lines the reference does not have |
| Floating translucent shapes near the figure | R2 | Crop the prop out, or pick another reference frame |
| Clothing missing, body shows through | O3 | Redraw as the clothed silhouette |
| Limb feature that was never intended (rip, band, blindfold) | O1 / O4 | Remove the interior line |
| Character drifts to generic | reference too small or too busy | Cleaner reference before touching `ip_scale` |
| Proportions wrong | O5 | Fix the outline |
| Hair / face wrong | **not a defect** | Out of scope — the artist draws the head |

## 5. Case log

| ID | Date | Character | Outline | Reference | Outcome | Rules broken |
|---|---|---|---|---|---|---|
| B1 | 9 Oct | — | back view | — | two-headed figure | O8 |
| Y1 | 9 Oct | Yor | front 3/4, contour-only, clothed, full figure | real manga line art, full-ish body, empty hands | **good** — identity, garment and pose all held | none |
| L1 | 9 Oct | Loid | back view, raised arm, collar/lapel/seams/folds drawn, arm exits frame | SD-rendered, cropped mid-thigh, **object in hand** | **bad** — ghosted double figure, suit in flat planes, floating circles | O2, O7, O8, R2, R3, R4 |
| M1 | 9 Oct | male, t-shirt + jeans | front, contour + jean seams, clothed, full body, centred | real manga panel, cropped mid-thigh, hands in pockets, **background present** | **good** — pose exact, garment and build transferred | R1 only |
| — | 10 Oct | 9 cases, 3 characters | clean, front, contour-only | three new full-body white-background sheets | measured, not eyeballed — see `V2_RESULTS.md` | none |

**Reading of L1:** four independent input violations, no evidence of a model fault.
Counterfeit gender bias was considered and rejected — POC character 1 was a teenage boy
in a long coat, 5/5 pass. **M1 settles it**: same pipeline, same parameters, a male
character and a clean outline produced a pass on the first attempt.

## 6. Prompt override, 10 October

`base.yaml`'s negative prompt carried a bare `background` token. It cancels every
background concept including the positive `plain white background`, the two neutralise,
and the model fills the field with flat grey. The override lives in
`configs/spyxfamily_eval.yaml` so `base.yaml` stays frozen; `config._deep_merge`
replaces only the three keys.

| Key | Before | After |
|---|---|---|
| `negative` | `… text, 2girls, background, speed lines …` | bare `background` **removed**; `grey background` raised to 1.6; added `gradient background`, `grey backdrop`, `flat grey fill`, `washed out`, `low contrast`; `cropped` removed (it fought `cropped at mid-thigh`) |
| `background` | `(plain white background:1.5), simple background` | `(white background:1.6), simple background, no shading behind the figure` |
| `style` | `(monochrome:2), (greyscale:1.6)` | `(monochrome:1.3), (greyscale:1.3)` plus `high contrast` |

`plain white background` is not a real danbooru tag; `white background` is. Weight 2 on
`monochrome` produces a uniform grey wash rather than ink.

It helped but did not solve it: B3's `white_ratio` went to 0.069 against 0.672 for the
artist's own panels.

## 7. Prompt for generating outlines

```
Generate a rough hand-drawn pose sketch, as if quickly drawn by a manga
artist with a black ballpoint pen on plain white paper.

SUBJECT: <pose, stated as FRONT or THREE-QUARTER FRONT>

WHAT TO DRAW:
- The outer contour of the CLOTHED body only.
- Separation lines only where one limb or garment panel overlaps another.
- The head as an EMPTY oval. No line across it, no features inside it.

WHAT NOT TO DRAW:
- No fold lines, no seams, no collar lines, no cuff bands, no buttons.
- No joint circles or any marks inside the body. Contour lines only.
- No closed shape on the torso: no ribcage oval, no chest band, no waistline.
- No facial features, no hair, no shading.
- No background, floor, furniture, props, shadow, text, or second figure.

LINE QUALITY: thick, slightly uneven, sketchy; some strokes overshoot.
BACKGROUND: pure white paper, nothing else.
FRAMING: centered, FULL BODY, figure fills ~80% of frame height, vertical 2:3.
```

Add a proportion clause for non-adults, e.g. `a child about 7 years old, large head
relative to body, short limbs`.

## 8. The third input question — settled 10 October

The eval config carries `description` and `text` per pose, which looked like two extra
inputs beyond the reference and the outline. Neither is.

- **`description`** is per **character**, not per outline. Ten new poses of Yor share one
  description. It now lives once under `characters:` in the config; a pose names which
  character it is and nothing more.
- **`text`** was ablated with `--no-pose-text`. Eight of nine cases were indistinguishable
  with and without it — ControlNet carries the pose on its own. The one exception was the
  **seated** pose: without the sentence the model read it as standing, because a contour
  alone cannot say "seated" when no chair is drawn.

**Conclusion:** adding a pose costs `file` + `character`, plus a short tag only when the
situation is ambiguous. There is no third input.
