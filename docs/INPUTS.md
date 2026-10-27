# Input guide — what to give the system

For the mangaka. You give **two images and one line of text**; the system draws the body,
you draw the head. Last updated 22 October 2026.

---

## 1. What the system can and cannot do

| Can | Cannot |
|---|---|
| One character, black and white | Two or more characters |
| Front or three-quarter front views | Back views, turned-away poses → two-headed figure |
| Full body, waist-up, shoulders-up | Backgrounds, props, objects in hands |
| Follow your outline's pose closely | Face and hair — **you draw the head** |
| Transfer clothing and build from the reference | Pure flat ink on white — output is still greyish and soft |
| Accept a phone photo of a pen sketch | |

## 2. The reference — *who* to draw

One image of the character, used for identity and clothing.

**Do**
- One character alone, on a plain background
- Full body (or at least down to mid-thigh)
- Front or three-quarter front
- Empty hands
- Either real manga art or a clean render — both work

**Don't**
- Props, bags, weapons, held objects — they reappear as floating shapes
- Scenery, panel borders, speech bubbles, other characters
- A view facing the opposite way from your outline

## 3. The outline — *how* to pose them

A rough line drawing of the pose. The system traces **every line literally**.

**Do**
- Draw the **outer contour of the clothed body** (coat tails, skirt, sleeves included)
- Leave the head as an **empty oval** — it sets the head's size and angle
- Add a line inside the body **only** where a limb crosses the body or another limb
- Get the proportions right — a child outline gives a child, an adult outline gives an adult
- Centre the figure, about 80% of the frame height, portrait format

**Don't**
- Joint circles, ribcage, waistline, muscle lines → they turn into rips, bands, belts
- Any line across the head → becomes a blindfold
- Folds, seams or buttons the reference doesn't have
- The nude body under the clothes → clothing goes transparent
- Back views
- Floor, chair, shadow, background

## 4. The text — one short line, optional

- The **character description** is written once per character (a list of tags: hair, clothing).
  You don't rewrite it per pose.
- A **pose line** is needed **only when the outline is ambiguous** — e.g. "seated on a stool",
  because a contour without a chair reads as standing.
- No colour words ("red ribbon"); describe shape instead.

## 5. When the output is wrong

| You see | Cause | Fix |
|---|---|---|
| Two heads, face on a turned body | back or turned outline | redraw front or 3/4 |
| Rips, bands, blindfold | line inside the body or head | erase it |
| Clothing see-through | outline drawn as nude body | redraw clothed silhouette |
| Floating translucent shapes | prop in the reference | crop it out or change reference |
| Scenery or panel border | busy reference | cut the character out onto white |
| Garment split into flat planes | interior lines the reference lacks | remove them |
| Seated pose comes out standing | ambiguous contour | add the pose line |
| Wrong proportions | outline | fix the outline |
| Face or hair wrong | **not a defect** | you draw the head |

## 6. Generating an outline with an image model

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

For a child, add: `a child about 7 years old, large head relative to body, short limbs`.

---

## Appendix — evidence behind the rules

| Rule | Evidence |
|---|---|
| Contour only, empty head oval | POC finding 3: joint circle → ripped jeans, head line → blindfold |
| Clothed silhouette | POC finding 2: coat tails transparent, knee socks vanish |
| Same facing direction | POC finding 11; case B1 (back view → two heads) |
| No props in reference | POC finding 6; case L1 (held object → floating circles) |
| Interior lines only if the reference has them | case M1: jean seams traced correctly because the reference had them |
| Pen sketch works | POC finding 7 |
| Real manga reference works | POC finding 8 |
| Pose line only when ambiguous | 10 Oct ablation: 8/9 poses identical without it; seated failed |

**Case log**

| ID | Date | Outcome | Rules broken |
|---|---|---|---|
| B1 | 9 Oct | two-headed figure | back view |
| Y1 | 9 Oct | good | none |
| L1 | 9 Oct | ghosted double figure, flat-plane suit, floating circles | back view, interior lines, cropped ref, prop in hand |
| M1 | 9 Oct | good — pose exact, garment and build transferred | busy reference only |
| eval | 10–22 Oct | 9 cases, 3 characters, measured — see `V2_RESULTS.md` | none |

Prompt override history (bare `background` token in the negative caused the grey field):
see `V2_RESULTS.md` and `configs/spyxfamily_eval.yaml`. Previous version of this file:
`_to_delete/INPUTS_old_1022.md`.
