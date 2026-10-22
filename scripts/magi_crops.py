# Kaggle cell — Magi character crops, second attempt.
#
# What went wrong the first time, visible in the 1425 crops:
#   panel borders and neighbouring panels inside the frame, speech text and SFX
#   still present, two characters in one crop, and extreme close-ups (an eye, a
#   mouth, a hand) that teach nothing about figure style.
#
# Six rules fix all four. Rules 2 and 3 are the ones that need the JSON, which is
# why it gets saved this time.

import json, os, glob
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel

PAGES   = sorted(glob.glob("/kaggle/input/<your-dataset>/**/*.jpg", recursive=True))
OUT     = "/kaggle/working/crops"
JSON    = "/kaggle/working/magi.json"
MIN_PX  = 256      # rule 4: shortest side of the character box
MARGIN  = 0.05     # rule 1: breathing room, not a square pad
OVERLAP = 0.15     # rule 3: max share of the box covered by another character
ASPECT  = (0.5, 2.0)   # rule 5: kills eye-slits and tall strips

os.makedirs(OUT, exist_ok=True)
model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()


def iou_frac(a, b):
    """Share of box a that box b covers."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


kept = dropped = 0
records = {}

for n, path in enumerate(PAGES):
    page = Image.open(path).convert("L")
    W, H = page.size
    with torch.no_grad():
        res = model.predict_detections_and_associations([np.array(page.convert("RGB"))])[0]

    chars = [list(map(float, b)) for b in res["characters"]]
    texts = [list(map(float, b)) for b in res["texts"]]
    records[os.path.basename(path)] = {"characters": chars, "texts": texts}

    for i, box in enumerate(chars):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0

        if min(w, h) < MIN_PX:                                   # rule 4
            dropped += 1; continue
        if not (ASPECT[0] <= w / max(h, 1) <= ASPECT[1]):        # rule 5
            dropped += 1; continue
        if any(iou_frac(box, o) > OVERLAP                        # rule 3
               for j, o in enumerate(chars) if j != i):
            dropped += 1; continue

        mx, my = w * MARGIN, h * MARGIN                          # rule 1
        cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
        cx1, cy1 = min(W, int(x1 + mx)), min(H, int(y1 + my))

        crop = np.array(page.crop((cx0, cy0, cx1, cy1)))
        for t in texts:                                          # rule 2
            tx0, ty0 = int(max(t[0], cx0) - cx0), int(max(t[1], cy0) - cy0)
            tx1, ty1 = int(min(t[2], cx1) - cx0), int(min(t[3], cy1) - cy0)
            if tx1 > tx0 and ty1 > ty0:
                crop[ty0:ty1, tx0:tx1] = 255

        im = Image.fromarray(crop)                               # rule 6
        s = 512 / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        l, t_ = (im.width - 512) // 2, (im.height - 512) // 2
        im.crop((l, t_, l + 512, t_ + 512)).save(
            f"{OUT}/{os.path.splitext(os.path.basename(path))[0]}_ch{i:02d}.png")
        kept += 1

    if n % 40 == 0:
        print(f"{n}/{len(PAGES)}  gardes={kept}  rejetes={dropped}")

json.dump(records, open(JSON, "w"))
print(f"\nTERMINE : {kept} crops, {dropped} rejetes")
print(f"JSON -> {JSON}   (garde-le, il evite de tout relancer la prochaine fois)")

import shutil
shutil.make_archive("/kaggle/working/sxf_crops_v2", "zip", OUT)
