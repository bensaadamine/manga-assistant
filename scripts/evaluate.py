"""Measure one baseline on the eval cases and append a row to the metrics log.

    python scripts/evaluate.py --config spyxfamily_eval.yaml --baseline b3
    python scripts/evaluate.py --config spyxfamily_eval.yaml --baseline b4 \
        --lora models/loras/spyxfamily/spyxfamily_style.safetensors
    python scripts/evaluate.py --compare            # print the log, no generation

Writes individual PNGs (run_poses.py only saves a grid, which cannot be measured)
to outputs/eval/<run_id>/ and one row per run to outputs/metrics.csv.

The style number is an ink profile compared against the ARTIST'S OWN held-out
panels, so it is a distance to a real target rather than an arbitrary statistic.
FID is reported only when both sides have enough images to mean anything; with 9
cases and n=2 it does not, and a printed FID over 18 images would be a lie.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from manga import evaluate, generate, pipeline, prompts, sketches
from manga.config import load

METRICS = Path("outputs/metrics.csv")
FID_MIN = 50          # cleanfid is unreliable below roughly this many images per side

FIELDS = [
    "run_id", "timestamp", "baseline", "lora", "lora_scale", "seed", "n", "cases",
    "identity", "style_gap", "black_ratio", "midtone_ratio", "edge_density",
    "white_ratio", "pose_dist", "pose_kps", "fid", "note",
]


def generate_all(cfg, models, args, out_dir: Path) -> list[dict]:
    """One row per generated image, with its file path."""
    rows = []
    for idx, pose in enumerate(cfg.poses):
        char = cfg.characters[pose["character"]] if "character" in pose else cfg.character
        control = sketches.load_sketch(Path(cfg.paths.outlines) / pose["file"])
        ref_path = pose.get("reference", char["reference"])
        reference = sketches.load_reference(ref_path) if args.baseline != "b1" else None
        prompt = prompts.build(cfg, pose.get("text", ""),
                               pose.get("description", char["description"]))
        negative = prompts.negative(cfg, char.get("negative_props", ""))

        images = generate.outline_to_image(
            models, cfg, prompt, negative, control, reference, n=args.n
        )
        for j, im in enumerate(images):
            p = out_dir / f"{idx:02d}_{Path(pose['file']).stem}_s{j}.png"
            im.save(p)
            rows.append({"path": p, "image": im, "control": control,
                         "reference": reference, "case": pose["file"]})
        print(f"  {pose['file']:22} {len(images)} image(s)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="spyxfamily_eval.yaml")
    ap.add_argument("--baseline", default="b3", choices=["b1", "b3", "b4"])
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--holdout", default=None, help="default: <paths.datasets>/<id>_holdout")
    ap.add_argument("--pose", action="store_true", help="also run OpenPose (slow, noisy)")
    ap.add_argument("--note", default="", help="free text stored with the row")
    ap.add_argument("--compare", action="store_true", help="print the log and exit")
    args = ap.parse_args()

    if args.compare:
        print_log()
        return

    if args.baseline == "b4" and not args.lora:
        ap.error("--baseline b4 needs --lora PATH")

    cfg = load(args.config)
    holdout = Path(args.holdout or f"{cfg.paths.datasets}/{cfg.character.id}_holdout")

    run_id = f"{args.baseline}_{datetime.now():%m%d_%H%M}"
    out_dir = Path("outputs/eval") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_id}] target ink profile from {holdout}")
    target = evaluate.ink_target(holdout)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in target.items()))

    models = pipeline.load(cfg)
    if args.lora:
        pipeline.load_lora(models, args.lora, scale=args.lora_scale)

    print(f"[{run_id}] generating")
    rows = generate_all(cfg, models, args, out_dir)

    print(f"[{run_id}] scoring")
    scorer = evaluate.IdentityScorer()
    ident, gaps, inks, poses, kps = [], [], [], [], []
    per_case = []
    for r in rows:
        stats = evaluate.ink_statistics(r["image"])
        gap = evaluate.style_gap(stats, target)
        row = {"case": r["case"], "style_gap": gap, **stats}
        if r["reference"] is not None:
            row["identity"] = scorer.score(r["reference"], r["image"])
            ident.append(row["identity"])
        if args.pose:
            d, k = evaluate.pose_distance(r["control"], r["image"])
            row["pose_dist"], row["pose_kps"] = d, k
            if k:
                poses.append(d)
                kps.append(k)
        gaps.append(gap)
        inks.append(stats)
        per_case.append(row)

    n_img = len(rows)
    n_held = len(list(holdout.rglob("*.png")))
    fid, note = "", args.note
    if n_img >= FID_MIN and n_held >= FID_MIN:
        fid = evaluate.style_fid(out_dir, holdout)
    else:
        note = (note + f" | FID skipped: {n_img} generated vs {n_held} held-out, "
                       f"need >={FID_MIN} each").strip(" |")

    mean = lambda k: float(np.mean([s[k] for s in inks]))
    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": args.baseline,
        "lora": Path(args.lora).name if args.lora else "",
        "lora_scale": args.lora_scale if args.lora else "",
        "seed": cfg.generation["seed"],
        "n": args.n,
        "cases": len(cfg.poses),
        "identity": round(float(np.mean(ident)), 4) if ident else "",
        "style_gap": round(float(np.mean(gaps)), 4),
        "black_ratio": round(mean("black_ratio"), 4),
        "midtone_ratio": round(mean("midtone_ratio"), 4),
        "edge_density": round(mean("edge_density"), 3),
        "white_ratio": round(mean("white_ratio"), 4),
        "pose_dist": round(float(np.mean(poses)), 4) if poses else "",
        "pose_kps": round(float(np.mean(kps)), 1) if kps else "",
        "fid": round(fid, 2) if fid != "" else "",
        "note": note,
    }

    (out_dir / "per_case.json").write_text(
        json.dumps({"target": target, "cases": per_case}, indent=2), encoding="utf-8")
    append(record)

    print()
    for k in ("identity", "style_gap", "black_ratio", "midtone_ratio",
              "edge_density", "white_ratio", "pose_dist", "fid"):
        if record[k] != "":
            print(f"  {k:14} {record[k]}")
    if note:
        print(f"  note           {note}")
    print(f"\nimages -> {out_dir}\nrow    -> {METRICS}")
    print_log()


def append(record: dict) -> None:
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    new = not METRICS.exists()
    with METRICS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(record)


def print_log() -> None:
    """Every run so far, oldest first. This is the progress record."""
    if not METRICS.exists():
        print(f"no runs logged yet ({METRICS})")
        return
    rows = list(csv.DictReader(METRICS.open(encoding="utf-8")))
    cols = ["run_id", "baseline", "lora_scale", "identity", "style_gap",
            "midtone_ratio", "white_ratio", "fid"]
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("\n" + "  ".join(c.ljust(width[c]) for c in cols))
    print("  ".join("-" * width[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    print("\nstyle_gap: distance to the artist's held-out ink profile. Lower is better.")
    print("identity:  DINOv2 cosine, reference vs output. Higher is better.")


if __name__ == "__main__":
    main()
