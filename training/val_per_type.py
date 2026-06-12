"""
val_per_type.py
---------------
Per-dice-type mean mAP@50 on the held-out validation split — the locked
quality bar: every type's mean must be >= 92% or the new ONNX does not
replace production (HANDOFF.md / SYNTH_RETRAIN_PLAN.md).

Usage:
    python val_per_type.py                          # combined_crop best.pt
    python val_per_type.py --weights <path.pt> --data <data.yaml>
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from crop_common import DATASETS, combined_classes

ROOT = Path(__file__).parent
BAR = 0.92


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=str(ROOT / "runs" / "combined_crop" / "weights"
                                / "best.pt"))
    ap.add_argument("--data",
                    default=str(DATASETS / "combined_crop" / "data.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    names, offsets = combined_classes()
    type_of = {}
    srcs = list(offsets)
    for i, src in enumerate(srcs):
        lo = offsets[src]
        hi = offsets[srcs[i + 1]] if i + 1 < len(srcs) else len(names)
        for c in range(lo, hi):
            type_of[c] = src

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, device=0, workers=0,
                  verbose=False, plots=False)

    print()
    print(f"weights: {args.weights}")
    print(f"split:   {args.split}")
    print(f"overall mAP@50 = {m.box.map50:.4f}")
    print()
    per_type: dict[str, list[float]] = {s: [] for s in srcs}
    print("per-class AP@50:")
    for ci, ap50 in zip(m.box.ap_class_index, m.box.ap50):
        ci = int(ci)
        per_type[type_of[ci]].append(float(ap50))
        print(f"  {names[ci]:<14} {ap50:.4f}")
    print()
    print(f"QUALITY BAR (>= {BAR:.0%} per-type mean mAP@50):")
    all_pass = True
    for src in srcs:
        aps = per_type[src]
        mean = sum(aps) / len(aps) if aps else 0.0
        ok = mean >= BAR and len(aps) > 0
        all_pass &= ok
        print(f"  {src:<6} mean mAP@50 = {mean:.4f}  ({len(aps)} classes)  "
              f"{'PASS' if ok else 'FAIL'}")
    print()
    print("VERDICT:", "PASS — eligible to replace production after live "
          "eval" if all_pass else "FAIL — do NOT deploy; iterate on the "
          "data mix")


if __name__ == "__main__":
    main()
