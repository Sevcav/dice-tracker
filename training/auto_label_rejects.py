"""
auto_label_rejects.py
---------------------
Convert the banked retrain_candidates/<type>/ frames into tray-cropped
YOLO training labels for the combined 27-class model — no manual labeling.

What the bank contains (written by eval_harness.py / dice_tracker.py):
  eval_miss_*.json   predicted boxes (full-frame px, sorted LEFT->RIGHT by
                     (cx, cy) — the same order the ground truth was keyed
                     in) + "truth" labels + count_mismatch flag.
  reject_*.json      predicted boxes only — NO ground truth (player hit R).

Auto-labelable: eval misses with count_mismatch == false. The model's
boxes localize fine (the proven failure mode is classification, not
localization), so box_i + truth_i is a correct label pair.

Skipped -> manual queue: count-mismatch frames (boxes incomplete) and
reject frames (no truth). Day-mode frames (in-tray color deviation >= 6)
are dropped entirely — the models train on IR only.

Output:
  training/datasets/auto_labeled/images/auto_<stem>.jpg   tray-cropped
  training/datasets/auto_labeled/labels/auto_<stem>.txt   combined ids
  training/datasets/auto_labeled/manual_queue.txt         needs a human
  training/datasets/auto_labeled/qc/*.jpg                 (--qc) overlays

Usage:
    python auto_label_rejects.py [--qc]
"""

import argparse
import json
from pathlib import Path

import cv2

from crop_common import (
    IR_MAX_DEVIATION, PROJECT, DATASETS,
    color_deviation_tray, combined_classes, tray_rect, write_yolo_boxes,
)

BANK = PROJECT / "retrain_candidates"
OUT  = DATASETS / "auto_labeled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", action="store_true",
                    help="write label-overlay images for visual QC")
    args = ap.parse_args()

    names, _ = combined_classes()
    name_to_id = {n: i for i, n in enumerate(names)}
    tx, ty, tw, th = tray_rect()

    img_out = OUT / "images"
    lbl_out = OUT / "labels"
    qc_out  = OUT / "qc"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    if args.qc:
        qc_out.mkdir(exist_ok=True)

    written, queued, dropped_day, dropped_bad = 0, [], 0, []
    for meta_path in sorted(BANK.glob("*/*.json")):
        meta = json.loads(meta_path.read_text())
        stem = meta_path.stem
        img_path = meta_path.with_suffix(".jpg")
        if not img_path.exists():
            dropped_bad.append(f"{stem}: missing jpg")
            continue

        truth = meta.get("truth")
        if truth is None or meta.get("count_mismatch"):
            why = "no ground truth" if truth is None else "count mismatch"
            queued.append(f"{meta_path}  ({why})")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None or frame.shape[:2] != (720, 1280):
            dropped_bad.append(f"{stem}: unreadable or not 1280x720")
            continue
        dev = color_deviation_tray(frame)
        if dev >= IR_MAX_DEVIATION:
            dropped_day += 1
            print(f"  [day-mode] {stem} (deviation {dev:.1f}) — dropped")
            continue

        boxes = meta["boxes"]
        if len(boxes) != len(truth):
            dropped_bad.append(f"{stem}: {len(boxes)} boxes vs "
                               f"{len(truth)} truth labels")
            continue

        out_boxes = []
        ok = True
        for label, (x1, y1, x2, y2) in zip(truth, boxes):
            cid = name_to_id.get(label)
            if cid is None:
                dropped_bad.append(f"{stem}: unknown label {label!r}")
                ok = False
                break
            # full-frame -> tray-crop coords
            cx1, cy1 = x1 - tx, y1 - ty
            cx2, cy2 = x2 - tx, y2 - ty
            # a die symbol fully outside the tray crop would mean the ROI
            # is stale — refuse rather than emit a clipped label
            if cx2 <= 0 or cy2 <= 0 or cx1 >= tw or cy1 >= th:
                dropped_bad.append(f"{stem}: box outside tray ROI")
                ok = False
                break
            out_boxes.append((cid, max(0.0, cx1), max(0.0, cy1),
                              min(float(tw), cx2), min(float(th), cy2)))
        if not ok:
            continue

        crop = frame[ty:ty + th, tx:tx + tw]
        cv2.imwrite(str(img_out / f"auto_{stem}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo_boxes(lbl_out / f"auto_{stem}.txt", out_boxes, tw, th)
        written += 1

        if args.qc:
            vis = crop.copy()
            for cid, x1, y1, x2, y2 in out_boxes:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 1)
                cv2.putText(vis, names[cid], (int(x1), int(y1) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imwrite(str(qc_out / f"auto_{stem}.jpg"), vis)

    (OUT / "manual_queue.txt").write_text("\n".join(queued)
                                          + ("\n" if queued else ""))
    print()
    print(f"auto-labeled : {written} frames -> {img_out}")
    print(f"manual queue : {len(queued)} frames -> {OUT / 'manual_queue.txt'}")
    print(f"day-mode drop: {dropped_day}")
    if dropped_bad:
        print(f"problems     : {len(dropped_bad)}")
        for d in dropped_bad:
            print(f"  - {d}")


if __name__ == "__main__":
    main()
