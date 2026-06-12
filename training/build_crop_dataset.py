"""
build_crop_dataset.py
---------------------
Assemble the tray-cropped combined 27-class dataset for the retrain:

  train = real train-split originals (tray crop, jittered origin)
        + auto-labeled retrain_candidates  (datasets/auto_labeled)
        + synthetic composites             (datasets/synth)
  valid = real valid-split originals (fixed tray crop)   <- the 92% bar
  test  = real test-split originals  (fixed tray crop)       is measured
                                                             on valid
Real frames come from capture_sessions/ at 1280x720 with labels
inverse-mapped from the Roboflow export (see crop_common.py). Synthetic
and auto-labeled data never enter valid/test — the bar is judged on real
frames only. Class ids follow merge_datasets.py order, matching the
production combined model.

Usage:
    python build_crop_dataset.py [--seed 42]
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from crop_common import (
    DATASETS, EXCLUDE_STEMS, IR_MAX_DEVIATION, SOURCES,
    color_deviation_tray, combined_classes, is_identity, originals_index,
    poly_bbox, read_polygons_raw, session_alignments, tray_rect, warp_pts,
    warp_to_ref, write_yolo_boxes,
)

OUT = DATASETS / "combined_crop"
CROP_JITTER = 12


def emit_real(entry: dict, off: int, split: str, img_out, lbl_out,
              rng: random.Random, stats: dict,
              alignments: dict) -> None:
    frame = cv2.imread(entry["raw"])
    if frame is None:
        stats["unreadable"] += 1
        return
    if color_deviation_tray(frame) >= IR_MAX_DEVIATION:
        stats["day_mode"] += 1
        return
    # normalize this session's tray position to the calibrated perspective
    M = alignments.get(Path(entry["raw"]).parent.name)
    if M is not None and not is_identity(M):
        frame = warp_to_ref(frame, M)
    tx, ty, tw, th = tray_rect()
    if split == "train":
        tx = max(0, min(tx + rng.randint(-CROP_JITTER, CROP_JITTER),
                        frame.shape[1] - tw))
        ty = max(0, min(ty + rng.randint(-CROP_JITTER, CROP_JITTER),
                        frame.shape[0] - th))
    boxes = []
    for cls, pts in read_polygons_raw(entry["label"]):
        if M is not None and not is_identity(M):
            pts = warp_pts(M, pts)
        x1, y1, x2, y2 = poly_bbox(pts)
        x1, x2 = x1 - tx, x2 - tx
        y1, y2 = y1 - ty, y2 - ty
        if x2 <= 0 or y2 <= 0 or x1 >= tw or y1 >= th:
            stats["box_outside_crop"] += 1
            continue
        boxes.append((cls + off, max(0.0, x1), max(0.0, y1),
                      min(float(tw), x2), min(float(th), y2)))
    if not boxes:
        stats["no_boxes"] += 1
        return
    crop = frame[ty:ty + th, tx:tx + tw]
    stem = entry["stem"]
    cv2.imwrite(str(img_out / f"real_{stem}.jpg"), crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    write_yolo_boxes(lbl_out / f"real_{stem}.txt", boxes, tw, th)
    stats["written"] += 1


def copy_extra(src_dir, img_out, lbl_out) -> int:
    n = 0
    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"
    if not img_dir.is_dir():
        return 0
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    names, offsets = combined_classes()
    index = originals_index()
    alignments = session_alignments()

    if OUT.exists():
        shutil.rmtree(OUT)

    for split in ["train", "valid", "test"]:
        img_out = OUT / split / "images"
        lbl_out = OUT / split / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        stats = {k: 0 for k in ["written", "unreadable", "day_mode",
                                "box_outside_crop", "no_boxes"]}
        for src in SOURCES:
            for entry in index[src][split]:
                if entry["stem"] in EXCLUDE_STEMS:
                    stats["excluded_bad_labels"] = \
                        stats.get("excluded_bad_labels", 0) + 1
                    continue
                emit_real(entry, offsets[src], split, img_out, lbl_out,
                          rng, stats, alignments)
        line = f"  {split}: {stats['written']} real"
        if split == "train":
            n_auto  = copy_extra(DATASETS / "auto_labeled", img_out, lbl_out)
            n_synth = copy_extra(DATASETS / "synth", img_out, lbl_out)
            line += f" + {n_auto} auto-labeled + {n_synth} synthetic"
        skipped = {k: v for k, v in stats.items()
                   if v and k != "written"}
        if skipped:
            line += f"   (skips: {skipped})"
        print(line)

    data_yaml = {
        "train": str(OUT / "train" / "images").replace("\\", "/"),
        "val":   str(OUT / "valid" / "images").replace("\\", "/"),
        "test":  str(OUT / "test" / "images").replace("\\", "/"),
        "nc":    len(names),
        "names": names,
    }
    with open(OUT / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"Wrote {OUT / 'data.yaml'} ({len(names)} classes)")


if __name__ == "__main__":
    main()
