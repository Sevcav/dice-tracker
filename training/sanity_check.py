"""
sanity_check.py
---------------
Visual smoke test of the trained ONNX models against held-out test frames.

For each model (block, d6, d16):
  - Load the ONNX from training/models/
  - Run inference on every image in training/datasets/<type>/test/images/
  - Draw bounding boxes + class labels + confidence
  - Save annotated images to training/sanity_check/<type>/

Outputs:
    training/sanity_check/block/<image>.jpg
    training/sanity_check/d6/<image>.jpg
    training/sanity_check/d16/<image>.jpg

Console output summarises per-model:
    - Total detections across all test images
    - Detections per class
    - Any images with zero detections (red flag)

Usage:
    python sanity_check.py              # all three models
    python sanity_check.py block d6     # subset
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATASETS   = ROOT / "datasets"
OUT_ROOT   = ROOT / "sanity_check"

# Confidence threshold for drawing — lower than runtime default so we see
# borderline predictions during sanity check. Real runtime uses 0.5+.
CONF_THRESHOLD = 0.25

# Distinct colors per class index (cycled if more classes than colors)
COLORS = [
    (30, 220, 30),    # green
    (220, 30, 30),    # blue (BGR)
    (30, 30, 220),    # red
    (220, 220, 30),   # cyan
    (220, 30, 220),   # magenta
    (30, 220, 220),   # yellow
    (180, 90, 200),   # purple
    (90, 180, 200),   # orange-ish
]


def run_one(name: str):
    onnx_path = MODELS_DIR / f"{name}.onnx"
    test_dir  = DATASETS / name / "test" / "images"
    out_dir   = OUT_ROOT / name

    if not onnx_path.exists():
        print(f"[{name}] ONNX missing: {onnx_path}")
        return
    if not test_dir.exists():
        print(f"[{name}] Test images missing: {test_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(f"  Model: {name}")
    print(f"  ONNX:  {onnx_path}")
    print(f"  Tests: {test_dir}")
    print("=" * 70)

    model = YOLO(str(onnx_path), task="detect")
    class_names = model.names
    print(f"  Classes: {class_names}")

    images = sorted(test_dir.glob("*.jpg"))
    print(f"  Test images: {len(images)}")

    total_detections = 0
    class_counts     = Counter()
    zero_detection_imgs = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    SKIP unreadable: {img_path.name}")
            continue

        results = model.predict(
            source=img, conf=CONF_THRESHOLD, verbose=False,
        )
        r = results[0]
        boxes = r.boxes

        n = len(boxes)
        total_detections += n
        if n == 0:
            zero_detection_imgs.append(img_path.name)

        # Draw
        annotated = img.copy()
        for i in range(n):
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            cls_name = class_names.get(cls_id, str(cls_id))
            color = COLORS[cls_id % len(COLORS)]

            class_counts[cls_name] += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated,
                          (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(str(out_dir / img_path.name), annotated)

    print()
    print(f"  Total detections: {total_detections} across {len(images)} images")
    print(f"  Per-class:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls:15s} {count}")
    if zero_detection_imgs:
        print()
        print(f"  WARNING: {len(zero_detection_imgs)} image(s) had ZERO detections:")
        for n in zero_detection_imgs:
            print(f"    {n}")
    print()
    print(f"  Annotated images saved to: {out_dir}")


def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else ["block", "d6", "d16"]
    for k in keys:
        run_one(k)
    print()
    print("Done. Open the sanity_check folder to inspect the boxes visually:")
    print(f"  {OUT_ROOT}")


if __name__ == "__main__":
    main()
