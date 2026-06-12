"""
analyze_recording.py
--------------------
Frame-by-frame analysis of a recorded live-detection clip.

For each frame, run YOLO inference + the stability tracker and log:
  - detection count
  - per-die class + confidence
  - tracker history length, settled state, locked labels

Output:
  - Prints a per-frame log to stdout (one line each)
  - Writes a CSV summary
  - Saves N evenly-spaced sample frames to recordings/<name>_samples/

Usage:
    python analyze_recording.py recordings/live_block_<ts>.mp4 [model_key]

Default model_key: block
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"

sys.path.insert(0, str(ROOT))
from live_detect_settle import (  # noqa: E402
    StabilityTracker, SETTLE_FRAMES, SETTLE_MOVE,
)


def analyze(video_path: Path, model_key: str = "block",
            conf_threshold: float = 0.40, n_samples: int = 12):
    if not video_path.exists():
        print(f"ERROR: {video_path} not found")
        return

    out_dir   = video_path.parent / f"{video_path.stem}_samples"
    out_dir.mkdir(exist_ok=True)
    csv_path  = video_path.parent / f"{video_path.stem}.csv"

    model = YOLO(str(MODELS_DIR / f"{model_key}.onnx"), task="detect")
    tracker = StabilityTracker()

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}  ({total} frames)")
    print(f"Model: {model_key}")
    print()

    sample_idxs = set(int(total * i / (n_samples - 1))
                      for i in range(n_samples))

    rows = [(
        "frame", "detections", "history",
        "settled", "locked_labels", "raw_labels",
    )]

    settled_count = 0
    detection_counts = Counter()
    transitions = []  # (frame, from_settled, to_settled)
    prev_settled = False

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # NOTE: the recorded video has overlay drawings already on it.  We
        # re-run YOLO against the same pixels - YOLO is somewhat robust to
        # the overlays (small text and outlines), but the metrics here may
        # be slightly noisier than on the raw frame.  Good enough for
        # diagnostic purposes.
        results = model.predict(
            source=frame, conf=conf_threshold, verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        n = len(boxes)
        detection_counts[n] += 1

        centroids  = []
        raw_labels = []
        for i in range(n):
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            raw_labels.append(
                (model.names.get(cls_id, str(cls_id)), conf)
            )

        settled, locked = tracker.update(centroids, raw_labels)
        if settled:
            settled_count += 1
        if settled != prev_settled:
            transitions.append((frame_n, prev_settled, settled))
        prev_settled = settled

        raw_str = "; ".join(f"{c[0]}:{c[1]:.2f}" for c in raw_labels)
        lock_str = " | ".join(locked) if locked else ""

        rows.append((
            frame_n, n, len(tracker.history),
            settled, lock_str, raw_str,
        ))

        if frame_n in sample_idxs:
            border = (0, 255, 0) if settled else (0, 100, 255)
            for i in range(n):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)
                if i < len(raw_labels):
                    cls_name, conf = raw_labels[i]
                    cv2.putText(frame, f"{cls_name} {conf:.2f}",
                                (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                border, 1)
            cv2.putText(frame, f"f{frame_n} det={n} settled={settled}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4)
            cv2.putText(frame, f"f{frame_n} det={n} settled={settled}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1)
            cv2.imwrite(str(out_dir / f"frame_{frame_n:04d}.jpg"), frame)

        frame_n += 1

    cap.release()

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print("Summary")
    print("-" * 60)
    print(f"  Total frames:        {frame_n}")
    print(f"  Settled frames:      {settled_count}  "
          f"({100*settled_count/max(frame_n,1):.1f}%)")
    print(f"  Detection-count distribution:")
    for n, c in sorted(detection_counts.items()):
        print(f"    {n} detections: {c} frames "
              f"({100*c/frame_n:.1f}%)")
    print(f"  Settled-state transitions: {len(transitions)}")
    if transitions[:6]:
        print("    First few:")
        for f_idx, was, now in transitions[:6]:
            print(f"      frame {f_idx}: {was} -> {now}")
    print()
    print(f"  CSV written:    {csv_path}")
    print(f"  Sample frames:  {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_recording.py <video.mp4> [model_key]")
        sys.exit(1)
    vid = Path(sys.argv[1])
    model_key = sys.argv[2] if len(sys.argv) > 2 else "block"
    analyze(vid, model_key)
