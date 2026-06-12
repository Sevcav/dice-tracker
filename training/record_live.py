"""
record_live.py
--------------
Record a short MP4 of the annotated live detection so it can be analyzed
frame-by-frame.

Same overlays as live_detect_settle.py (dim while settling, yellow when
locked, green border flash, HUD).  Records every frame as the user sees it.

Usage:
    python record_live.py            # records 10 seconds, block model
    python record_live.py d6 15      # records 15 seconds with d6 model
    python record_live.py d16 20     # 20 seconds with d16

Output:
    training/recordings/live_<model>_<timestamp>.mp4

Controls during recording:
    q       Stop early
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# Reuse the tracker from live_detect_settle if importable; else inline a copy.
sys.path.insert(0, str(Path(__file__).parent))
from live_detect_settle import (  # noqa: E402
    StabilityTracker, COLORS, SETTLED_COLOR, dim, draw_box,
    SETTLE_FRAMES, SETTLE_MOVE, DETECT_EVERY_N,
)

CAMERA_INDEX = 0
RESOLUTION   = (1280, 720)
ROOT         = Path(__file__).parent
MODELS_DIR   = ROOT / "models"
REC_DIR      = ROOT / "recordings"
REC_DIR.mkdir(exist_ok=True)


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "block"
    duration  = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"Loading {model_key} model...")
    onnx_path = MODELS_DIR / f"{model_key}.onnx"
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found")
        return
    model = YOLO(str(onnx_path), task="detect")
    print(f"  Classes: {list(model.names.values())}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Warm up + measure FPS for the writer
    print("Warming up camera...")
    frame_times = []
    for _ in range(30):
        ret, _ = cap.read()
        frame_times.append(time.time())
    intervals = [frame_times[i+1] - frame_times[i]
                 for i in range(len(frame_times) - 1)]
    actual_fps = max(5.0, min(30.0, 1.0 / max(0.01,
                                              sum(intervals) / len(intervals))))
    print(f"  Measured ~{actual_fps:.1f} FPS")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = REC_DIR / f"live_{model_key}_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, actual_fps,
                             (actual_w, actual_h))

    win = "Recording... Q to stop early"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    tracker = StabilityTracker()
    conf_threshold = 0.40

    print(f"Recording {duration}s to {out_path}")
    print("Q to stop early.")
    start = time.time()
    frame_n = 0
    cached_results = None

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_n += 1

        if frame_n % DETECT_EVERY_N == 0 or cached_results is None:
            results = model.predict(
                source=frame, conf=conf_threshold, verbose=False,
            )
            cached_results = results[0]

        boxes = cached_results.boxes
        centroids  = []
        raw_labels = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            raw_labels.append(
                (model.names.get(cls_id, str(cls_id)), conf)
            )

        settled, locked_labels = tracker.update(centroids, raw_labels)
        use_locked = settled and locked_labels and len(locked_labels) == len(boxes)

        annotated = frame.copy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            cls_id = int(boxes.cls[i].item())
            if use_locked:
                label = locked_labels[i]
                color = SETTLED_COLOR
                thickness = 3
            else:
                cls_name = model.names.get(cls_id, str(cls_id))
                conf     = float(boxes.conf[i].item())
                label    = f"{cls_name} {conf:.2f}"
                color    = dim(COLORS[cls_id % len(COLORS)])
                thickness = 2
            draw_box(annotated, x1, y1, x2, y2, color, label, thickness=thickness)

        if settled:
            cv2.rectangle(annotated, (5, 5),
                          (actual_w - 5, actual_h - 5),
                          (0, 255, 0), 6)

        elapsed = time.time() - start
        status = "SETTLED" if settled else "settling..."
        hud = (
            f"Model: {model_key}  Conf: {conf_threshold:.2f}  "
            f"Time: {elapsed:4.1f}/{duration}s  Status: {status}"
        )
        hud2 = (
            f"Detections: {len(boxes)}  "
            f"History: {min(len(tracker.history), SETTLE_FRAMES)}/{SETTLE_FRAMES}"
        )
        for i, line in enumerate([hud, hud2]):
            y = 30 + i * 28
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

        writer.write(annotated)
        cv2.imshow(win, annotated)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {out_path}  ({frame_n} frames)")


if __name__ == "__main__":
    main()
