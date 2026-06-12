"""
live_detect.py
--------------
Live YOLO detection against the Arducam, on the PC.

Mirrors what the Pi will eventually do, but with a live preview window so
we can iterate quickly. Run this BEFORE deploying to the Pi to validate
each model handles real-world frames (not just held-out test frames).

Controls (in the preview window):
    1       Switch to block model
    2       Switch to d6 model
    3       Switch to d16 model
    +/-     Adjust confidence threshold by 0.05
    s       Save current annotated frame to live_snapshots/
    q       Quit

Defaults:
    Model:          block
    Confidence:     0.40 (lower than runtime to see borderline picks)
    Resolution:     1280x720 (what we trained against after Roboflow resize)

Usage:
    python live_detect.py
"""

import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
SNAP_DIR   = ROOT / "live_snapshots"
SNAP_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0
RESOLUTION   = (1280, 720)

# Distinct colors per class index (cycled if more classes than colors)
COLORS = [
    (30, 220, 30),    # green
    (220, 30, 30),    # blue
    (30, 30, 220),    # red
    (220, 220, 30),   # cyan
    (220, 30, 220),   # magenta
    (30, 220, 220),   # yellow
    (180, 90, 200),   # purple
    (90, 180, 200),   # orange-ish
    (255, 165, 0),    # navy-orange
    (0, 165, 255),    # orange
    (128, 0, 128),    # dark purple
    (0, 128, 128),    # teal
    (128, 128, 0),    # dark cyan
    (255, 0, 128),    # pink
    (0, 255, 128),    # lime
    (128, 255, 0),    # bright green
]


def load_model(name: str) -> YOLO:
    onnx_path = MODELS_DIR / f"{name}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")
    print(f"Loading {name} model from {onnx_path}")
    model = YOLO(str(onnx_path), task="detect")
    print(f"  Classes: {list(model.names.values())}")
    return model


def draw_detections(frame, results, class_names):
    annotated = frame.copy()
    boxes = results.boxes

    counts = Counter()
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        cls_name = class_names.get(cls_id, str(cls_id))
        color = COLORS[cls_id % len(COLORS)]
        counts[cls_name] += 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated, counts


def main():
    # --- State ---
    current_model_key = "block"
    models = {
        "block": load_model("block"),
        "d6":    load_model("d6"),
        "d16":   load_model("d16"),
    }
    conf_threshold = 0.40

    # --- Camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print()
    print(f"Camera open at {actual_w}x{actual_h}")
    print()
    print("Controls:  1=block  2=d6  3=d16  +/-=conf  s=save  q=quit")
    print()

    win = "Live YOLO Detection - 1:block 2:d6 3:d16 +/-:conf s:save q:quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    # FPS smoother
    last_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed")
            continue

        model = models[current_model_key]
        results = model.predict(
            source=frame, conf=conf_threshold, verbose=False,
        )
        annotated, counts = draw_detections(frame, results[0], model.names)

        # FPS calc
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps

        # HUD
        hud_lines = [
            f"Model: {current_model_key}    Conf: {conf_threshold:.2f}    FPS: {fps_smooth:4.1f}",
            f"Detections: {sum(counts.values())}    "
            + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
        ]
        for i, line in enumerate(hud_lines):
            y = 30 + i * 28
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            current_model_key = "block"
            print(f"  -> {current_model_key}")
        elif key == ord('2'):
            current_model_key = "d6"
            print(f"  -> {current_model_key}")
        elif key == ord('3'):
            current_model_key = "d16"
            print(f"  -> {current_model_key}")
        elif key in (ord('+'), ord('=')):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"  conf = {conf_threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"  conf = {conf_threshold:.2f}")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = SNAP_DIR / f"live_{current_model_key}_{ts}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
