"""
live_detect_supervision.py
--------------------------
Live YOLO detection using Roboflow's `supervision` library for temporal
smoothing. Replaces the hand-rolled StabilityTracker with the canonical
ByteTrack + DetectionsSmoother pipeline.

Architecture:
    raw frame
      -> YOLO.predict           (per-frame raw detections)
      -> sv.Detections.from_ultralytics
      -> sv.ByteTrack           (assign persistent tracker IDs)
      -> sv.DetectionsSmoother  (smooth box positions across N frames)
      -> per-tracker-id class history (majority vote for stable labels)
      -> annotated frame

This addresses two flicker modes:
  - Bounding-box position jitter:  fixed by DetectionsSmoother
  - Class-label flicker on a single die: fixed by per-tracker-id majority vote

Lighting variability is NOT addressed by this layer.  It operates on
detection outputs, not input pixels.

Controls (in preview window):
    1       Switch to block model
    2       Switch to d6 model
    3       Switch to d16 model
    +/-     Adjust confidence threshold by 0.05
    s       Save current annotated frame to live_snapshots/
    r       Reset tracker + label history
    q       Quit

Settings (tunable at top):
    SMOOTHER_LENGTH      DetectionsSmoother window (default 5)
    LABEL_HISTORY_LEN    per-die class vote window (default 8)
    LABEL_MIN_VOTES      min votes for a label to be stable (default 4)
"""

import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
SNAP_DIR   = ROOT / "live_snapshots"
SNAP_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0
RESOLUTION   = (1280, 720)

# ── Smoothing settings ──────────────────────────────────────────────────────
SMOOTHER_LENGTH   = 5    # frames of box-position smoothing
LABEL_HISTORY_LEN = 8    # frames of class-label history per tracker ID
LABEL_MIN_VOTES   = 4    # min same-class votes needed to be "stable"

# Colors for box annotations
LOCKED_COLOR    = sv.Color(r=0, g=255, b=255)     # bright yellow
UNLOCKED_COLOR  = sv.Color(r=140, g=140, b=140)   # gray (still settling)

# Distinct colors per class index (kept for raw inspection if needed)
PER_CLASS = [
    sv.Color(r=30, g=220, b=30),
    sv.Color(r=220, g=30, b=30),
    sv.Color(r=30, g=30, b=220),
    sv.Color(r=220, g=220, b=30),
    sv.Color(r=220, g=30, b=220),
    sv.Color(r=30, g=220, b=220),
    sv.Color(r=180, g=90, b=200),
    sv.Color(r=90, g=180, b=200),
]


class LabelStabilizer:
    """Per-tracker-id majority vote over the last N class predictions."""
    def __init__(self, history_len=LABEL_HISTORY_LEN,
                 min_votes=LABEL_MIN_VOTES):
        self.history_len = history_len
        self.min_votes   = min_votes
        self._hist = defaultdict(lambda: deque(maxlen=self.history_len))

    def reset(self):
        self._hist.clear()

    def update(self, tracker_ids, class_ids, confidences):
        """
        Returns list of (display_label, is_stable) parallel to inputs.
        display_label is the majority-voted class name with a confidence;
        is_stable is True if we have >= min_votes for that class.
        """
        result = []
        for tid, cid, conf in zip(tracker_ids, class_ids, confidences):
            self._hist[tid].append((cid, conf))
            votes = Counter(v[0] for v in self._hist[tid])
            top_class, top_n = votes.most_common(1)[0]
            mean_conf = (sum(v[1] for v in self._hist[tid] if v[0] == top_class)
                         / max(top_n, 1))
            stable = top_n >= self.min_votes
            result.append((int(top_class), mean_conf, stable))
        return result


def load_model(name: str) -> YOLO:
    onnx_path = MODELS_DIR / f"{name}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")
    print(f"Loading {name} model from {onnx_path}")
    model = YOLO(str(onnx_path), task="detect")
    print(f"  Classes: {list(model.names.values())}")
    return model


def draw_annotations(frame, detections, model, label_states):
    """
    detections:    sv.Detections (smoothed, with tracker_id)
    label_states:  list of (class_id, mean_conf, is_stable) parallel to detections
    """
    out = frame.copy()
    for i in range(len(detections)):
        x1, y1, x2, y2 = [int(v) for v in detections.xyxy[i]]
        tid = (int(detections.tracker_id[i])
               if detections.tracker_id is not None else -1)
        cid, conf, stable = label_states[i]
        cls_name = model.names.get(cid, str(cid))
        if stable:
            color = (0, 255, 255)  # bright yellow BGR
            thickness = 3
            label = f"#{tid} {cls_name} {int(round(conf*100))}%"
        else:
            color = (140, 140, 140)
            thickness = 2
            label = f"#{tid} {cls_name}? {int(round(conf*100))}%"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def main():
    print("=" * 70)
    print("  Live YOLO + supervision (ByteTrack + DetectionsSmoother)")
    print(f"  Smoother length: {SMOOTHER_LENGTH}")
    print(f"  Label history:   {LABEL_HISTORY_LEN}  (min {LABEL_MIN_VOTES} votes to stabilize)")
    print("=" * 70)
    print()

    current_model_key = "block"
    models = {
        "block": load_model("block"),
        "d6":    load_model("d6"),
        "d16":   load_model("d16"),
    }
    conf_threshold = 0.40

    # Camera frame rate is approximate at 30 — we'll override after measuring
    def make_tracker():
        return sv.ByteTrack(
            frame_rate=30,
            lost_track_buffer=120,           # ~4s of detection loss before track dies
            minimum_consecutive_frames=3,    # 3 frames before ID is committed
            track_activation_threshold=0.30, # forgiving on initial low-confidence
        )
    tracker     = make_tracker()
    smoother    = sv.DetectionsSmoother(length=SMOOTHER_LENGTH)
    label_stab  = LabelStabilizer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera open at {actual_w}x{actual_h}")
    print()
    print("Controls:  1=block  2=d6  3=d16  +/-=conf  s=save  r=reset  q=quit")
    print()

    win = "Live YOLO + supervision - 1:block 2:d6 3:d16  +/-:conf  s:save  r:reset  q:quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    last_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        model = models[current_model_key]
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        # Compute per-tracker stable labels (only on detections with a valid id)
        if (detections.tracker_id is not None
                and len(detections) > 0
                and detections.class_id is not None):
            label_states = label_stab.update(
                detections.tracker_id.tolist(),
                detections.class_id.tolist(),
                detections.confidence.tolist(),
            )
        else:
            label_states = []

        annotated = draw_annotations(frame, detections, model, label_states)

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_smooth = (0.9 * fps_smooth + 0.1 * inst_fps
                          if fps_smooth > 0 else inst_fps)

        stable_count = sum(1 for _, _, s in label_states if s)
        hud_lines = [
            f"Model: {current_model_key}    Conf: {conf_threshold:.2f}    "
            f"FPS: {fps_smooth:4.1f}",
            f"Detections: {len(detections)}    "
            f"Stable: {stable_count}/{len(detections)}",
        ]
        for i, line in enumerate(hud_lines):
            y = 30 + i * 28
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Border flash when ALL detections are stable AND >= 1 detection
        all_stable = (len(detections) > 0
                      and stable_count == len(detections))
        if all_stable:
            cv2.rectangle(annotated, (5, 5),
                          (actual_w - 5, actual_h - 5),
                          (0, 255, 0), 6)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            current_model_key = "block"
            tracker = make_tracker()
            smoother = sv.DetectionsSmoother(length=SMOOTHER_LENGTH)
            label_stab.reset()
            print(f"  -> {current_model_key}")
        elif key == ord('2'):
            current_model_key = "d6"
            tracker = make_tracker()
            smoother = sv.DetectionsSmoother(length=SMOOTHER_LENGTH)
            label_stab.reset()
            print(f"  -> {current_model_key}")
        elif key == ord('3'):
            current_model_key = "d16"
            tracker = make_tracker()
            smoother = sv.DetectionsSmoother(length=SMOOTHER_LENGTH)
            label_stab.reset()
            print(f"  -> {current_model_key}")
        elif key in (ord('+'), ord('=')):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"  conf = {conf_threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"  conf = {conf_threshold:.2f}")
        elif key == ord('r'):
            tracker = make_tracker()
            smoother = sv.DetectionsSmoother(length=SMOOTHER_LENGTH)
            label_stab.reset()
            print("  reset trackers and history")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = SNAP_DIR / f"sv_{current_model_key}_{ts}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
