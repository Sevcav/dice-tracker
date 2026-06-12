"""
live_detect_settle.py
---------------------
Live YOLO detection with the legacy stability-tracking + settle-lock pattern
ported on top.

Why this exists:
    The plain live_detect.py shows raw YOLO output every frame, which causes
    apparent "fidgety" flicker when a die's top class and second-place class
    are close in confidence. The legacy CV pipeline solved this with a
    stability tracker that:
      1. Waits for dice to stop moving (~0.5s)
      2. Locks a majority-vote reading from the stable frames
      3. Holds that reading until dice move again

This script ports that pattern on top of YOLO predictions.

Workflow:
    - Drop dice in tray
    - Live preview shows raw YOLO predictions (dim color) while dice settle
    - Once SETTLE_FRAMES consecutive frames have the same die count AND
      centroids within SETTLE_MOVE pixels, a "settled" lock fires
    - Settled labels show in BRIGHT YELLOW with green border flash
    - Lock persists until dice move again, then unlocks and restarts

Controls (in preview window):
    1       Switch to block model
    2       Switch to d6 model
    3       Switch to d16 model
    +/-     Adjust confidence threshold by 0.05
    s       Save current annotated frame to live_snapshots/
    r       Reset the stability tracker
    q       Quit

Settings (tunable at top of file):
    SETTLE_FRAMES        Frames of stillness before locking (default 6)
    SETTLE_MOVE          Max centroid movement in px (default 12)
    SETTLE_COUNT_TOL     Allowable noisy frames (default 2)
    DETECT_EVERY_N       Run YOLO every N frames (default 1 = every frame)
"""

import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
SNAP_DIR   = ROOT / "live_snapshots"
SNAP_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0
RESOLUTION   = (1280, 720)

# ── Stability settings (from legacy detect_dice.py) ─────────────────────────
SETTLE_FRAMES    = 6      # consecutive frames of stillness required
SETTLE_MOVE      = 12     # max centroid drift in pixels
SETTLE_COUNT_TOL = 2      # frames with off-count tolerated within window
DETECT_EVERY_N   = 1      # only run YOLO every N frames (1 = every frame)

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
    (255, 165, 0),    (0, 165, 255),
    (128, 0, 128),    (0, 128, 128),
    (128, 128, 0),    (255, 0, 128),
    (0, 255, 128),    (128, 255, 0),
]

SETTLED_COLOR = (0, 255, 255)   # bright yellow for locked labels
DIM_FACTOR    = 0.55             # dim raw "still settling" boxes


def dim(color, factor=DIM_FACTOR):
    return tuple(int(c * factor) for c in color)


# ── Stability tracker ───────────────────────────────────────────────────────
# Lightly adapted from legacy detect_dice.py:DiceStabilityTracker
# Tracks centroids across frames and reports `settled` once dice are still.
# Also returns per-die "stable label" via majority vote across settled frames.
class StabilityTracker:
    def __init__(self):
        self.history     = deque(maxlen=SETTLE_FRAMES + 4)
        self.label_hist  = deque(maxlen=SETTLE_FRAMES + 4)

    def reset(self):
        self.history.clear()
        self.label_hist.clear()

    @staticmethod
    def _match(prev, curr, max_dist):
        """Match curr centroids to prev by nearest-neighbour."""
        matched_idx = []
        used = set()
        for px, py in prev:
            best_d = float('inf')
            best_j = -1
            for j, (cx, cy) in enumerate(curr):
                if j in used:
                    continue
                d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j == -1 or best_d > max_dist:
                return None
            matched_idx.append(best_j)
            used.add(best_j)
        return matched_idx

    def update(self, centroids, labels):
        """
        centroids: list of (cx, cy)
        labels:    list of (class_name, confidence) parallel to centroids

        Returns (settled: bool, locked_labels: list[str] or None)
        """
        self.history.append(list(centroids))
        self.label_hist.append(list(labels))

        if len(self.history) < SETTLE_FRAMES:
            return False, None

        recent = list(self.history)[-SETTLE_FRAMES:]
        recent_labels = list(self.label_hist)[-SETTLE_FRAMES:]
        counts = [len(h) for h in recent]
        modal_count = max(set(counts), key=counts.count)
        if modal_count == 0:
            return False, None

        bad = sum(1 for c in counts if c != modal_count)
        if bad > SETTLE_COUNT_TOL:
            return False, None

        good_frames        = [h for h in recent if len(h) == modal_count]
        good_frame_labels  = [recent_labels[i] for i, h in enumerate(recent)
                              if len(h) == modal_count]
        if len(good_frames) < SETTLE_FRAMES - SETTLE_COUNT_TOL:
            return False, None

        # Anchor = first good frame; match subsequent frames by proximity
        anchor        = good_frames[0]
        anchor_labels = good_frame_labels[0]
        # Per-anchor-index list of (label_name, confidence) across good frames
        per_die_labels = [[(anchor_labels[i][0], anchor_labels[i][1])]
                          for i in range(modal_count)]

        for frame_idx in range(1, len(good_frames)):
            f      = good_frames[frame_idx]
            f_lbls = good_frame_labels[frame_idx]
            matched_idx = self._match(anchor, f, SETTLE_MOVE * 4)
            if matched_idx is None:
                return False, None

            # Camera-shake detection: if all dice drift by the same vector,
            # treat as shake and skip the move check
            deltas_x = [f[j][0] - a[0] for a, j in zip(anchor, matched_idx)]
            deltas_y = [f[j][1] - a[1] for a, j in zip(anchor, matched_idx)]
            shake = False
            if len(deltas_x) > 1:
                med_dx = sorted(deltas_x)[len(deltas_x) // 2]
                med_dy = sorted(deltas_y)[len(deltas_y) // 2]
                if abs(med_dx) > 2 or abs(med_dy) > 2:
                    spread_x = max(deltas_x) - min(deltas_x)
                    spread_y = max(deltas_y) - min(deltas_y)
                    if spread_x <= 4 and spread_y <= 4:
                        shake = True

            if not shake:
                for (ax, ay), j in zip(anchor, matched_idx):
                    bx, by = f[j]
                    if abs(ax - bx) > SETTLE_MOVE or abs(ay - by) > SETTLE_MOVE:
                        return False, None

            # Accumulate per-die label votes
            for anchor_i, j in enumerate(matched_idx):
                per_die_labels[anchor_i].append(
                    (f_lbls[j][0], f_lbls[j][1])
                )

        # Majority vote per die; tie-break by mean confidence
        locked = []
        for votes in per_die_labels:
            class_counts = Counter(v[0] for v in votes)
            top_class, top_n = class_counts.most_common(1)[0]
            mean_conf = (sum(v[1] for v in votes if v[0] == top_class)
                         / max(top_n, 1))
            locked.append(f"{top_class} {mean_conf:.2f}")

        return True, locked


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_model(name: str) -> YOLO:
    onnx_path = MODELS_DIR / f"{name}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")
    print(f"Loading {name} model from {onnx_path}")
    model = YOLO(str(onnx_path), task="detect")
    print(f"  Classes: {list(model.names.values())}")
    return model


def draw_box(frame, x1, y1, x2, y2, color, label, *, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def annotate(frame, results, class_names, locked_labels, settled):
    """Draw boxes. If settled, override labels with the locked ones."""
    annotated = frame.copy()
    boxes = results.boxes
    n     = len(boxes)

    # Compute centroids for current frame
    centroids = []
    raw_labels = []
    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        centroids.append((cx, cy))
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        raw_labels.append((class_names.get(cls_id, str(cls_id)), conf))

    # If settled, we need to attach locked labels to detections in the
    # current frame.  Match by nearest-neighbour to the anchor positions
    # the tracker established (we approximate by matching the locked
    # labels to the current centroids in their original order — they
    # were locked on anchor of last good frame, which is close to now).
    use_locked = settled and locked_labels and len(locked_labels) == n

    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        cls_id = int(boxes.cls[i].item())
        if use_locked:
            label = locked_labels[i]
            color = SETTLED_COLOR
            thickness = 3
        else:
            cls_name = class_names.get(cls_id, str(cls_id))
            conf     = float(boxes.conf[i].item())
            label    = f"{cls_name} {conf:.2f}"
            color    = dim(COLORS[cls_id % len(COLORS)])
            thickness = 2
        draw_box(annotated, x1, y1, x2, y2, color, label, thickness=thickness)

    return annotated, centroids, raw_labels


def main():
    print("=" * 70)
    print("  Live YOLO Detection with Settle Lock")
    print(f"  Settle: {SETTLE_FRAMES} frames at <={SETTLE_MOVE}px drift")
    print(f"  Models from: {MODELS_DIR}")
    print("=" * 70)
    print()

    current_model_key = "block"
    models = {
        "block": load_model("block"),
        "d6":    load_model("d6"),
        "d16":   load_model("d16"),
    }
    conf_threshold = 0.40
    tracker = StabilityTracker()

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

    win = "Live YOLO + Settle - 1:block 2:d6 3:d16  +/-:conf  s:save  r:reset  q:quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    last_time  = time.time()
    fps_smooth = 0.0
    frame_n    = 0
    cached_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed")
            continue
        frame_n += 1

        model = models[current_model_key]

        # Run YOLO every DETECT_EVERY_N frames (default 1)
        if frame_n % DETECT_EVERY_N == 0 or cached_results is None:
            results = model.predict(
                source=frame, conf=conf_threshold, verbose=False,
            )
            cached_results = results[0]

        # Build centroids + raw labels for tracker
        centroids = []
        raw_labels = []
        boxes = cached_results.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            raw_labels.append(
                (model.names.get(cls_id, str(cls_id)), conf)
            )

        settled, locked_labels = tracker.update(centroids, raw_labels)

        # Annotate
        annotated, _, _ = annotate(
            frame, cached_results, model.names, locked_labels, settled
        )

        # Green border flash when settled
        if settled:
            cv2.rectangle(annotated,
                          (5, 5), (actual_w - 5, actual_h - 5),
                          (0, 255, 0), 6)

        # FPS smoothing
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_smooth = (0.9 * fps_smooth + 0.1 * inst_fps
                          if fps_smooth > 0 else inst_fps)

        # HUD
        status = "SETTLED" if settled else "settling..."
        status_color = (0, 255, 0) if settled else (0, 200, 255)
        hud_lines = [
            (f"Model: {current_model_key}    Conf: {conf_threshold:.2f}    "
             f"FPS: {fps_smooth:4.1f}    Status: {status}", (255, 255, 255)),
            (f"Detections: {len(boxes)}  "
             f"History: {len(tracker.history)}/{SETTLE_FRAMES}",
             (200, 200, 200)),
        ]
        for i, (line, col) in enumerate(hud_lines):
            y = 30 + i * 28
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        status_color if i == 0 else col, 1)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            current_model_key = "block"
            tracker.reset()
            print(f"  -> {current_model_key}")
        elif key == ord('2'):
            current_model_key = "d6"
            tracker.reset()
            print(f"  -> {current_model_key}")
        elif key == ord('3'):
            current_model_key = "d16"
            tracker.reset()
            print(f"  -> {current_model_key}")
        elif key in (ord('+'), ord('=')):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"  conf = {conf_threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"  conf = {conf_threshold:.2f}")
        elif key == ord('r'):
            tracker.reset()
            print("  tracker reset")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            tag = "settled" if settled else "settling"
            out_path = SNAP_DIR / f"settle_{current_model_key}_{tag}_{ts}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
