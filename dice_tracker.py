"""
dice_tracker.py
---------------
The actual Blood Bowl Dice Tracker app — the production runtime that wires
together camera capture, YOLO detection, supervision-based smoothing, and
the SETUP.txt UX (settle -> SPACE to confirm -> log).

This replaces the live_detect_* dev scripts. Those stay in training/ for
future debugging; this is the entrypoint.

Architecture per session:
    YOLO ONNX (per dice type)
      -> sv.ByteTrack + DetectionsSmoother (per-frame stability)
      -> LabelStabilizer (per-tracker-id class majority vote)
      -> settle detection (all tracker labels stable AND >0 dice)
      -> SPACE = confirm and log; BACKSPACE = undo last; R = reject (for retrain)
      -> tracker resets between rolls

Controls (in preview window):
    1   set Player 1 as active roller
    2   set Player 2 as active roller
    B   switch to block model        (block dice)
    D   switch to d6 model           (BB d6)
    X   switch to d16 model          (D16 trapezohedron)
    SPACE   confirm settled roll, log it under active player
    BACKSPACE  undo last logged roll
    R   reject current settled read; save frame for future retraining
    S   save session to disk (session.json + session.csv)
    L   load session.json (resume earlier game)
    Q   quit (asks to save if there are unsaved rolls)

Output:
    session.json                  full session state for reload
    session.csv                   tabular roll log
    retrain_candidates/<type>/    frames + JSON the model got wrong
"""

import csv
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import db
import d16_geometry
import inference_backend as backend
from inference_backend import predict_detections
from dice_types import CLASS_TO_TYPE, majority_type

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "training" / "models"
SESSION_JSON = ROOT / "session.json"
SESSION_CSV  = ROOT / "session.csv"
RETRAIN_DIR  = ROOT / "retrain_candidates"
RETRAIN_DIR.mkdir(exist_ok=True)
(RETRAIN_DIR / "block").mkdir(exist_ok=True)
(RETRAIN_DIR / "d6").mkdir(exist_ok=True)
(RETRAIN_DIR / "d16").mkdir(exist_ok=True)

CAMERA_INDEX = 0
RESOLUTION   = (1280, 720)
CONF_THRESHOLD = 0.40

# ── IR-mode self-check ──────────────────────────────────────────────────────
# The models are trained exclusively on IR-mode frames. Day-mode frames
# (photoresistor exposed to bright light) are out-of-distribution and
# silently degrade accuracy — the May 13 morning session ran in day mode
# without anyone noticing. Measured IN-TRAY on reference frames (2026-06-11):
# true IR frames = 4.0-5.2 (chroma noise on the tray's high-contrast
# graphics), day-mode frames = ~10.8-11. Threshold sits between the bands.
# The tray ROI is used (when tray_roi.json exists) so ambient lamp spill on
# the floor around the rig can't trip a false day-mode warning.
DAY_MODE_DEVIATION = 8.0
IR_CHECK_INTERVAL  = 90    # frames between live re-checks (~3s at 30fps)
TRAY_ROI_FILE = ROOT / "tray_roi.json"


def _load_tray_roi():
    """Tray rect from tray_roi.json scaled to RESOLUTION, or None."""
    try:
        d = json.loads(TRAY_ROI_FILE.read_text())
        sx = RESOLUTION[0] / d["frame_width"]
        sy = RESOLUTION[1] / d["frame_height"]
        return (int(d["x"] * sx), int(d["y"] * sy),
                int(d["w"] * sx), int(d["h"] * sy))
    except Exception:
        return None


_TRAY_ROI = _load_tray_roi()


def load_model_meta(stem: str) -> dict:
    """Sidecar metadata for models/<stem>.onnx — currently whether the
    model was trained tray-cropped (models/<stem>.onnx.json, written by
    train_all.py). Missing sidecar = full-frame model (the default for
    every model trained before the 2026-06 crop retrain)."""
    meta = {"tray_crop": False}
    try:
        meta.update(json.loads(
            (MODELS_DIR / f"{stem}.onnx.json").read_text()))
    except Exception:
        pass
    return meta


def tray_crop_rect(actual_w: int, actual_h: int) -> tuple | None:
    """Tray ROI (x, y, w, h) scaled to the actual capture resolution —
    the inference crop for tray-crop-trained models."""
    try:
        d = json.loads(TRAY_ROI_FILE.read_text())
        sx = actual_w / d["frame_width"]
        sy = actual_h / d["frame_height"]
        return (int(d["x"] * sx), int(d["y"] * sy),
                int(d["w"] * sx), int(d["h"] * sy))
    except Exception:
        return None


# predict_detections is provided by inference_backend (re-exported above so
# eval_harness's `from dice_tracker import predict_detections` keeps working).
# It crops/pads to the tray ROI for crop-trained models and returns
# full-frame coords — identical geometry on the ultralytics (PC) and
# torch-free onnx (Pi) backends.


def color_deviation(frame) -> float:
    """Mean absolute channel deviation from grayscale (0 = pure mono),
    measured inside the tray ROI when available."""
    if _TRAY_ROI is not None:
        x, y, w, h = _TRAY_ROI
        frame = frame[y:y + h, x:x + w]
    small = frame[::4, ::4].astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return float((np.abs(r - g).mean() + np.abs(g - b).mean()
                  + np.abs(r - b).mean()) / 3)


def alignment_check(cap, actual_w: int, actual_h: int) -> bool:
    """Pre-flight camera alignment: overlay the saved tray-corner reference
    (tray_roi.json) on the live feed so the camera can be adjusted back to
    the training-time position whenever the rig has been moved. The model
    only knows the calibrated perspective — this runs before every session.

    SPACE/ENTER = aligned, continue.  Q/ESC = abort.
    Returns True to proceed, False if the user aborted.
    """
    try:
        d = json.loads(TRAY_ROI_FILE.read_text())
        sx = actual_w / d["frame_width"]
        sy = actual_h / d["frame_height"]
        pts = [(int(cx * sx), int(cy * sy)) for cx, cy in d["corners"]]
    except Exception:
        print("No tray_roi.json reference — skipping alignment check.")
        return True

    print("Alignment check: match the live tray edges to the GREEN outline")
    print("(adjust the camera arm if needed), then press SPACE to continue.")
    # NOTE: window titles and cv2.putText HUD strings must be plain ASCII.
    # Non-ASCII (em dashes etc.) gets mangled on Windows and can even create
    # DUPLICATE ghost windows that destroyWindow can't close.
    win = "Alignment check - match tray to GREEN outline, SPACE to continue"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    proceed = None
    while proceed is None:
        ret, frame = cap.read()
        if not ret:
            continue
        for i in range(4):
            p1, p2 = pts[i], pts[(i + 1) % 4]
            cv2.line(frame, p1, p2, (0, 255, 0), 3)
            cv2.circle(frame, p1, 8, (0, 255, 0), -1)
            cv2.putText(frame, ["TL", "TR", "BR", "BL"][i],
                        (p1[0] + 12, p1[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        hud_line(frame, "ALIGNMENT CHECK - match tray edges to green outline",
                 25, color=(0, 255, 0), scale=0.7)
        hud_line(frame, "SPACE/ENTER = aligned, start  |  Q = quit",
                 50, color=(0, 255, 255), scale=0.6)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (32, 13):
            proceed = True
        elif key in (ord('q'), 27):
            proceed = False
    cv2.destroyWindow(win)
    # On Windows the window only actually closes once the GUI event queue
    # is pumped — destroyWindow alone leaves a frozen ghost window.
    for _ in range(5):
        cv2.waitKey(1)
    return proceed


def _draw_align_overlay(frame, actual_w, actual_h):
    """Draw the green tray-corner reference on a frame (shared by the GUI
    and headless/phone alignment paths). Returns the annotated copy, or the
    original if no tray_roi.json reference exists."""
    try:
        d = json.loads(TRAY_ROI_FILE.read_text())
        sx = actual_w / d["frame_width"]
        sy = actual_h / d["frame_height"]
        pts = [(int(cx * sx), int(cy * sy)) for cx, cy in d["corners"]]
    except Exception:
        return frame
    out = frame.copy()
    for i in range(4):
        p1, p2 = pts[i], pts[(i + 1) % 4]
        cv2.line(out, p1, p2, (0, 255, 0), 3)
        cv2.circle(out, p1, 8, (0, 255, 0), -1)
    hud_line(out, "Match tray to GREEN outline, then Confirm on phone",
             25, color=(0, 255, 0), scale=0.7)
    return out


def alignment_check_web(cap, actual_w, actual_h, web_control) -> bool:
    """Headless alignment: stream the green-outline overlay to the phone
    (/align) and wait for the operator to tap Confirm. No monitor needed —
    the sealed rig's only screen is the phone. Returns True to proceed.

    Falls through (returns True) if there's no tray_roi.json reference or
    no web control to drive the phone UI."""
    if web_control is None:
        print("No web control — skipping alignment (headless).")
        return True
    if not TRAY_ROI_FILE.exists():
        print("No tray_roi.json reference — skipping alignment check.")
        return True
    print("Headless alignment: open  /align  on the phone, match the tray")
    print("to the green outline, then tap Confirm.")
    web_control.take_alignment()   # clear any stale confirm
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        overlay = _draw_align_overlay(frame, actual_w, actual_h)
        ok, buf = cv2.imencode(".jpg", overlay,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            web_control.set_frame(buf.tobytes())
        if web_control.take_alignment():
            print("Alignment confirmed from phone.")
            return True


# ── Smoothing / settle settings ─────────────────────────────────────────────
SMOOTHER_LENGTH    = 5
LABEL_HISTORY_LEN  = 8
LABEL_MIN_VOTES    = 4
SETTLE_REQUIRES_ALL_STABLE = True   # green flash only when every die is stable
# The detection COUNT must also be constant this many consecutive frames
# before settle. Without this, a die whose detection flickers in/out near
# the conf threshold lets the roll settle during an "out" frame and the die
# is silently dropped from the read — 2026-06-11 eval: 16 of 50 rolls lost
# dice this way, and 14/16 saved miss-frames had ALL dice detectable on the
# very frame that settled. (The legacy DiceStabilityTracker had a count
# requirement; it was lost in the supervision port.)
COUNT_STABLE_FRAMES = 10

# Uncertainty marker thresholds — PER DICE TYPE, recalibrated 2026-06-12
# for the tray-crop combined model (its confidence scale runs lower than
# the old full-frame model: block correct reads average 0.74 vs 0.90).
# From the live eval sessions:
#   block: 66/66 correct, conf p10=0.53 -> 0.60 keeps the historical ~20%
#          flag-noise rate (no wrong reads observed to calibrate against)
#   d16:   correct mean 0.86 vs wrong 0.69 -> 0.80 catches 67% of wrong
#          reads while flagging only 12% of correct ones
#   d6:    no live data with this model yet — block's distribution is the
#          closest proxy; recalibrate after the first d6 session
# Labels below the threshold show orange with a "?" so the player's eye
# is drawn to questionable dice BEFORE confirming (nudge to re-read).
# This is the on-screen precursor of the OLED uncertainty markers.
CONF_UNCERTAIN = {"block": 0.60, "d6": 0.60, "d16": 0.80}
CONF_UNCERTAIN_DEFAULT = 0.80


def uncertain_threshold(label: str) -> float:
    return CONF_UNCERTAIN.get(CLASS_TO_TYPE.get(label, ""),
                              CONF_UNCERTAIN_DEFAULT)

WEB_PORT = 5000

# Nudge-to-resettle: once settled, if ANY tracked die's center moves more
# than this many pixels since the lock, release back to "watching" so the
# system re-reads.  Lets a player nudge a single misread die to retry
# without disturbing the rest of the roll.
NUDGE_PIXEL_THRESHOLD = 20


# ── Label stabilizer (per-tracker-id majority vote) ─────────────────────────
class LabelStabilizer:
    def __init__(self, history_len=LABEL_HISTORY_LEN,
                 min_votes=LABEL_MIN_VOTES):
        self.history_len = history_len
        self.min_votes   = min_votes
        self._hist = defaultdict(lambda: deque(maxlen=self.history_len))

    def reset(self):
        self._hist.clear()

    def update(self, tracker_ids, class_ids, confidences):
        out = []
        for tid, cid, conf in zip(tracker_ids, class_ids, confidences):
            self._hist[tid].append((cid, conf))
            votes = Counter(v[0] for v in self._hist[tid])
            top_class, top_n = votes.most_common(1)[0]
            mean_conf = (sum(v[1] for v in self._hist[tid] if v[0] == top_class)
                         / max(top_n, 1))
            stable = top_n >= self.min_votes
            out.append((int(top_class), float(mean_conf), bool(stable)))
        return out


def make_tracker():
    # ByteTrack (PC) or a numpy IoU tracker (Pi) — selected by the backend.
    return backend.make_tracker()


# ── Session state ───────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.rolls: list[dict] = []
        self.next_roll_id      = 1
        self.active_player     = "P1"
        self.player1_name      = "Player 1"
        self.player2_name      = "Player 2"
        self.started_at        = time.time()

    def record(self, dice_type: str, results: list[str],
               confidences: list[float],
               rejected: bool = False,
               raw_image_path: str | None = None) -> dict:
        roll = {
            "roll_id":         self.next_roll_id,
            "timestamp":       time.time(),
            "player":          self.active_player,
            "dice_type":       dice_type,
            "results":         results,
            "confidences":     [round(c, 3) for c in confidences],
            "rejected":        rejected,
            "raw_image_path":  raw_image_path,
        }
        self.rolls.append(roll)
        self.next_roll_id += 1
        return roll

    def undo(self) -> dict | None:
        if not self.rolls:
            return None
        return self.rolls.pop()

    def save(self):
        payload = {
            "started_at":     self.started_at,
            "saved_at":       time.time(),
            "active_player":  self.active_player,
            "player1_name":   self.player1_name,
            "player2_name":   self.player2_name,
            "next_roll_id":   self.next_roll_id,
            "rolls":          self.rolls,
        }
        SESSION_JSON.write_text(json.dumps(payload, indent=2))

        with open(SESSION_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["roll_id", "timestamp", "player", "dice_type",
                        "results", "confidences", "rejected"])
            for r in self.rolls:
                w.writerow([
                    r["roll_id"], r["timestamp"], r["player"], r["dice_type"],
                    "|".join(r["results"]),
                    "|".join(str(c) for c in r["confidences"]),
                    r["rejected"],
                ])

    def load(self):
        if not SESSION_JSON.exists():
            print("No session.json to load.")
            return
        d = json.loads(SESSION_JSON.read_text())
        self.started_at     = d.get("started_at", time.time())
        self.active_player  = d.get("active_player", "P1")
        self.player1_name   = d.get("player1_name", "Player 1")
        self.player2_name   = d.get("player2_name", "Player 2")
        self.next_roll_id   = d.get("next_roll_id", 1)
        self.rolls          = d.get("rolls", [])
        print(f"Loaded session: {len(self.rolls)} rolls, "
              f"next id = {self.next_roll_id}")


# ── Drawing helpers ─────────────────────────────────────────────────────────
def draw_detections(frame, detections, model, label_states):
    out = frame.copy()
    for i in range(len(detections)):
        x1, y1, x2, y2 = [int(v) for v in detections.xyxy[i]]
        tid = (int(detections.tracker_id[i])
               if detections.tracker_id is not None else -1)
        cid, conf, stable = label_states[i]
        cls_name = model.names.get(cid, str(cid))
        if stable and conf >= uncertain_threshold(cls_name):
            color = (0, 255, 255)        # yellow: stable + confident
            thickness = 3
            text = f"#{tid} {cls_name} {int(round(conf*100))}%"
        elif stable:
            color = (0, 140, 255)        # orange: stable but UNCERTAIN
            thickness = 3
            text = f"#{tid} {cls_name}? {int(round(conf*100))}% NUDGE?"
        else:
            color = (140, 140, 140)      # gray: still settling
            thickness = 2
            text = f"#{tid} {cls_name}? {int(round(conf*100))}%"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def hud_line(frame, text, y, color=(255, 255, 255), scale=0.55):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1)


# ── Main loop ───────────────────────────────────────────────────────────────
def _has_display() -> bool:
    """True only if an OpenCV GUI window can actually open. The production
    Pi rig is a sealed box with NO monitor (phone is the only screen), so
    this is normally False there."""
    import os
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    try:
        probe = "._disp_probe"
        cv2.namedWindow(probe, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(probe)
        for _ in range(3):
            cv2.waitKey(1)
        return True
    except Exception:
        return False


def main():
    print("=" * 70)
    print("  Blood Bowl Dice Tracker")
    print("=" * 70)

    # Headless = no cv2 window; control + alignment via phone/GPIO only.
    # Forced with --headless, and the default on the Pi (no display, or the
    # torch-free onnx backend). --gui forces the dev preview window.
    headless = ("--headless" in sys.argv
                or (backend.BACKEND == "onnx" and "--gui" not in sys.argv)
                or not _has_display())
    if "--gui" in sys.argv:
        headless = False
    print(f"Display mode: {'HEADLESS (phone UI only)' if headless else 'GUI preview'}")

    session = Session()
    if "--load" in sys.argv:
        session.load()

    # Database: games/rolls are the primary record the web app reads.
    # The game row is created lazily on the first logged roll so aborted
    # startups don't litter the games list.
    db.init_db()
    game_id: int | None = None

    def ensure_game() -> int:
        nonlocal game_id
        if game_id is None:
            game_id = db.create_game(session.player1_name,
                                     session.player2_name)
            print(f"  [db] game {game_id} started")
        return game_id

    # Phone web UI (live control + post-game review). --no-web disables.
    web_control = None
    if "--no-web" not in sys.argv:
        try:
            from webapp import WebControl, lan_ip, start_in_thread
            web_control = WebControl()
            start_in_thread(web_control, port=WEB_PORT)
            print(f"Web UI — phone: http://{lan_ip()}:{WEB_PORT}/")
        except Exception as e:
            print(f"Web UI disabled: {e}")

    # Physical rig I/O (Pi only; a no-op stub on the PC). Button presses are
    # translated into the SAME synthetic keystrokes the main loop already
    # handles, so there is one code path for keyboard and buttons. A
    # player's OWN confirm button sets them active AND confirms in one press
    # (player attribution, per DESIGN.md): emit the player key then SPACE.
    import queue as _queue
    from hardware import Hardware
    hw_keys: "_queue.Queue[int]" = _queue.Queue()
    _BTN_KEYS = {"p1": [ord('1'), 32], "p2": [ord('2'), 32],
                 "reject": [ord('r')], "undo": [8]}

    def _on_button(name: str):
        for k in _BTN_KEYS.get(name, []):
            hw_keys.put(k)
        print(f"  [button] {name}")

    hw = Hardware(on_event=_on_button)
    if hw.available:
        print("Physical buttons/LEDs active (GPIO).")
    if "--no-web" in sys.argv and not hw.available:
        print("(no web, no GPIO — keyboard only)")

    print(f"Models dir: {MODELS_DIR}  (backend: {backend.BACKEND})")
    # Load whatever per-type models exist; the combined model is the
    # production path. The dedicated block/d6/d16 ONNX are full-frame and
    # optional (manual override on the PC) — on a fresh Pi only
    # combined.onnx may be present, which is fine.
    models = {}
    model_meta = {}
    for k in ("block", "d6", "d16"):
        p = MODELS_DIR / f"{k}.onnx"
        if p.exists():
            models[k] = backend.load_model(p)
            model_meta[k] = load_model_meta(k)
    # The combined 27-class model reads any dice type in one inference —
    # "auto" mode derives the dice type from the detected face labels, so
    # no manual switching is needed during play. Same Pi cost as a single
    # per-type model. Only enabled when the model file exists (i.e. has
    # been trained and passed the quality bar).
    if (MODELS_DIR / "combined.onnx").exists():
        models["auto"] = backend.load_model(MODELS_DIR / "combined.onnx")
        model_meta["auto"] = load_model_meta("combined")
        if model_meta["auto"]["tray_crop"]:
            print("  combined model is tray-crop-trained — inference will "
                  "crop to the tray ROI")
    if not models:
        print("ERROR: no models found in", MODELS_DIR)
        return
    current_type = "auto" if "auto" in models else next(iter(models))
    print(f"Dice mode: {current_type}"
          + ("" if "auto" in models else "  (no combined.onnx — manual "
             "type switching via phone/keys)"))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera open at {actual_w}x{actual_h}")
    crop_rect = tray_crop_rect(actual_w, actual_h)

    # IR-mode self-check before starting — models only know IR frames.
    # Auto-exposure takes ~2s to settle after open and reads high during
    # settling, so warm up generously and take the median of 5 probes.
    t_warm = time.time()
    while time.time() - t_warm < 2.0:
        cap.read()
    devs = []
    for _ in range(5):
        ret, probe = cap.read()
        if ret:
            devs.append(color_deviation(probe))
    day_mode = False
    if devs:
        dev = sorted(devs)[len(devs) // 2]
        day_mode = dev >= DAY_MODE_DEVIATION
        print(f"Lighting self-check: color deviation {dev:.1f} -> "
              f"{'DAY mode' if day_mode else 'IR mode'}")
        if day_mode:
            print("!" * 70)
            print("  WARNING: camera is in DAY mode (color image).")
            print("  Models are trained on IR frames — reads WILL degrade.")
            print("  Check the photoresistor light shield is seated properly.")
            print("!" * 70)
            # Headless rig has no console to answer a prompt — proceed with
            # the warning (the live HUD/phone day-mode flag stays on, and
            # the re-check fires every IR_CHECK_INTERVAL frames). Only the
            # interactive GUI path blocks for a y/N.
            if not headless and \
                    input("  Continue anyway? [y/N] ").strip().lower() != "y":
                cap.release()
                print("Aborted — fix the light shield and restart.")
                return
            if headless:
                print("  (headless: continuing; day-mode flag stays live)")
    print()

    # Camera alignment pre-flight — the rig moves between sessions and the
    # model only knows the calibrated tray perspective. Headless = confirm
    # from the phone (/align); GUI = the on-screen overlay.
    if headless:
        aligned = alignment_check_web(cap, actual_w, actual_h, web_control)
    else:
        aligned = alignment_check(cap, actual_w, actual_h)
    if not aligned:
        cap.release()
        print("Aborted at alignment check.")
        return
    print("Controls:")
    print("  1/2: active player    B/D/X: model    SPACE: confirm    BKSP: undo")
    print("  R: reject + save for retrain    S: save session    Q: quit")
    print()

    tracker    = make_tracker()
    smoother   = backend.make_smoother(SMOOTHER_LENGTH)
    label_stab = LabelStabilizer()

    # Roll-level state machine:
    # - "watching"  : default, looking for a settled roll
    # - "settled"   : current detections are locked + ready for SPACE/REJECT
    # - "confirmed" : last roll confirmed, waiting for dice to move so next can start
    state = "watching"
    last_settled_frame = None
    last_settled_labels: list[tuple[int, float, bool]] = []
    last_settled_dets = None
    last_settled_d16: list[dict] = []   # d16 geometry verdicts at settle
    # tracker_id -> (cx, cy) at settle time, used for nudge detection
    last_settled_centroids: dict[int, tuple[int, int]] = {}
    flash_until = 0.0
    count_hist: deque = deque(maxlen=COUNT_STABLE_FRAMES)

    win = "Dice Tracker - SPACE confirm, R reject, BKSP undo, S save, Q quit"
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, actual_w, actual_h)

    last_time  = time.time()
    fps_smooth = 0.0
    frame_count = 0

    def reset_for_next_roll():
        nonlocal tracker, smoother, last_settled_dets, last_settled_labels, \
            last_settled_d16, state
        tracker = make_tracker()
        smoother = backend.make_smoother(SMOOTHER_LENGTH)
        label_stab.reset()
        last_settled_dets   = None
        last_settled_labels = []
        last_settled_d16    = []
        last_settled_centroids.clear()
        count_hist.clear()
        state = "watching"

    def current_centroids_by_tid(dets) -> dict[int, tuple[int, int]]:
        """Return tracker_id -> (cx, cy) for current frame detections."""
        out: dict[int, tuple[int, int]] = {}
        if dets is None or dets.tracker_id is None:
            return out
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            x1, y1, x2, y2 = (int(v) for v in dets.xyxy[i])
            out[tid] = ((x1 + x2) // 2, (y1 + y2) // 2)
        return out

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Apply pending requests from the phone web UI
        if web_control is not None:
            req_type, req_player = web_control.take_requests()
            if req_type and req_type in models and req_type != current_type:
                current_type = req_type
                reset_for_next_roll()
                print(f"  model -> {current_type} (web)")
            if req_player and req_player != session.active_player:
                session.active_player = req_player
                print(f"  active player -> {req_player} (web)")

        # Live IR re-check — lighting can change mid-session (sun, lamps).
        frame_count += 1
        if frame_count % IR_CHECK_INTERVAL == 0:
            was_day = day_mode
            day_mode = color_deviation(frame) >= DAY_MODE_DEVIATION
            if day_mode and not was_day:
                print("  [WARNING] camera flipped to DAY mode — reads unreliable!")
            elif was_day and not day_mode:
                print("  [ok] camera back in IR mode")

        model = models[current_type]
        detections = predict_detections(model, frame,
                                        model_meta[current_type], crop_rect)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

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

        n_det = len(detections)
        stable_count = sum(1 for _, _, s in label_states if s)
        # In auto mode the dice type is whatever the combined model sees
        detected_type = (majority_type(
            [model.names.get(c, str(c)) for c, _, _ in label_states])
            if current_type == "auto" else current_type)
        if detected_type == "d16":
            # d16 settle gates on STABLE boxes only: a die shows 2-3 small
            # glyph boxes and the marginal one flickers at the conf
            # threshold — requiring the raw count to hold for the full
            # window means the roll never settles. Transients are ignored
            # and excluded from the read; the adjacency layer makes a
            # 2-face read safe (sides determine the top).
            stable_ids = [i for i, (_c, _cf, s) in enumerate(label_states)
                          if s]
            n_units = (len(d16_geometry.cluster_faces(
                [list(map(float, detections.xyxy[i])) for i in stable_ids]))
                if stable_ids else 0)
            all_stable = len(stable_ids) > 0
            count_hist.append(("d16", len(stable_ids), n_units))
        else:
            stable_ids = list(range(n_det))
            all_stable = (n_det > 0 and stable_count == n_det)
            count_hist.append(n_det)
        count_stable = (len(count_hist) == COUNT_STABLE_FRAMES
                        and len(set(count_hist)) == 1)

        # State machine
        if state == "watching":
            if all_stable and count_stable:
                state = "settled"
                last_settled_frame  = frame.copy()
                # d16: lock only the stable boxes — transients are not part
                # of the read, and a vanishing transient must not trip the
                # nudge-release either.
                if detected_type == "d16" and len(stable_ids) < n_det:
                    keep = np.array(stable_ids, dtype=int)
                    last_settled_dets   = detections[keep]
                    last_settled_labels = [label_states[i]
                                           for i in stable_ids]
                else:
                    last_settled_dets   = detections
                    last_settled_labels = list(label_states)
                last_settled_centroids = current_centroids_by_tid(
                    last_settled_dets)
                flash_until = time.time() + 1.0   # initial settle flash
                print(f"  [settled] {len(last_settled_labels)} dice, labels: "
                      f"{[model.names.get(c, c) for c, _, _ in last_settled_labels]}")
                # d16 geometric cross-check: the 3 face boxes of a die are
                # physically married (adjacency table in d16_geometry).
                last_settled_d16 = []
                if detected_type == "d16":
                    last_settled_d16 = d16_geometry.analyze_roll(
                        [model.names.get(c, str(c))
                         for c, _, _ in last_settled_labels],
                        [list(map(float, last_settled_dets.xyxy[i]))
                         for i in range(len(last_settled_dets))],
                        [conf for _, conf, _ in last_settled_labels])
                    for v in last_settled_d16:
                        if v["status"] == "impossible":
                            print(f"  [d16-check] IMPOSSIBLE read — "
                                  f"{v['note']} — nudge the die to re-read")
                        elif v["status"] == "deduced":
                            print(f"  [d16-check] {v['note']}")

        elif state == "settled":
            # Release back to "watching" if labels became unstable...
            if not all_stable:
                state = "watching"
                last_settled_centroids.clear()
            else:
                # ...OR if any locked die has moved more than NUDGE_PIXEL_THRESHOLD
                # since settle. Lets a player nudge a misread die to re-read it
                # without losing the rest of the roll.
                now_centroids = current_centroids_by_tid(detections)
                for tid, (cx, cy) in last_settled_centroids.items():
                    if tid not in now_centroids:
                        # This die disappeared from detections — treat as motion
                        state = "watching"
                        last_settled_centroids.clear()
                        print(f"  [nudged] tracker #{tid} disappeared, re-reading")
                        break
                    ncx, ncy = now_centroids[tid]
                    if (abs(ncx - cx) > NUDGE_PIXEL_THRESHOLD or
                            abs(ncy - cy) > NUDGE_PIXEL_THRESHOLD):
                        state = "watching"
                        last_settled_centroids.clear()
                        print(f"  [nudged] tracker #{tid} moved "
                              f"({cx},{cy})->({ncx},{ncy}), re-reading")
                        break

        elif state == "confirmed":
            # Wait for dice to physically move (count drops to 0 or all
            # tracker IDs change) before allowing next roll to settle.
            # Practically: any change in count or trackers triggers reset.
            if n_det == 0:
                reset_for_next_roll()

        # Annotate
        annotated = draw_detections(frame, detections, model, label_states)

        # Green border while settled
        if state == "settled":
            cv2.rectangle(annotated, (5, 5),
                          (actual_w - 5, actual_h - 5),
                          (0, 255, 0), 6)
        elif state == "confirmed":
            cv2.rectangle(annotated, (5, 5),
                          (actual_w - 5, actual_h - 5),
                          (200, 100, 200), 4)   # purple = confirmed, waiting

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            inst = 1.0 / dt
            fps_smooth = (0.9 * fps_smooth + 0.1 * inst
                          if fps_smooth > 0 else inst)

        # HUD
        player_label = (session.player1_name if session.active_player == "P1"
                        else session.player2_name)
        dice_label = (f"auto[{detected_type or '?'}]"
                      if current_type == "auto" else current_type)
        hud_line(annotated,
                 f"Player: {session.active_player} ({player_label})    "
                 f"Dice: {dice_label}    State: {state}    "
                 f"FPS: {fps_smooth:4.1f}", 25)
        hud_line(annotated,
                 f"Rolls logged: {len(session.rolls)}    "
                 f"Detections: {n_det}    Stable: {stable_count}/{n_det}", 50)
        if day_mode:
            hud_line(annotated,
                     "DAY MODE - photoresistor seeing light, reads unreliable!",
                     75, color=(0, 0, 255), scale=0.6)
        if state == "settled" and any(v["status"] == "impossible"
                                      for v in last_settled_d16):
            hud_line(annotated,
                     "D16 READ GEOMETRICALLY IMPOSSIBLE - nudge die to re-read",
                     100, color=(0, 0, 255), scale=0.6)
        hint = {
            "watching":  "Roll dice...",
            "settled":   "SPACE = confirm  |  R = reject (save for retrain)",
            "confirmed": "Pick up dice, roll again...",
        }.get(state, "")
        hud_line(annotated, hint, actual_h - 20, color=(0, 255, 255), scale=0.6)

        # Live read shared by the phone UI and the physical OLEDs.
        dice_status = [{
            "label": model.names.get(cid, str(cid)),
            "conf": int(round(conf * 100)),
            "stable": bool(stable),
            "uncertain": bool(stable and conf < uncertain_threshold(
                model.names.get(cid, str(cid)))),
        } for cid, conf, stable in label_states]
        if web_control is not None:
            recent = [(f"#{lr['roll_id']} {lr['player']} {lr['dice_type']}: "
                       + ", ".join(lr["results"])
                       + ("  [rejected]" if lr["rejected"] else ""))
                      for lr in session.rolls[-5:][::-1]]
            web_control.update_status(
                state=state, dice_type=current_type,
                detected_type=detected_type,
                player=session.active_player,
                p1_name=session.player1_name, p2_name=session.player2_name,
                rolls=len(session.rolls), dice=dice_status,
                recent=recent, day_mode=day_mode)

        # Physical OLEDs mirror the live read; LEDs signal state — both
        # players' LEDs glow when a roll is settled (press your confirm),
        # the active player's stays lit otherwise.
        if hw.available:
            hw.show(session.active_player, dice_status, state)
            settled = (state == "settled")
            hw.set_led("p1", settled or session.active_player == "P1")
            hw.set_led("p2", settled or session.active_player == "P2")
            hw.set_led("reject", settled)
            hw.set_led("undo", bool(session.rolls))

        if headless:
            # No window: stream the annotated frame to the phone and pace
            # the loop with a short sleep (waitKey is the GUI delay we'd
            # otherwise rely on). Input comes from GPIO buttons + phone.
            if web_control is not None and frame_count % 2 == 0:
                ok, buf = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    web_control.set_frame(buf.tobytes())
            time.sleep(0.005)
            key = 255
        else:
            cv2.imshow(win, annotated)
            key = cv2.waitKey(1) & 0xFF

        # Phone actions (reject / undo / confirm) -> the same synthetic keys
        # the dispatch below already handles, so phone + buttons + keyboard
        # share one path.
        if web_control is not None:
            act = web_control.take_action()
            if act == "reject":
                hw_keys.put(ord('r'))
            elif act == "undo":
                hw_keys.put(8)
            elif act == "confirm":
                hw_keys.put(32)

        # Drain one queued keystroke (GPIO button or phone action) per frame
        # when no physical keyboard key is pending (255 = none). Multi-key
        # button actions (e.g. P1 = '1' then SPACE) drain across consecutive
        # frames, which the state machine handles fine.
        if key == 255 and not hw_keys.empty():
            try:
                key = hw_keys.get_nowait()
            except Exception:
                pass

        if key in (ord('q'), 27):
            if session.rolls:
                session.save()
                print(f"  AUTO-SAVED {len(session.rolls)} rolls to "
                      f"{SESSION_JSON} on quit")
            if game_id is not None:
                db.end_game(game_id)
                print(f"  [db] game {game_id} closed")
            break

        elif key == ord('1'):
            session.active_player = "P1"
            print(f"  active player -> P1 ({session.player1_name})")
        elif key == ord('2'):
            session.active_player = "P2"
            print(f"  active player -> P2 ({session.player2_name})")

        elif key == ord('a'):
            if "auto" in models:
                current_type = "auto"; reset_for_next_roll()
                print(f"  model -> {current_type}")
            else:
                print("  (no combined.onnx — auto mode unavailable)")
        elif key == ord('b'):
            current_type = "block"; reset_for_next_roll()
            print(f"  model -> {current_type}")
        elif key == ord('d'):
            current_type = "d6"; reset_for_next_roll()
            print(f"  model -> {current_type}")
        elif key == ord('x'):
            current_type = "d16"; reset_for_next_roll()
            print(f"  model -> {current_type}")

        elif key == 32:   # SPACE = confirm
            if state == "settled":
                results_str = [model.names.get(c, str(c))
                               for c, _, _ in last_settled_labels]
                confs       = [conf for _, conf, _ in last_settled_labels]
                # d16: when the verified adjacency table contradicts a
                # low-confidence top face, the geometry wins (two flanking
                # sides uniquely determine the top). Inactive until
                # d16_geometry.ADJACENCY_VERIFIED is set.
                for v in last_settled_d16:
                    if (v["status"] == "deduced"
                            and d16_geometry.ADJACENCY_VERIFIED
                            and v["indices"]):
                        i = v["indices"][0]
                        if (i < len(results_str)
                                and confs[i] < CONF_UNCERTAIN["d16"]):
                            old = results_str[i]
                            results_str[i] = f"D16_{v['deduced_top']}"
                            print(f"  [d16-check] corrected {old} -> "
                                  f"{results_str[i]} ({v['note']})")
                logged_type = (majority_type(results_str) or "unknown"
                               if current_type == "auto" else current_type)
                rec = session.record(
                    dice_type=logged_type,
                    results=results_str,
                    confidences=confs,
                    rejected=False,
                )
                db.add_roll(ensure_game(), rec["roll_id"], rec["player"],
                            logged_type, results_str, rec["confidences"])
                print(f"  CONFIRMED roll #{rec['roll_id']}: "
                      f"{session.active_player} {logged_type} -> {results_str}")
                state = "confirmed"
            else:
                print("  (SPACE ignored — no settled roll to confirm)")

        elif key == ord('r'):
            if state == "settled":
                ts = time.strftime("%Y%m%d_%H%M%S")
                reject_labels = [model.names.get(c, str(c))
                                 for c, _, _ in last_settled_labels]
                logged_type = (majority_type(reject_labels) or "unknown"
                               if current_type == "auto" else current_type)
                reject_dir = RETRAIN_DIR / logged_type
                reject_dir.mkdir(exist_ok=True)
                img_path = reject_dir / f"reject_{ts}.jpg"
                cv2.imwrite(str(img_path), last_settled_frame)
                # Also write predicted labels alongside the frame
                tids = (last_settled_dets.tracker_id.tolist()
                        if last_settled_dets.tracker_id is not None
                        else [None] * len(last_settled_dets))
                meta = {
                    "timestamp":  time.time(),
                    "dice_type":  logged_type,
                    "predicted":  [model.names.get(c, str(c))
                                   for c, _, _ in last_settled_labels],
                    "confidences": [round(c, 3)
                                    for _, c, _ in last_settled_labels],
                    "tracker_ids": [int(t) if t is not None else None
                                    for t in tids],
                    "boxes": [list(map(int, last_settled_dets.xyxy[i]))
                              for i in range(len(last_settled_dets))],
                }
                (img_path.with_suffix(".json")
                 ).write_text(json.dumps(meta, indent=2))
                results_str = meta["predicted"]
                rec = session.record(
                    dice_type=logged_type,
                    results=results_str,
                    confidences=meta["confidences"],
                    rejected=True,
                    raw_image_path=str(img_path),
                )
                db.add_roll(ensure_game(), rec["roll_id"], rec["player"],
                            logged_type, results_str, rec["confidences"],
                            rejected=True, raw_image_path=str(img_path))
                print(f"  REJECTED — frame saved to {img_path}")
                state = "confirmed"
            else:
                print("  (R ignored — no settled roll to reject)")

        elif key == 8:   # BACKSPACE = undo
            r = session.undo()
            if r is not None:
                if game_id is not None:
                    db.delete_last_roll(game_id)
                print(f"  UNDID roll #{r['roll_id']}: {r['results']}")
                reset_for_next_roll()
            else:
                print("  (nothing to undo)")

        elif key == ord('s'):
            session.save()
            print(f"  SAVED to {SESSION_JSON} and {SESSION_CSV}")

        elif key == ord('l'):
            session.load()

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    hw.cleanup()
    print(f"Total rolls in session: {len(session.rolls)}")
    print("Done.")


if __name__ == "__main__":
    main()
