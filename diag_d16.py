"""20-second live d16 diagnostic: logs per-frame what the detection
stack sees so a stuck settle can be diagnosed from data, not guesses.

Put a d16 in the tray (anywhere it was refusing to settle), then:
    python diag_d16.py
Watch nothing, touch nothing — it exits by itself after ~20s and writes
diag_d16_log.txt + diag_d16_frame.jpg next to this script.
"""

import time
from collections import Counter

import cv2
import supervision as sv
from ultralytics import YOLO

import d16_geometry
from dice_tracker import (
    CAMERA_INDEX, COUNT_STABLE_FRAMES, MODELS_DIR, RESOLUTION,
    LabelStabilizer, load_model_meta, make_tracker, predict_detections,
    tray_crop_rect,
)

LOG = open("diag_d16_log.txt", "w")


def log(s):
    print(s)
    LOG.write(s + "\n")


model = YOLO(str(MODELS_DIR / "combined_crop.onnx"), task="detect")
meta = load_model_meta("combined_crop")
log(f"meta: {meta}")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
crop_rect = tray_crop_rect(aw, ah)
log(f"camera {aw}x{ah}, crop_rect {crop_rect}")

tracker = make_tracker()
smoother = sv.DetectionsSmoother(length=5)
stab = LabelStabilizer()

t0 = time.time()
frame_n = 0
saved = False
while time.time() - t0 < 20:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_n += 1
    raw = predict_detections(model, frame, meta, crop_rect)
    dets = tracker.update_with_detections(raw)
    dets = smoother.update_with_detections(dets)
    if dets.tracker_id is not None and len(dets) > 0:
        states = stab.update(dets.tracker_id.tolist(),
                             dets.class_id.tolist(),
                             dets.confidence.tolist())
    else:
        states = []
    stable_ids = [i for i, (_c, _cf, s) in enumerate(states) if s]
    n_units = (len(d16_geometry.cluster_faces(
        [list(map(float, dets.xyxy[i])) for i in stable_ids]))
        if stable_ids else 0)
    raw_desc = Counter(model.names[int(c)] for c in raw.class_id) \
        if len(raw) else {}
    trk = (dets.tracker_id.tolist() if dets.tracker_id is not None
           and len(dets) else [])
    log(f"f{frame_n:03d} raw={len(raw)} {dict(raw_desc)} "
        f"tracked={len(dets)} ids={trk} "
        f"stable={len(stable_ids)} units={n_units}")
    if not saved and frame_n == 60:
        cv2.imwrite("diag_d16_frame.jpg", frame)
        saved = True
cap.release()
LOG.close()
print("\nDone -> diag_d16_log.txt + diag_d16_frame.jpg")
