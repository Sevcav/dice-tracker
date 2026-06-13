"""Offline pre-check: run combined_crop (padded tray-crop inference +
adjacency deduction) on the d16 frames the BASELINE model misread today,
and score against the recorded value truth. No smoothing stack — single
frames — so this is indicative, not the official eval number."""
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, r"C:\Users\chapm\Dice Code")
import d16_geometry
import dice_tracker as dt

model = YOLO(str(dt.MODELS_DIR / "combined_crop.onnx"), task="detect")
meta = dt.load_model_meta("combined_crop")
crop_rect = dt.tray_crop_rect(1280, 720)
print("meta:", meta)

bank = Path(r"C:\Users\chapm\Dice Code\retrain_candidates\d16")
n_ok = n_tot = 0
for jp in sorted(bank.glob("eval_miss_*.json")):
    rec = json.loads(jp.read_text())
    if rec.get("truth_mode") != "value" or len(rec["truth"]) != 1:
        continue
    truth = rec["truth"][0]
    frame = cv2.imread(str(jp.with_suffix(".jpg")))
    if frame is None or frame.shape[:2] != (720, 1280):
        continue
    dets = dt.predict_detections(model, frame, meta, crop_rect)
    labels = [model.names[int(c)] for c in dets.class_id] if len(dets) else []
    verdicts = d16_geometry.analyze_roll(
        labels, [list(map(float, dets.xyxy[i])) for i in range(len(dets))],
        [float(c) for c in dets.confidence]) if labels else []
    if len(verdicts) == 1:
        pred = str(verdicts[0]["top"])
        status = verdicts[0]["status"]
    else:
        pred, status = "?", f"{len(verdicts)} dice clusters"
    ok = pred == truth
    n_tot += 1
    n_ok += ok
    print(f"  {jp.stem}: baseline read {rec['predicted'][0]:>2}, "
          f"truth {truth:>2} -> new model reads {pred:>2} "
          f"({status}) {'OK' if ok else 'MISS'}")
print(f"\nnew model on baseline-miss frames: {n_ok}/{n_tot}")
