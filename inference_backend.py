"""
inference_backend.py
--------------------
Single seam between dice_tracker/eval_harness and the detection stack, so
the SAME app runs on the PC (ultralytics + supervision) and the Pi
(torch-free onnx_backend) with no logic changes.

Selection (override with env DICE_BACKEND=onnx|ultralytics):
  - "ultralytics" when torch + ultralytics + supervision import (PC/dev)
  - "onnx"        otherwise (the Pi: onnxruntime + numpy only)

Uniform surface used by the app:
  BACKEND                         -> "onnx" | "ultralytics"
  load_model(onnx_path)           -> model with .names (dict id->label)
  predict_detections(model, frame, meta, crop_rect) -> Detections-like
  make_tracker()                  -> tracker with update_with_detections
  make_smoother(length)           -> smoother with update_with_detections
Detections-like always has .xyxy .class_id .confidence .tracker_id, and
supports len()/bool-mask/int-array indexing.

predict_detections / tray-crop+pad geometry is identical on both paths
(parity verified bit-for-bit on banked frames, 2026-06-14).
"""

import os

# tracker tunables — one source of truth for both backends
TRACKER_KW = dict(frame_rate=30, lost_track_buffer=120,
                  minimum_consecutive_frames=3,
                  track_activation_threshold=0.30)


def _pick_backend() -> str:
    forced = os.environ.get("DICE_BACKEND")
    if forced in ("onnx", "ultralytics"):
        return forced
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
        import supervision  # noqa: F401
        return "ultralytics"
    except Exception:
        return "onnx"


BACKEND = _pick_backend()


CONF_THRESHOLD = 0.40   # shared by both backends


def _crop_with_pad(frame, meta, crop_rect):
    """Tray-crop + pad geometry shared by both backends. Returns
    (src_image, ox, oy). Identical math to the original dice_tracker
    implementation — bottom-wall dice project below the ROI rect, so the
    crop is padded (mostly downward) for crop-trained models."""
    if not (meta.get("tray_crop") and crop_rect is not None):
        return frame, 0, 0
    x, y, w, h = crop_rect
    pad = int(meta.get("pad", 0))
    padb = int(meta.get("pad_bottom", 0))
    H, W = frame.shape[:2]
    ox = max(0, x - pad)
    oy = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad + padb)
    return frame[oy:y2, ox:x2], ox, oy


if BACKEND == "ultralytics":
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    def load_model(path):
        return YOLO(str(path), task="detect")

    def predict_detections(model, frame, meta, crop_rect):
        """Crop-match inference; full-frame coords out. agnostic_nms=True so
        one physical die can't survive NMS as two overlapping classes."""
        src, ox, oy = _crop_with_pad(frame, meta, crop_rect)
        results = model.predict(source=src, conf=CONF_THRESHOLD,
                                agnostic_nms=True, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)
        if (ox or oy) and len(dets) > 0:
            dets.xyxy = dets.xyxy + np.array([ox, oy, ox, oy],
                                             dtype=dets.xyxy.dtype)
        return dets

    def make_tracker():
        return sv.ByteTrack(**TRACKER_KW)

    def make_smoother(length=5):
        return sv.DetectionsSmoother(length=length)

else:  # onnx (Pi)
    from onnx_backend import (ByteTrackLite, DetectionsSmootherLite,
                              OnnxModel)
    from onnx_backend import predict_detections as predict_detections  # noqa: F401

    def load_model(path):
        return OnnxModel(str(path))

    def make_tracker():
        return ByteTrackLite(**TRACKER_KW)

    def make_smoother(length=5):
        return DetectionsSmootherLite(length=length)
