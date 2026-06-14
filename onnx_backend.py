"""
onnx_backend.py
---------------
Torch-free / supervision-free inference + tracking for the Raspberry Pi.

dice_tracker.py was built on ultralytics.YOLO + supervision (ByteTrack,
DetectionsSmoother). Both pull in torch, which is too heavy for the Pi.
This module reproduces the exact same interface using ONLY onnxruntime +
numpy + opencv, so the Pi runs the identical app logic with a light
dependency set.

Provided so dice_tracker can stay backend-agnostic:
  OnnxModel(path)              -> .names (dict id->label), .predict_raw(frame)
  Detections                   -> .xyxy .class_id .confidence .tracker_id
                                  (+ __len__, integer/array indexing)
  predict_detections(model, frame, meta, crop_rect) -> Detections
                                  (tray-crop + pad, agnostic NMS, full-frame
                                   coords — same contract as dice_tracker's)
  ByteTrackLite(...)           -> .update_with_detections(dets)
  DetectionsSmootherLite(len)  -> .update_with_detections(dets)

Decode target (verified against combined.onnx): output0 [1, 31, 8400],
YOLOv11 transposed layout — 4 bbox (cx,cy,w,h, in letterboxed 640 px) +
27 class scores, 8400 anchors. Class names come from the ONNX file's
'names' metadata, so ultralytics is never needed to read labels.

Parity with the ultralytics path is checked by tools/_check_parity (the
banked-frame oracle in training/_parity_oracle.json).
"""

from __future__ import annotations

import ast
import json

import cv2
import numpy as np
import onnxruntime as ort

# Match dice_tracker.CONF_THRESHOLD / NMS defaults.
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.45          # ultralytics default NMS IoU
INPUT_SIZE = 640


# ── Detections container (supervision.Detections-compatible subset) ──────────
class Detections:
    """Minimal stand-in for sv.Detections. Holds float xyxy (N,4), int
    class_id (N,), float confidence (N,), and optional int tracker_id (N,)."""

    def __init__(self, xyxy, class_id=None, confidence=None, tracker_id=None):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        n = len(self.xyxy)
        self.class_id = (np.asarray(class_id, dtype=int)
                         if class_id is not None else np.zeros(n, int))
        self.confidence = (np.asarray(confidence, dtype=np.float32)
                           if confidence is not None else np.ones(n, np.float32))
        self.tracker_id = (np.asarray(tracker_id, dtype=int)
                           if tracker_id is not None else None)

    def __len__(self):
        return len(self.xyxy)

    @classmethod
    def empty(cls):
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, int),
                   np.zeros(0, np.float32), None)

    def __getitem__(self, idx):
        """Supports boolean mask, int-array, and slice indexing (the forms
        dice_tracker uses)."""
        idx = np.asarray(idx) if isinstance(idx, (list, np.ndarray)) else idx
        return Detections(
            self.xyxy[idx],
            self.class_id[idx],
            self.confidence[idx],
            self.tracker_id[idx] if self.tracker_id is not None else None,
        )


# ── Model wrapper ────────────────────────────────────────────────────────────
class OnnxModel:
    def __init__(self, path: str, providers=None):
        self.session = ort.InferenceSession(
            str(path),
            providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.names = self._read_names()

    def _read_names(self) -> dict:
        meta = self.session.get_modelmeta().custom_metadata_map
        raw = meta.get("names")
        if not raw:
            return {}
        try:                       # stored as a python-dict repr
            return {int(k): v for k, v in ast.literal_eval(raw).items()}
        except Exception:
            try:
                return {int(k): v for k, v in json.loads(raw).items()}
            except Exception:
                return {}

    def predict_raw(self, bgr: np.ndarray, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
                    agnostic=True):
        """Run inference on a BGR image. Returns (xyxy, cls, conf) in the
        image's OWN pixel coordinates (letterbox undone)."""
        blob, scale, (padx, pady) = _letterbox(bgr, INPUT_SIZE)
        out = self.session.run(None, {self.input_name: blob})[0]  # (1,31,8400)
        return _decode(out, scale, padx, pady, conf, iou, agnostic)


# ── Pre/post-processing ──────────────────────────────────────────────────────
def _letterbox(bgr, size):
    """Resize keeping aspect, pad to size x size (gray 114) — matches
    ultralytics' default letterbox so geometry/scale agree."""
    h, w = bgr.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    canvas[pady:pady + nh, padx:padx + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])  # (1,3,H,W)
    return blob, scale, (padx, pady)


def _decode(out, scale, padx, pady, conf_thr, iou_thr, agnostic):
    # out: (1, 4+nc, 8400) -> (8400, 4+nc)
    p = out[0].T
    boxes_cxcywh = p[:, :4]
    scores_all = p[:, 4:]
    cls = scores_all.argmax(1)
    conf = scores_all.max(1)
    keep = conf >= conf_thr
    if not keep.any():
        return (np.zeros((0, 4), np.float32),
                np.zeros(0, int), np.zeros(0, np.float32))
    boxes_cxcywh, cls, conf = boxes_cxcywh[keep], cls[keep], conf[keep]

    # cxcywh (letterbox px) -> xyxy in original image px
    cx, cy, bw, bh = boxes_cxcywh.T
    x1 = (cx - bw / 2 - padx) / scale
    y1 = (cy - bh / 2 - pady) / scale
    x2 = (cx + bw / 2 - padx) / scale
    y2 = (cy + bh / 2 - pady) / scale
    xyxy = np.stack([x1, y1, x2, y2], 1)

    idx = _nms(xyxy, conf, cls, iou_thr, agnostic)
    return xyxy[idx], cls[idx], conf[idx]


def _nms(boxes, scores, cls, iou_thr, agnostic):
    """Greedy NMS. agnostic=True suppresses across classes (matches
    dice_tracker's agnostic_nms=True — one box per physical die)."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        suppress = iou > iou_thr
        if not agnostic:
            suppress &= (cls[rest] == cls[i])
        order = rest[~suppress]
    return np.array(keep, int)


# ── Same contract as dice_tracker.predict_detections ─────────────────────────
def predict_detections(model: OnnxModel, frame, meta: dict,
                       crop_rect) -> Detections:
    """Tray-crop (+ pad) for crop-trained models, run inference, return
    detections in FULL-FRAME coords. Mirrors dice_tracker.predict_detections
    exactly (including the pad/pad_bottom sidecar handling)."""
    ox = oy = 0
    src = frame
    if meta.get("tray_crop") and crop_rect is not None:
        x, y, w, h = crop_rect
        pad = int(meta.get("pad", 0))
        padb = int(meta.get("pad_bottom", 0))
        H, W = frame.shape[:2]
        ox = max(0, x - pad)
        oy = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad + padb)
        src = frame[oy:y2, ox:x2]

    xyxy, cls, conf = model.predict_raw(src)
    if len(xyxy) and (ox or oy):
        xyxy = xyxy + np.array([ox, oy, ox, oy], np.float32)
    return Detections(xyxy, cls, conf, None)


# ── Tracking + smoothing (supervision-free) ─────────────────────────────────
def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    inter_x1 = np.maximum(ax1[:, None], bx1[None])
    inter_y1 = np.maximum(ay1[:, None], by1[None])
    inter_x2 = np.minimum(ax2[:, None], bx2[None])
    inter_y2 = np.minimum(ay2[:, None], by2[None])
    iw = np.maximum(0, inter_x2 - inter_x1)
    ih = np.maximum(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = ((ax2 - ax1) * (ay2 - ay1))[:, None]
    area_b = ((bx2 - bx1) * (by2 - by1))[None]
    return inter / (area_a + area_b - inter + 1e-9)


class ByteTrackLite:
    """IoU-greedy tracker matching how dice_tracker uses sv.ByteTrack: stable
    integer IDs across frames for a near-static settled scene. Not the full
    ByteByte algorithm (no Kalman/two-stage), but for dice that come to rest
    in a fixed tray, greedy-IoU association gives the same persistent IDs the
    settle logic needs. Honoured params mirror make_tracker().
    """

    def __init__(self, frame_rate=30, lost_track_buffer=120,
                 minimum_consecutive_frames=3, track_activation_threshold=0.30,
                 iou_match=0.30):
        self.lost_buffer = lost_track_buffer
        self.min_hits = minimum_consecutive_frames
        self.act_thr = track_activation_threshold
        self.iou_match = iou_match
        self._next_id = 1
        self._tracks = {}   # id -> dict(box, hits, misses, active)

    def update_with_detections(self, dets: "Detections") -> "Detections":
        det_boxes = dets.xyxy
        det_conf = dets.confidence
        ids = [t for t in self._tracks]
        trk_boxes = np.array([self._tracks[t]["box"] for t in ids],
                             np.float32) if ids else np.zeros((0, 4), np.float32)

        assigned = -np.ones(len(det_boxes), int)
        if len(det_boxes) and len(trk_boxes):
            iou = _iou_matrix(det_boxes, trk_boxes)
            order = np.argsort(-det_conf)
            used_tracks = set()
            for di in order:
                ti = int(np.argmax(iou[di]))
                if iou[di, ti] >= self.iou_match and ids[ti] not in used_tracks:
                    assigned[di] = ids[ti]
                    used_tracks.add(ids[ti])

        seen = set()
        for di in range(len(det_boxes)):
            tid = assigned[di]
            if tid == -1:
                if det_conf[di] < self.act_thr:
                    continue
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"box": det_boxes[di], "hits": 0,
                                     "misses": 0, "active": False}
            t = self._tracks[tid]
            t["box"] = det_boxes[di]
            t["hits"] += 1
            t["misses"] = 0
            if t["hits"] >= self.min_hits:
                t["active"] = True
            assigned[di] = tid
            seen.add(tid)

        for tid in list(self._tracks):
            if tid not in seen:
                self._tracks[tid]["misses"] += 1
                if self._tracks[tid]["misses"] > self.lost_buffer:
                    del self._tracks[tid]

        # emit only currently-matched active tracks (matches ByteTrack output)
        keep = [di for di in range(len(det_boxes))
                if assigned[di] != -1 and self._tracks[assigned[di]]["active"]]
        if not keep:
            return Detections.empty()
        keep = np.array(keep, int)
        return Detections(det_boxes[keep], dets.class_id[keep],
                          det_conf[keep], assigned[keep])


class DetectionsSmootherLite:
    """Per-track rolling-mean box smoothing, like sv.DetectionsSmoother:
    averages each track's box over the last `length` frames to damp jitter."""

    def __init__(self, length=5):
        self.length = length
        self._hist = {}   # tracker_id -> list[box]

    def update_with_detections(self, dets: "Detections") -> "Detections":
        if dets.tracker_id is None or len(dets) == 0:
            return dets
        out = dets.xyxy.copy()
        present = set()
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            present.add(tid)
            h = self._hist.setdefault(tid, [])
            h.append(dets.xyxy[i])
            if len(h) > self.length:
                h.pop(0)
            out[i] = np.mean(h, axis=0)
        for tid in list(self._hist):
            if tid not in present:
                del self._hist[tid]
        return Detections(out, dets.class_id, dets.confidence, dets.tracker_id)
