"""
block_dice_classifier_onnx.py
-----------------------------
Drop-in replacement for BlockDiceClassifier that uses ONNX Runtime
instead of PyTorch. Used on the Raspberry Pi (no torch installed).

Usage:
    from classifier.block_dice_classifier_onnx import BlockDiceClassifier
    clf = BlockDiceClassifier()
    label, confidence = clf.predict(crop_bgr)
"""

import os
import json
import numpy as np
import cv2
import onnxruntime as ort

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH   = os.path.join(BASE, "classifier", "block_dice_model.onnx")
LABELS_PATH = os.path.join(BASE, "classifier", "labels.json")

# Human-readable display names for each class
DISPLAY_NAMES = {
    "both_down":   "Both Down",
    "pow":         "POW!",
    "push":        "Push",
    "player_down": "Player Down",
    "stumble":     "Stumble",
    "d6_bb_logo":  "BB Logo (6)",
    "d6_1":        "1",
    "d6_2":        "2",
    "d6_3":        "3",
    "d6_4":        "4",
    "d6_5":        "5",
}

# ImageNet normalization (matches PyTorch training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L channel for illumination invariance — matches PyTorch path."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


class BlockDiceClassifier:
    def __init__(self):
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(
                f"ONNX model not found at {ONNX_PATH}.\n"
                "Export it on Windows: python classifier/export_onnx.py\n"
                "Then SCP both block_dice_model.onnx and labels.json to the Pi."
            )
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels file missing: {LABELS_PATH}")

        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
        # labels = {"0": "both_down", "1": "d6_1", ...}
        self.classes = [labels[str(i)] for i in range(len(labels))]

        self.session = ort.InferenceSession(
            ONNX_PATH,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        print(f"[BlockDiceClassifier ONNX] Loaded. Classes: {self.classes}")

    def _to_input(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop → preprocessed float32 tensor (1, 3, 128, 128)."""
        normalised = _preprocess_crop(crop_bgr)
        rgb = cv2.cvtColor(normalised, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = x.transpose(2, 0, 1)         # HWC → CHW
        return np.expand_dims(x, 0)      # add batch dim

    def predict(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        x = self._to_input(crop_bgr)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        # Softmax
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def predict_display(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        label, conf = self.predict(crop_bgr)
        return DISPLAY_NAMES.get(label, label), conf
