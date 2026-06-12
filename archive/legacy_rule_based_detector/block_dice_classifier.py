"""
block_dice_classifier.py
------------------------
Inference wrapper.  Load once, call predict() per cropped die image.

Usage:
    from classifier.block_dice_classifier import BlockDiceClassifier
    clf = BlockDiceClassifier()
    label, confidence = clf.predict(crop_bgr)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "classifier", "block_dice_model.pt")
LABELS_PATH= os.path.join(BASE, "classifier", "labels.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Human-readable display names for each class
DISPLAY_NAMES = {
    "both_down":   "Both Down",
    "pow":         "POW!",
    "push":        "Push",
    "player_down": "Player Down",
    "stumble":     "Stumble",
    "d6_bb_logo":  "BB Logo (6)",
}

_inference_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize a die crop before classification to handle variable lighting
    (sunlight, store windows, overhead fluorescent, etc.)

    Steps:
      1. CLAHE on L channel (luminance equalisation — handles bright/dark spots)
      2. Convert back to BGR for the CNN (which expects colour input)

    This is illumination-invariant: the symbol shapes are preserved but
    absolute brightness and colour cast are normalised out.
    """
    # CLAHE on LAB L-channel
    lab   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


class BlockDiceClassifier:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run: python classifier/prepare_training_data.py\n"
                "Then: python classifier/train_block_dice.py"
            )

        checkpoint   = torch.load(MODEL_PATH, map_location=DEVICE)
        self.classes = checkpoint["class_names"]
        num_classes  = checkpoint["num_classes"]

        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(DEVICE)
        self.model.eval()

        print(f"[BlockDiceClassifier] Loaded model. Classes: {self.classes}")

    def predict(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Parameters
        ----------
        crop_bgr : np.ndarray
            BGR image crop of a single die face (any size).

        Returns
        -------
        (label, confidence)
            label      : class name string e.g. "push"
            confidence : float 0.0 – 1.0
        """
        normalised = _preprocess_crop(crop_bgr)
        rgb   = cv2.cvtColor(normalised, cv2.COLOR_BGR2RGB)
        tensor = _inference_tf(rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = probs.argmax().item()

        label      = self.classes[idx]
        confidence = probs[idx].item()
        return label, confidence

    def predict_display(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Same as predict() but returns the human-readable display name."""
        label, conf = self.predict(crop_bgr)
        return DISPLAY_NAMES.get(label, label), conf
