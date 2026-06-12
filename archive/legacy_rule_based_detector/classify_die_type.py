"""
classify_die_type.py
--------------------
Determines WHAT TYPE of die a crop is, before reading its value.

Colour-INDEPENDENT logic — works for any dice colour.

Decision order:
  1. Pyramid shape → d8 or d16
  2. Symbol coverage:
       HIGH (>30%)  → block die  (large symbols fill the face)
       LOW  (<12%)  → d6_bb      (small pips, lots of empty face)
       MID  (12-30%) → use pip geometry to decide
  3. In mid-range: count round pip-like blobs
       Clean 1-5 round blobs of uniform size → d6_bb
       Otherwise → block die

Why coverage-first?
  Block dice symbols (Push arrow, Skull, POW!) cover 30-65% of the face.
  d6 pip faces (1-5 pips) cover 5-20%.
  The BB logo face (6) covers ~25-50% — but read_die.py handles this
  by running the CNN first and catching d6_bb_logo before pip counting.

  Pip geometry alone is unreliable because circular elements appear in
  block die symbols (the skull eye sockets, the push arrow curves).
  Coverage is a much stronger signal.
"""

from __future__ import annotations
import cv2
import numpy as np
from detection.detect_dice import DiceDetection
from typing import Literal

DieType = Literal["d6_bb", "block", "d8", "d16", "unknown"]

# ── Size thresholds ────────────────────────────────────────────────────────────
D16_MIN_AREA_FRACTION = 0.010
D8_MIN_AREA_FRACTION  = 0.003

# ── Coverage bands ─────────────────────────────────────────────────────────────
BLOCK_HIGH_COVERAGE  = 0.22   # definitely block die
D6_LOW_COVERAGE      = 0.10   # definitely d6 pip face
# Between 0.10 and 0.22: use pip geometry


def _symbol_coverage(crop: np.ndarray) -> float:
    """
    Fraction of the die face covered by markings (colour-independent).
    Uses adaptive threshold on the centre 80% of the crop.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cy, cx = h // 10, w // 10
    centre = gray[cy:h-cy, cx:w-cx]
    if centre.size == 0:
        return 0.0

    thresh = cv2.adaptiveThreshold(
        centre, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=6
    )
    return np.count_nonzero(thresh) / thresh.size


def _has_clean_pips(crop: np.ndarray) -> bool:
    """
    Returns True if the crop contains 1-5 small, round, uniformly-sized
    blobs — the geometric signature of d6 pip faces.

    Tries both polarities (dark-on-light and light-on-dark) to be
    colour-independent. Only called when coverage is ambiguous (12-30%).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Pip must be smaller than 1/5 of the face area
    max_pip_area = max(30, (h * w) // 5)
    min_pip_area = max(8,  (h * w) // 300)

    for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            polarity,
            blockSize=11, C=4
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea        = True
        params.minArea             = min_pip_area
        params.maxArea             = max_pip_area
        params.filterByCircularity = True
        params.minCircularity      = 0.70   # pips are very round
        params.filterByConvexity   = True
        params.minConvexity        = 0.80
        params.filterByInertia     = True
        params.minInertiaRatio     = 0.50

        detector  = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh)
        n = len(keypoints)

        if not (1 <= n <= 5):
            continue

        # All pips on one face are the same size — reject wildly mixed blobs
        if n > 1:
            sizes = [kp.size for kp in keypoints]
            if max(sizes) / max(min(sizes), 1) > 3.0:
                continue

        return True  # clean pips found

    return False


def classify_die_type(
    detection: DiceDetection,
    frame_shape: tuple[int, int],
) -> DieType:
    """
    Classify a detected die into its type — colour-independent.
    """
    crop = detection.crop
    _, _, w, h = detection.bbox
    frame_h, frame_w = frame_shape[:2]
    area_frac = (w * h) / (frame_h * frame_w)
    shape     = detection.shape_hint

    # ── 1. Pyramid → d8 or d16 ────────────────────────────────────────────────
    if shape == "pyramid":
        if area_frac >= D16_MIN_AREA_FRACTION:
            return "d16"
        elif area_frac >= D8_MIN_AREA_FRACTION:
            return "d8"
        return "unknown"

    # ── 2. Coverage — primary classifier ─────────────────────────────────────
    coverage = _symbol_coverage(crop)

    if coverage > BLOCK_HIGH_COVERAGE:
        return "block"

    if coverage < D6_LOW_COVERAGE:
        return "d6_bb"

    # ── 3. Mid-range: use pip geometry ────────────────────────────────────────
    if _has_clean_pips(crop):
        return "d6_bb"

    return "block"


def describe(die_type: DieType) -> str:
    return {
        "d6_bb":   "d6 (Blood Bowl)",
        "block":   "Block Die",
        "d8":      "d8",
        "d16":     "d16",
        "unknown": "Unknown",
    }.get(die_type, die_type)
