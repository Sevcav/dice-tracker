"""
read_numbered.py
----------------
Reads the numeric face value from a d8 or d16 die crop using EasyOCR.

Handles:
  - d8  : values 1–8
  - d16 : values 1–16

The cream/tan die body with black numerals is pre-processed to maximise
OCR accuracy (high contrast, de-noised, upscaled).

Usage:
    from reading.read_numbered import NumberedDiceReader
    reader = NumberedDiceReader()
    value, confidence = reader.read(crop_bgr, die_type="d8")
    # value = int (1-8 or 1-16) or None if unreadable
"""

from __future__ import annotations
import cv2
import numpy as np
import re

# EasyOCR is imported lazily (first call) to keep startup fast
_ocr_reader = None


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[OCR] Loading EasyOCR model (first time only, may take ~10s)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("[OCR] Ready.")
    return _ocr_reader


# ── Image pre-processing ───────────────────────────────────────────────────────

def _preprocess(crop: np.ndarray, upscale: int = 3) -> np.ndarray:
    """
    Prepare die face for OCR:
    1. Upscale (OCR works better on larger images)
    2. Convert to grayscale
    3. Sharpen
    4. Adaptive threshold → clean black-on-white text
    """
    h, w = crop.shape[:2]
    big  = cv2.resize(crop, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # Sharpen
    kernel  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp   = cv2.filter2D(gray, -1, kernel)

    # Adaptive threshold
    thresh  = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    # Slight morphological cleanup
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)

    # Convert back to BGR so EasyOCR receives correct format
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


# ── OCR result parsing ─────────────────────────────────────────────────────────

def _parse_result(results: list, die_type: str) -> tuple[int | None, float]:
    """
    Extract the most likely integer from EasyOCR result list.
    Returns (value, confidence) or (None, 0.0).
    """
    max_range = 8 if die_type == "d8" else 16

    candidates = []
    for (_, text, conf) in results:
        # Strip non-numeric chars
        text_clean = re.sub(r"[^0-9]", "", text.strip())
        if not text_clean:
            continue
        try:
            val = int(text_clean)
        except ValueError:
            continue
        if 1 <= val <= max_range:
            candidates.append((val, conf))

    if not candidates:
        return None, 0.0

    # Pick highest confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


# ── Public class ───────────────────────────────────────────────────────────────

class NumberedDiceReader:
    """Reads d8 and d16 die faces via OCR."""

    def __init__(self):
        # Trigger model load at construction time so first read is fast
        _get_ocr()

    def read(self, crop_bgr: np.ndarray, die_type: str = "d8") -> tuple[int | None, float]:
        """
        Parameters
        ----------
        crop_bgr : np.ndarray  BGR image of a single die face
        die_type : "d8" | "d16"

        Returns
        -------
        (value, confidence)
            value      : int 1-8 or 1-16, or None if unreadable
            confidence : float 0.0-1.0
        """
        ocr    = _get_ocr()
        prepped = _preprocess(crop_bgr)

        # EasyOCR: allow_list restricts to digits and common look-alikes
        results = ocr.readtext(
            prepped,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
        )

        value, conf = _parse_result(results, die_type)

        # Fallback: try without preprocessing if first attempt failed
        if value is None:
            results2 = ocr.readtext(
                crop_bgr,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
            )
            value, conf = _parse_result(results2, die_type)

        return value, conf


# ── d6 pip counter — colour-independent ───────────────────────────────────────

def count_pips(crop_bgr: np.ndarray) -> int | None:
    """
    Count pip dots on a d6 face — works for ANY die colour.

    Strategy: pips are small, round, uniform blobs that contrast with the
    die body. We don't know which is lighter — pip or body — so we try
    both polarities of adaptive threshold and pick the result that gives
    1-5 clean, similarly-sized circular blobs.

    Returns pip count (1-5) or None if no clean pip pattern found.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Scale-aware pip size: a pip shouldn't be larger than 1/4 of the face
    # and shouldn't be smaller than ~8px² (tiny crops at any camera height)
    max_pip_area = max(30, (h * w) // 4)
    min_pip_area = max(8, (h * w) // 200)

    best_result = None

    # Scale blockSize to crop size — must be odd, at least 5
    block_size = max(5, (min(h, w) // 6) | 1)

    for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            polarity,
            blockSize=block_size, C=4
        )

        # Morphological cleanup — close tiny gaps in pips, remove specks
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea        = True
        params.minArea             = min_pip_area
        params.maxArea             = max_pip_area
        params.filterByCircularity = True
        params.minCircularity      = 0.55   # pips are round; arrows/skulls are not
        params.filterByConvexity   = True
        params.minConvexity        = 0.65
        params.filterByInertia     = True
        params.minInertiaRatio     = 0.35

        detector  = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh)
        n = len(keypoints)

        if not (1 <= n <= 5):
            continue

        # Pips on a single face are all the same size — reject if wildly mixed
        if n > 1:
            sizes = [kp.size for kp in keypoints]
            size_ratio = max(sizes) / max(min(sizes), 1)
            if size_ratio > 3.5:
                continue

        # Prefer the polarity that gives more blobs (more confident pip count)
        if best_result is None or n > best_result:
            best_result = n

    return best_result
