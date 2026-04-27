"""
read_die.py
-----------
Top-level die reader. Given a crop and die type, returns the result.

All cube dice (d6 and block) go through the SAME CNN — the CNN is
trained on all classes including d6 pip faces (d6_1 through d6_5),
the BB logo (d6_bb_logo), and all block faces. No pip counting needed.

Usage:
    from reading.read_die import DieReader, DieResult
    reader = DieReader()
    result = reader.read(crop_bgr, die_type)
    print(result.display)   # e.g. "4", "Push", "Both Down", "14"
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal

DieType = Literal["d6_bb", "block", "d8", "d16", "unknown"]

# CNN classes that are d6 pip faces
D6_PIP_CLASSES  = {"d6_1", "d6_2", "d6_3", "d6_4", "d6_5"}
D6_LOGO_CLASS   = "d6_bb_logo"
BLOCK_CLASSES   = {"both_down", "pow", "push", "player_down", "stumble"}

DISPLAY = {
    "both_down":   "Both Down",
    "pow":         "POW!",
    "push":        "Push",
    "player_down": "Player Down",
    "stumble":     "Stumble",
    "d6_bb_logo":  "6 (BB)",
    "d6_1":        "1",
    "d6_2":        "2",
    "d6_3":        "3",
    "d6_4":        "4",
    "d6_5":        "5",
}

PIP_VALUE = {"d6_1": 1, "d6_2": 2, "d6_3": 3, "d6_4": 4, "d6_5": 5}


@dataclass
class DieResult:
    die_type:    str
    raw_value:   str | int | None
    display:     str
    confidence:  float
    is_numeric:  bool


class DieReader:
    def __init__(self):
        self._ocr       = None
        self._block_clf = None

    def _ensure_ocr(self):
        if self._ocr is None:
            from reading.read_numbered import NumberedDiceReader
            self._ocr = NumberedDiceReader()

    def _ensure_block_clf(self):
        if self._block_clf is None:
            from classifier.block_dice_classifier import BlockDiceClassifier
            self._block_clf = BlockDiceClassifier()

    def read(self, crop_bgr: np.ndarray, die_type: DieType) -> DieResult:
        """
        Read a single die face.

        d8 / d16  → OCR
        d6_bb     → CNN (handles pips 1-5 AND BB logo face 6)
        block     → CNN (handles all block symbols)
        unknown   → CNN (try anyway)

        The CNN is now trained on ALL cube face types, so both d6_bb
        and block route through it. The initial die_type from
        classify_die_type is used as a hint but the CNN label wins.
        """
        if die_type in ("d8", "d16"):
            return self._read_numbered(crop_bgr, die_type)

        # All cube dice — run CNN
        self._ensure_block_clf()
        label, conf = self._block_clf.predict(crop_bgr)

        # d6 pip face
        if label in D6_PIP_CLASSES:
            pip = PIP_VALUE[label]
            return DieResult(die_type="d6_bb", raw_value=pip,
                             display=str(pip), confidence=conf, is_numeric=True)

        # d6 BB logo face (6)
        if label == D6_LOGO_CLASS:
            return DieResult(die_type="d6_bb", raw_value=6,
                             display="6 (BB)", confidence=conf, is_numeric=True)

        # Block die symbol
        if label in BLOCK_CLASSES:
            return DieResult(die_type="block", raw_value=label,
                             display=DISPLAY.get(label, label),
                             confidence=conf, is_numeric=False)

        # Fallback
        return DieResult(die_type=die_type, raw_value=None,
                         display="?", confidence=conf, is_numeric=False)

    def _read_numbered(self, crop: np.ndarray, die_type: str) -> DieResult:
        self._ensure_ocr()
        value, conf = self._ocr.read(crop, die_type)
        if value is not None:
            return DieResult(die_type=die_type, raw_value=value,
                             display=str(value), confidence=conf, is_numeric=True)
        return DieResult(die_type=die_type, raw_value=None,
                         display="?", confidence=0.0, is_numeric=True)
