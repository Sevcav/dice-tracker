"""
detect_dice.py
--------------
Detects individual dice in a camera frame using adaptive edge/threshold
detection — colour-independent so it works with ANY dice colour.

Strategy:
  1. Find the red NAF tray ROI (the tray IS always red — that's fine)
  2. Inside the tray, use adaptive thresholding + Canny edges to find
     square-ish blobs of die-face size, regardless of colour
  3. Split merged blobs (touching dice) using distance-transform peaks
  4. Return one DiceDetection per die, with a crop for the classifier

Works with any dice colour — black, cream, purple, blue, etc.

Usage:
    from detection.detect_dice import detect_dice, find_tray_roi
    roi  = find_tray_roi(frame)
    dice = detect_dice(frame, roi=roi)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class DiceDetection:
    bbox:         tuple[int, int, int, int]  # (x, y, w, h) in original frame
    crop:         np.ndarray                  # BGR crop of the die
    colour_hint:  str                         # kept for API compat — always "unknown"
    shape_hint:   Literal["cube", "pyramid", "unknown"]


# ── Tunable parameters ─────────────────────────────────────────────────────────

# Die area in pixels (at current camera height)
MIN_AREA = 300
MAX_AREA = 4_000

# Minimum pixel side-length — real dice are at least 28px on a side
MIN_SIDE = 28

# Aspect ratio — dice faces are roughly square
MIN_ASPECT = 0.50
MAX_ASPECT = 2.00

# Squareness — max(w,h)/min(w,h). Corner brackets are very non-square.
MAX_SQUARENESS_RATIO = 1.65

# Edge exclusion band inside the ROI (pixels from each edge)
# Rejects tray rim, stitching, and corner brackets
EDGE_EXCLUSION_PX = 35

# Stability: frames dice must be still before "settled" is declared
SETTLE_FRAMES = 6
SETTLE_MOVE   = 12   # max centroid movement (px) — covers lighting-induced drift
SETTLE_COUNT_TOLERANCE = 2  # allow this many frames with different die count

# Watershed split: blobs larger than this trigger split attempt
SINGLE_DIE_MAX_AREA = 1_200   # lower threshold — split sooner
SPLIT_MIN_AREA      = 120

# Tray wall fraction: shrink each side of the detected tray bounding box
# by this fraction to land on the inner playing surface.
# 0.10 = remove 10% from each edge (20% total per axis)
TRAY_WALL_FRACTION = 0.07


# ── Red tray ROI detection ─────────────────────────────────────────────────────

def find_tray_roi(frame: np.ndarray) -> tuple[int,int,int,int] | None:
    """
    Detect the red NAF dice tray and return the INNER playing surface.

    Strategy:
      1. Find the largest solid red region (the tray walls)
      2. Take its bounding rectangle — that's the outer tray boundary
      3. Shrink by TRAY_WALL_FRACTION on each side — removes the walls
         so only the flat inner surface (where dice land) is searched

    This is robust to notebooks/risers under the camera because we pick
    the LARGEST solid red blob, and the tray is always the dominant
    red object in the scene.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps in HSV — need two ranges
    # Saturation > 150 excludes the notebook riser (S=128-139) while
    # keeping the tray's deep saturated red (S=180+)
    mask1 = cv2.inRange(hsv, np.array([0,   150, 60]),  np.array([10,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 150, 60]),  np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Close holes (dice sitting in tray / dark decoration)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    # Pick the largest red blob that is at least 3% of the frame
    # and reasonably square (tray aspect ratio never exceeds 2:1)
    best_cnt  = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.03:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) / max(min(w, h), 1) > 2.5:
            continue   # too elongated to be the tray
        if area > best_area:
            best_area = area
            best_cnt  = cnt

    if best_cnt is None:
        return None

    # ── Find true tray edges by scanning the red mask ─────────────────────────
    # Rather than trusting the bounding box (which includes the notebook),
    # scan the mask row/column by row to find where the MAJORITY of rows
    # have red pixels — this gives the true tray boundary even when the
    # notebook is touching it.

    # Start with bounding rect of the best contour
    x, y, w, h = cv2.boundingRect(best_cnt)

    # Crop the red mask to the bounding rect area
    mask_crop = red_mask[y:y+h, x:x+w]

    # For each column, count how many rows in that column are red
    col_counts = np.sum(mask_crop > 0, axis=0)  # shape: (w,)
    row_counts = np.sum(mask_crop > 0, axis=1)  # shape: (h,)

    # The tray columns have consistently high red pixel counts
    # (the walls are solid red). The notebook column has much lower counts.
    # Find the rightmost column where count > 20% of column height
    threshold_col = h * 0.20   # column must have red in 20% of its height
    threshold_row = w * 0.35   # row must have red in 35% of its width (stricter — rejects thin bottom blobs)

    valid_cols = np.where(col_counts > threshold_col)[0]
    valid_rows = np.where(row_counts > threshold_row)[0]

    if len(valid_cols) == 0 or len(valid_rows) == 0:
        pass  # fall through to use original bounding rect
    else:
        # True tray left/right/top/bottom in mask_crop coordinates
        true_x1 = int(valid_cols[0])
        true_x2 = int(valid_cols[-1])
        true_y1 = int(valid_rows[0])
        true_y2 = int(valid_rows[-1])
        # Translate back to full frame coordinates
        x = x + true_x1
        y = y + true_y1
        w = true_x2 - true_x1
        h = true_y2 - true_y1

    # Shrink by TRAY_WALL_FRACTION on every side to get inner playing surface
    # where dice actually land (removes tray walls, rim, corner brackets)
    inset_x = int(w * TRAY_WALL_FRACTION)
    inset_y = int(h * TRAY_WALL_FRACTION)

    x += inset_x;  y += inset_y
    w -= inset_x * 2;  h -= inset_y * 2

    # Small additional trim for persistent bleed on right/bottom
    w -= int(w * 0.01)   # trim 1% off right
    h -= int(h * 0.02)   # trim 2% off bottom

    if w <= 0 or h <= 0:
        return None

    # Clamp to frame bounds
    x = max(0, x);  y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    return (x, y, w, h)


def crop_to_roi(frame, roi):
    x, y, w, h = roi
    return frame[y:y+h, x:x+w].copy(), (x, y)


# ── Colour-independent die mask ────────────────────────────────────────────────

def _mask_dice(frame: np.ndarray) -> np.ndarray:
    """
    Produce a binary mask of die-face regions — works for ANY die colour
    including dark/black dice on a red tray.

    Three detection paths combined:
      1. Adaptive threshold  — finds locally bright dice (cream, white, etc.)
      2. Canny edges         — finds all die outlines regardless of colour
      3. Dark-object mask    — specifically finds dark/black dice that are
                               darker than the red tray background (low V in HSV)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # ── Red tray mask (exclude from all paths) ────────────────────────────────
    red1 = cv2.inRange(hsv, np.array([0,   120, 60]),  np.array([10,  255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 120, 60]),  np.array([180, 255, 255]))
    red  = cv2.bitwise_or(red1, red2)
    # Dilate red mask slightly so die edges touching the tray don't bleed in
    k_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red   = cv2.dilate(red, k_red, iterations=1)
    not_red = cv2.bitwise_not(red)

    # ── Path 1: adaptive threshold (finds light/cream dice) ──────────────────
    adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=4
    )
    adapt = cv2.bitwise_and(adapt, not_red)

    # ── Path 2: Canny edges → dilate (finds all die outlines) ────────────────
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    k_edge  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_d = cv2.dilate(edges, k_edge, iterations=2)
    edges_d = cv2.bitwise_and(edges_d, not_red)

    # ── Path 3: dark-object mask (finds black/dark dice on red tray) ─────────
    # Black dice have V < 90 — distinct dark islands on the brighter red tray.
    # The die face has light-coloured symbols (pips, logo) that break it into
    # a ring/donut shape. We use a large closing kernel then flood-fill the
    # interior holes to get a solid blob for the whole die face.
    dark_mask = cv2.threshold(v_ch, 90, 255, cv2.THRESH_BINARY_INV)[1]
    # Remove strongly red pixels
    dark_mask = cv2.bitwise_and(dark_mask, not_red)
    # Large close to bridge across the light symbols on the die face
    k_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_dark)
    # Flood-fill interior holes — turns donuts into solid discs
    dark_filled = dark_mask.copy()
    fh_d, fw_d = dark_filled.shape
    flood_mask = np.zeros((fh_d + 2, fw_d + 2), np.uint8)
    cv2.floodFill(dark_filled, flood_mask, (0, 0), 255)
    dark_filled = cv2.bitwise_not(dark_filled)
    dark_mask = cv2.bitwise_or(dark_mask, dark_filled)
    # Remove tiny noise blobs
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k_open)

    # ── Combine all three paths ───────────────────────────────────────────────
    combined = cv2.bitwise_or(adapt, edges_d)
    combined = cv2.bitwise_or(combined, dark_mask)

    # ── Morphological closing to fill die face interiors ─────────────────────
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_open2 = cv2.getStructuringElement(cv2.MORPH_RECT,    (2, 2))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open2)

    return combined


# ── Shape hint ─────────────────────────────────────────────────────────────────

def _shape_hint(contour) -> Literal["cube", "pyramid", "unknown"]:
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / h
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    verts  = len(approx)
    if 0.75 < aspect < 1.33 and 4 <= verts <= 6:
        return "cube"
    if verts >= 5 or aspect < 0.75:
        return "pyramid"
    return "unknown"


# ── Watershed blob splitter ────────────────────────────────────────────────────

def _split_blob(mask_roi: np.ndarray) -> list[tuple[int,int,int,int]]:
    """
    Split a merged blob into individual die bounding boxes using
    distance-transform local maxima (one peak per die centre).
    Returns list of (x, y, w, h) relative to mask_roi origin.
    """
    h, w = mask_roi.shape[:2]

    dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)

    # Estimate single-die size from blob area — use this to set kernel
    blob_area = int(cv2.countNonZero(mask_roi))
    # Approximate side length of one die in this blob
    est_dice  = max(1, round(blob_area / 1_200))   # guess how many dice
    die_px    = max(10, int(np.sqrt(blob_area / max(est_dice, 1))))

    # Smooth just enough to merge noise within a single die face,
    # but NOT so much that adjacent dice merge — kernel ~ 1/3 die width
    smooth_k = max(3, (die_px // 3) | 1)   # must be odd
    dist_smooth = cv2.GaussianBlur(dist, (smooth_k, smooth_k), 0)
    cv2.normalize(dist_smooth, dist_smooth, 0, 1.0, cv2.NORM_MINMAX)

    # Local maxima kernel ~ 1/2 die width — finds one peak per die
    kernel_size = max(5, (die_px // 2) | 1)
    dil = cv2.dilate(dist_smooth,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size)))
    peak_mask = np.uint8((dist_smooth >= dil - 0.001) * 255)
    _, thresh_mask = cv2.threshold(dist_smooth, 0.25, 1.0, cv2.THRESH_BINARY)
    peak_mask = peak_mask & np.uint8(thresh_mask * 255)

    n_labels, _, _, centroids = cv2.connectedComponentsWithStats(peak_mask)
    centres = [(int(centroids[i][0]), int(centroids[i][1]))
               for i in range(1, n_labels)]

    if not centres:
        return [(0, 0, w, h)]

    blob_area   = int(cv2.countNonZero(mask_roi))
    die_half    = max(10, int(np.sqrt(blob_area / len(centres)) / 2) + 4)

    bboxes = []
    for (cx_p, cy_p) in centres:
        x0 = max(0, cx_p - die_half);  y0 = max(0, cy_p - die_half)
        x1 = min(w, cx_p + die_half);  y1 = min(h, cy_p + die_half)
        bboxes.append((x0, y0, x1 - x0, y1 - y0))
    return bboxes


# ── Main detection ─────────────────────────────────────────────────────────────

def _contours_to_detections(
    frame: np.ndarray,
    mask:  np.ndarray,
) -> list[DiceDetection]:
    fh, fw = frame.shape[:2]
    detections = []

    ex = EDGE_EXCLUSION_PX
    edge_x_min, edge_x_max = ex, fw - ex
    edge_y_min, edge_y_max = ex, fh - ex

    def _in_edge_band(cx, cy):
        return (cx < edge_x_min or cx > edge_x_max or
                cy < edge_y_min or cy > edge_y_max)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA * 8:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # ── Large blob: try to split into individual dice ─────────────────────
        if area > SINGLE_DIE_MAX_AREA:
            blob_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted   = cnt.copy()
            shifted[:, :, 0] -= bx
            shifted[:, :, 1] -= by
            cv2.drawContours(blob_mask, [shifted], -1, 255, cv2.FILLED)

            for (sx, sy, sw, sh) in _split_blob(blob_mask):
                fx, fy = bx + sx, by + sy
                if sw < MIN_SIDE or sh < MIN_SIDE:
                    continue
                cx_i = int(fx + sw / 2);  cy_i = int(fy + sh / 2)
                if _in_edge_band(cx_i, cy_i):
                    continue
                die_half = min(sw, sh) // 2 + 6
                x1 = max(0, cx_i - die_half);  y1 = max(0, cy_i - die_half)
                x2 = min(fw, cx_i + die_half); y2 = min(fh, cy_i + die_half)
                crop = frame[y1:y2, x1:x2].copy()
                side = min(sw, sh)
                detections.append(DiceDetection(
                    bbox=(cx_i - side//2, cy_i - side//2, side, side),
                    crop=crop,
                    colour_hint="unknown",
                    shape_hint="cube",
                ))
            continue

        # ── Normal single-die path ────────────────────────────────────────────
        x, y, w, h = bx, by, bw, bh

        if w < MIN_SIDE or h < MIN_SIDE:
            continue

        aspect = w / h
        if not (MIN_ASPECT < aspect < MAX_ASPECT):
            continue

        squareness = max(w, h) / min(w, h)
        if squareness > MAX_SQUARENESS_RATIO:
            continue

        cx = x + w / 2;  cy = y + h / 2
        if _in_edge_band(cx, cy):
            continue

        pad = 6
        x1 = max(0, x - pad);   y1 = max(0, y - pad)
        x2 = min(fw, x+w+pad);  y2 = min(fh, y+h+pad)
        crop = frame[y1:y2, x1:x2].copy()

        detections.append(DiceDetection(
            bbox=(x, y, w, h),
            crop=crop,
            colour_hint="unknown",
            shape_hint=_shape_hint(cnt),
        ))

    return detections


def detect_dice(frame: np.ndarray,
                roi: tuple[int,int,int,int] | None = None) -> list[DiceDetection]:
    """
    Detect all dice in a frame, colour-independently.
    Bounding boxes are always in full-frame coordinates.
    If no ROI is provided, returns empty — we ONLY search inside the tray.
    """
    if roi is None:
        return []   # Don't search whole frame — too many false positives
    search_frame, (ox, oy) = crop_to_roi(frame, roi)

    mask = _mask_dice(search_frame)
    detections = _contours_to_detections(search_frame, mask)

    # Translate bounding boxes back to full-frame coords
    translated = []
    for d in detections:
        x, y, w, h = d.bbox
        fx, fy = x + ox, y + oy
        pad = 6
        x1 = max(0, fx-pad);   y1 = max(0, fy-pad)
        x2 = min(frame.shape[1], fx+w+pad)
        y2 = min(frame.shape[0], fy+h+pad)
        translated.append(DiceDetection(
            bbox=(fx, fy, w, h),
            crop=frame[y1:y2, x1:x2].copy(),
            colour_hint="unknown",
            shape_hint=d.shape_hint,
        ))
    detections = translated

    detections = _suppress_overlaps(detections)
    detections.sort(key=lambda d: (d.bbox[1] // 60, d.bbox[0]))
    return detections


def _suppress_overlaps(detections: list[DiceDetection],
                        iou_thresh=0.15) -> list[DiceDetection]:
    if len(detections) <= 1:
        return detections

    def iou(a, b):
        ax, ay, aw, ah = a.bbox;  bx, by, bw, bh = b.bbox
        ix = max(ax, bx);  iy = max(ay, by)
        ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
        inter = max(0, ix2-ix) * max(0, iy2-iy)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0

    def centres_too_close(a, b):
        ax, ay, aw, ah = a.bbox;  bx, by, bw, bh = b.bbox
        dist = ((ax+aw/2 - bx-bw/2)**2 + (ay+ah/2 - by-bh/2)**2) ** 0.5
        return dist < min(aw, ah, bw, bh) * 0.75

    keep = []
    used = [False] * len(detections)
    sorted_d = sorted(enumerate(detections),
                      key=lambda x: x[1].bbox[2]*x[1].bbox[3], reverse=True)
    for i, d in sorted_d:
        if used[i]:
            continue
        keep.append(d)
        for j, other in sorted_d:
            if not used[j] and j != i:
                if iou(d, other) > iou_thresh or centres_too_close(d, other):
                    used[j] = True
    return keep


# ── Stability tracker ──────────────────────────────────────────────────────────

class DiceStabilityTracker:
    """
    Tracks whether dice have stopped moving.

    Key improvements over naive index-based tracking:
    - Matches centroids by proximity across frames (not by sort order)
      so reordering detections doesn't look like movement
    - Allows SETTLE_COUNT_TOLERANCE noisy frames before resetting
    - Uses a running "anchor" set of positions — once a die is seen
      in roughly the same spot for several frames, it's considered still
    """

    def __init__(self):
        self._history: list[list[tuple[int, int]]] = []

    @staticmethod
    def _match_centroids(
        prev: list[tuple[int, int]],
        curr: list[tuple[int, int]],
        max_dist: float,
    ) -> list[tuple[int, int]] | None:
        """
        Match curr centroids to prev by nearest neighbour.
        Returns matched curr positions in prev order, or None if
        any prev centroid has no match within max_dist.
        """
        matched = []
        used = set()
        for px, py in prev:
            best_d = float('inf')
            best_j = -1
            for j, (cx, cy) in enumerate(curr):
                if j in used:
                    continue
                d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j == -1 or best_d > max_dist:
                return None   # a prev die has no matching curr die
            matched.append(curr[best_j])
            used.add(best_j)
        return matched

    def update(self, detections: list[DiceDetection]) -> bool:
        centroids = [(d.bbox[0] + d.bbox[2] // 2, d.bbox[1] + d.bbox[3] // 2)
                     for d in detections]
        self._history.append(centroids)
        if len(self._history) > SETTLE_FRAMES + 4:
            self._history.pop(0)
        if len(self._history) < SETTLE_FRAMES:
            return False

        recent = self._history[-SETTLE_FRAMES:]
        counts = [len(h) for h in recent]

        # Most common die count across recent frames
        modal_count = max(set(counts), key=counts.count)
        if modal_count == 0:
            return False

        # Too many frames with wrong count → not settled
        bad = sum(1 for c in counts if c != modal_count)
        if bad > SETTLE_COUNT_TOLERANCE:
            return False

        # Use only frames with the modal count
        good_frames = [h for h in recent if len(h) == modal_count]
        if len(good_frames) < SETTLE_FRAMES - SETTLE_COUNT_TOLERANCE:
            return False

        # Check that every die stays within SETTLE_MOVE px across all good frames
        # Use the first good frame as anchor, match subsequent frames by proximity
        anchor = good_frames[0]
        max_match_dist = SETTLE_MOVE * 4   # generous match radius

        for frame in good_frames[1:]:
            matched = self._match_centroids(anchor, frame, max_match_dist)
            if matched is None:
                return False   # a die disappeared or jumped too far

            # Compute per-die deltas from anchor
            deltas_x = [mx - ax for (ax, ay), (mx, my) in zip(anchor, matched)]
            deltas_y = [my - ay for (ax, ay), (mx, my) in zip(anchor, matched)]

            # If ALL dice moved in the same direction by a similar amount,
            # it's camera shake — subtract the median shift and ignore it
            if len(deltas_x) > 1:
                med_dx = sorted(deltas_x)[len(deltas_x) // 2]
                med_dy = sorted(deltas_y)[len(deltas_y) // 2]
                # Only treat as camera shake if median shift is significant
                # and all dice agree (low variance)
                if abs(med_dx) > 2 or abs(med_dy) > 2:
                    spread_x = max(deltas_x) - min(deltas_x)
                    spread_y = max(deltas_y) - min(deltas_y)
                    if spread_x <= 4 and spread_y <= 4:
                        # All dice moved together — camera shake, ignore
                        continue

            # Check individual die movement
            for (ax, ay), (mx, my) in zip(anchor, matched):
                if abs(ax - mx) > SETTLE_MOVE or abs(ay - my) > SETTLE_MOVE:
                    return False

        return True

    def reset(self):
        self._history.clear()


def draw_detections(frame: np.ndarray, detections: list[DiceDetection],
                    labels: list[str] | None = None) -> np.ndarray:
    out = frame.copy()
    for i, d in enumerate(detections):
        x, y, w, h = d.bbox
        colour = (30, 220, 30)
        cv2.rectangle(out, (x, y), (x+w, y+h), colour, 2)
        label = labels[i] if labels and i < len(labels) else f"Die {i+1}"
        cv2.putText(out, label, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out
