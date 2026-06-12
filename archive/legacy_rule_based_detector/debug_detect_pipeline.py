"""
debug_detect_pipeline.py
------------------------
Run the existing detection pipeline against detect_input.jpg and dump every
intermediate stage to disk so we can see exactly where dice get lost.

Outputs (all written next to detect_input.jpg):
    debug_00_input.jpg          — original frame with ROI rectangle drawn
    debug_01_roi_crop.jpg       — pixels inside the calibrated ROI only
    debug_02_path1_adapt.jpg    — adaptive-threshold mask (light dice)
    debug_03_path2_edges.jpg    — Canny edges dilated (all dice outlines)
    debug_04_path3_dark.jpg     — dark-object mask (black dice)
    debug_05_combined.jpg       — OR of all three paths post-morphology
    debug_06_contours_all.jpg   — every contour found, no filters
    debug_07_contours_kept.jpg  — only contours that survive size/aspect/
                                   squareness/edge-band filtering
    debug_08_final.jpg          — what detect_dice() actually returns

Print summary explains how many contours each filter rejects and why.
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.detect_dice import (
    find_tray_roi,
    detect_dice,
    _mask_dice,
    crop_to_roi,
    MIN_AREA, MAX_AREA, MIN_SIDE, MIN_ASPECT, MAX_ASPECT,
    MAX_SQUARENESS_RATIO, EDGE_EXCLUSION_PX, SINGLE_DIE_MAX_AREA,
)

INPUT  = os.path.join(os.path.dirname(__file__), "detect_input.jpg")
OUTDIR = os.path.dirname(__file__)


def out(name): return os.path.join(OUTDIR, name)


def main():
    frame = cv2.imread(INPUT)
    if frame is None:
        print(f"Could not read {INPUT}")
        return
    fh, fw = frame.shape[:2]
    print(f"Input frame: {fw}x{fh}")

    # ── ROI ──────────────────────────────────────────────────────────────────
    roi = find_tray_roi(frame)
    if roi is None:
        print("find_tray_roi returned None — calibration missing or stale")
        return
    rx, ry, rw, rh = roi
    print(f"ROI: x={rx}, y={ry}, w={rw}, h={rh}")

    annotated = frame.copy()
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 3)
    cv2.imwrite(out("debug_00_input.jpg"), annotated)

    # ── Crop to ROI (this is what _mask_dice actually sees) ──────────────────
    search_frame, (ox, oy) = crop_to_roi(frame, roi)
    cv2.imwrite(out("debug_01_roi_crop.jpg"), search_frame)

    # ── Re-derive each mask path manually so we can save them individually ──
    gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    red1 = cv2.inRange(hsv, np.array([0,   120, 60]),  np.array([10,  255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 120, 60]),  np.array([180, 255, 255]))
    red  = cv2.bitwise_or(red1, red2)
    k_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red   = cv2.dilate(red, k_red, iterations=1)
    not_red = cv2.bitwise_not(red)

    # Path 1
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=4,
    )
    adapt_masked = cv2.bitwise_and(adapt, not_red)
    cv2.imwrite(out("debug_02_path1_adapt.jpg"), adapt_masked)

    # Path 2
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    k_edge  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_d = cv2.dilate(edges, k_edge, iterations=2)
    edges_d = cv2.bitwise_and(edges_d, not_red)
    cv2.imwrite(out("debug_03_path2_edges.jpg"), edges_d)

    # Path 3
    dark_mask = cv2.threshold(v_ch, 90, 255, cv2.THRESH_BINARY_INV)[1]
    dark_mask = cv2.bitwise_and(dark_mask, not_red)
    k_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_dark)
    dark_filled = dark_mask.copy()
    fh_d, fw_d = dark_filled.shape
    flood_mask = np.zeros((fh_d + 2, fw_d + 2), np.uint8)
    cv2.floodFill(dark_filled, flood_mask, (0, 0), 255)
    dark_filled = cv2.bitwise_not(dark_filled)
    dark_mask = cv2.bitwise_or(dark_mask, dark_filled)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k_open)
    cv2.imwrite(out("debug_04_path3_dark.jpg"), dark_mask)

    # ── Combined (this is what _mask_dice returns) ───────────────────────────
    combined = _mask_dice(search_frame)
    cv2.imwrite(out("debug_05_combined.jpg"), combined)

    # ── All contours, no filters ─────────────────────────────────────────────
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    all_vis = search_frame.copy()
    cv2.drawContours(all_vis, contours, -1, (0, 200, 255), 2)
    cv2.putText(all_vis, f"Contours found: {len(contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imwrite(out("debug_06_contours_all.jpg"), all_vis)

    # ── Walk filters by hand to count rejections ─────────────────────────────
    print(f"\nFilter parameters:")
    print(f"  MIN_AREA            = {MIN_AREA}")
    print(f"  MAX_AREA            = {MAX_AREA}  (with 8x multiplier = {MAX_AREA*8})")
    print(f"  SINGLE_DIE_MAX_AREA = {SINGLE_DIE_MAX_AREA}")
    print(f"  MIN_SIDE            = {MIN_SIDE}")
    print(f"  MIN_ASPECT/MAX      = {MIN_ASPECT} / {MAX_ASPECT}")
    print(f"  MAX_SQUARENESS      = {MAX_SQUARENESS_RATIO}")
    print(f"  EDGE_EXCLUSION_PX   = {EDGE_EXCLUSION_PX}  (frame {fw}x{fh})")

    rejects = {
        "below MIN_AREA":   0,
        "above MAX_AREA*8": 0,
        "MIN_SIDE":         0,
        "aspect":           0,
        "squareness":       0,
        "edge-band":        0,
        "split candidate":  0,   # would be sent to watershed
        "kept (single)":    0,
    }
    kept_vis = search_frame.copy()
    sx_off, sy_off = 0, 0   # working in ROI coords now

    ex = EDGE_EXCLUSION_PX
    edge_x_min, edge_x_max = ex, fw - ex
    edge_y_min, edge_y_max = ex, fh - ex

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            rejects["below MIN_AREA"] += 1
            continue
        if area > MAX_AREA * 8:
            rejects["above MAX_AREA*8"] += 1
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if area > SINGLE_DIE_MAX_AREA:
            rejects["split candidate"] += 1
            # Draw in orange for visibility — would be sent to watershed
            fx, fy = x + ox, y + oy
            cv2.rectangle(kept_vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(kept_vis, f"SPLIT a={int(area)}", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            continue

        if w < MIN_SIDE or h < MIN_SIDE:
            rejects["MIN_SIDE"] += 1
            continue
        aspect = w / h
        if not (MIN_ASPECT < aspect < MAX_ASPECT):
            rejects["aspect"] += 1
            continue
        squareness = max(w, h) / min(w, h)
        if squareness > MAX_SQUARENESS_RATIO:
            rejects["squareness"] += 1
            continue

        # edge-band check uses FULL-frame coords
        cx_full = (x + w / 2) + ox
        cy_full = (y + h / 2) + oy
        if (cx_full < edge_x_min or cx_full > edge_x_max or
                cy_full < edge_y_min or cy_full > edge_y_max):
            rejects["edge-band"] += 1
            continue

        rejects["kept (single)"] += 1
        cv2.rectangle(kept_vis, (x, y), (x + w, y + h), (0, 220, 30), 2)
        cv2.putText(kept_vis, f"a={int(area)} {w}x{h}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 30), 1)

    cv2.imwrite(out("debug_07_contours_kept.jpg"), kept_vis)

    print(f"\nFilter results:")
    for reason, n in rejects.items():
        print(f"  {reason:20s} : {n}")

    # ── Final: what the actual detect_dice() returns ─────────────────────────
    dets = detect_dice(frame, roi=roi)
    final = frame.copy()
    cv2.rectangle(final, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)
    for i, d in enumerate(dets):
        x, y, w, h = d.bbox
        cv2.rectangle(final, (x, y), (x + w, y + h), (0, 220, 30), 3)
        cv2.putText(final, f"Die {i+1}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 30), 2)
    cv2.imwrite(out("debug_08_final.jpg"), final)
    print(f"\ndetect_dice() returned {len(dets)} dice")


if __name__ == "__main__":
    main()
