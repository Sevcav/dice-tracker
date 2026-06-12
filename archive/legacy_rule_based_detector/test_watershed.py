"""
test_watershed.py
-----------------
Quick visual test of the watershed blob-splitting fix.
Run this with dice sitting close together in the tray.

Controls:
    Q / ESC  Quit
    S        Save current frame as watershed_debug.jpg

Shows four panes side by side:
  1. Original camera view with detected bounding boxes
  2. Cream mask (before contour analysis)
  3. Distance-transform heat map of the largest blob
  4. Watershed-split result overlaid on original
"""

import cv2
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection.detect_dice import (
    find_tray_roi, _mask_cream_dice, _mask_black_dice, detect_dice,
    SINGLE_DIE_MAX_AREA, _split_blob
)


def make_panel(img, label, size=(400, 300)):
    panel = cv2.resize(img, size)
    cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
    return panel


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cv2.namedWindow("Watershed Test — Q=quit  S=save", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Watershed Test — Q=quit  S=save", 1300, 650)

    tray_roi = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 30 == 1:
            new_roi = find_tray_roi(frame)
            if new_roi:
                tray_roi = new_roi

        # --- Pane 1: detections on original ---
        pane1 = frame.copy()
        if tray_roi:
            rx, ry, rw, rh = tray_roi
            cv2.rectangle(pane1, (rx, ry), (rx+rw, ry+rh), (0, 140, 255), 2)

        dets = detect_dice(frame, roi=tray_roi)
        for i, d in enumerate(dets):
            x, y, w, h = d.bbox
            cv2.rectangle(pane1, (x, y), (x+w, y+h), (30, 220, 30), 2)
            cv2.putText(pane1, f"Die {i+1} ({w}x{h})", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 220, 30), 1)

        # --- Pane 2: cream mask ---
        if tray_roi:
            rx, ry, rw, rh = tray_roi
            roi_frame = frame[ry:ry+rh, rx:rx+rw]
        else:
            roi_frame = frame

        cream_mask = _mask_cream_dice(roi_frame)
        black_mask = _mask_black_dice(roi_frame)
        # Show cream in white, black dice mask in blue, combined
        combined = cv2.cvtColor(cream_mask, cv2.COLOR_GRAY2BGR)
        combined[black_mask > 0] = (255, 80, 0)   # blue = black dice
        pane2_bgr  = combined

        # --- Pane 3: distance transform of largest blob ---
        contours, _ = cv2.findContours(cream_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pane3 = np.zeros_like(roi_frame)
        largest_blob_area = 0
        largest_cnt = None
        for c in contours:
            a = cv2.contourArea(c)
            if a > largest_blob_area:
                largest_blob_area = a
                largest_cnt = c

        # Print all blob sizes to console every 60 frames
        if frame_count % 60 == 1 and contours:
            sizes = sorted([(int(cv2.contourArea(c)),
                             cv2.boundingRect(c)[2],
                             cv2.boundingRect(c)[3])
                            for c in contours], reverse=True)[:6]
            print(f"Blobs (area, w, h): {sizes}")

        if largest_cnt is not None and largest_blob_area > SINGLE_DIE_MAX_AREA:
            bx, by, bw, bh = cv2.boundingRect(largest_cnt)
            blob_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted = largest_cnt.copy()
            shifted[:, :, 0] -= bx
            shifted[:, :, 1] -= by
            cv2.drawContours(blob_mask, [shifted], -1, 255, cv2.FILLED)

            dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
            dist_norm = dist.copy()
            cv2.normalize(dist_norm, dist_norm, 0, 255.0, cv2.NORM_MINMAX)
            dist_color = cv2.applyColorMap(np.uint8(dist_norm), cv2.COLORMAP_JET)

            pane3 = np.zeros_like(roi_frame)
            pane3[by:by+bh, bx:bx+bw] = dist_color

            # Show what threshold was chosen and how many peaks found
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            best_n, best_t = 0, 0
            for t in [v/100.0 for v in range(55, 20, -5)]:
                _, fg = cv2.threshold(dist, t, 1.0, cv2.THRESH_BINARY)
                n, _ = cv2.connectedComponents(np.uint8(fg*255))
                n -= 1
                if n > best_n:
                    best_n = n; best_t = t
                elif n < best_n - 1:
                    break
            cv2.putText(pane3, f"area={int(largest_blob_area)} peaks={best_n} t={best_t:.2f}",
                        (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw split boxes on pane2
            sub_bboxes = _split_blob(blob_mask)
            cv2.putText(pane3, f"splits→{len(sub_bboxes)}", (3, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            for (sx, sy, sw, sh) in sub_bboxes:
                cv2.rectangle(pane2_bgr,
                              (bx+sx, by+sy), (bx+sx+sw, by+sy+sh),
                              (0, 0, 255), 2)
        elif largest_cnt is not None:
            bx, by, bw, bh = cv2.boundingRect(largest_cnt)
            cv2.putText(pane3, f"area={int(largest_blob_area)} (single die, no split)", (3,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        # --- Pane 4: final boxes on ROI frame ---
        pane4 = roi_frame.copy()
        for i, d in enumerate(dets):
            # Translate full-frame coords back to roi coords if needed
            x, y, w, h = d.bbox
            if tray_roi:
                x -= tray_roi[0]; y -= tray_roi[1]
            cv2.rectangle(pane4, (x, y), (x+w, y+h), (0, 255, 128), 2)
            cv2.putText(pane4, f"{i+1}", (x+2, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

        # Assemble 2x2 grid
        s = (480, 360)
        row1 = np.hstack([make_panel(pane1, f"Detections: {len(dets)}", s),
                          make_panel(pane2_bgr, "Cream mask + split boxes", s)])
        row2 = np.hstack([make_panel(pane3, "Dist-transform (largest blob)", s),
                          make_panel(pane4, "Final splits on ROI", s)])
        grid = np.vstack([row1, row2])

        cv2.imshow("Watershed Test — Q=quit  S=save", grid)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            cv2.imwrite("watershed_debug.jpg", grid)
            print("Saved watershed_debug.jpg")

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
