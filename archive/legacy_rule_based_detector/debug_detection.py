"""
debug_detection.py
------------------
Captures one frame, runs detection, saves annotated full frame
and individual crops side by side so you can see exactly what
the detector is finding and cropping.

Press any key to capture a fresh frame, Q to quit.
"""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.detect_dice import find_tray_roi, detect_dice, _mask_dice, crop_to_roi
from detection.classify_die_type import classify_die_type, _symbol_coverage

cap = cv2.VideoCapture(0)
cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Debug", 1400, 700)

tray_roi = None
frame_count = 0

print("Running — press S to save debug image, Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_count += 1

    # Update ROI every 30 frames
    if frame_count % 30 == 1:
        new_roi = find_tray_roi(frame)
        if new_roi:
            tray_roi = new_roi

    dets = detect_dice(frame, roi=tray_roi)

    # ── Left panel: full frame with ROI and bboxes ────────────────────────────
    vis = frame.copy()
    if tray_roi:
        rx, ry, rw, rh = tray_roi
        cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), (200, 80, 0), 2)

    for i, d in enumerate(dets):
        x, y, w, h = d.bbox
        coverage = _symbol_coverage(d.crop)
        die_type = classify_die_type(d, frame.shape)
        label = f"#{i+1} {die_type} cov={coverage:.2f}"
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis, label, (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(vis, str(i+1), (x+4, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.putText(vis, f"Dice: {len(dets)}  ROI: {tray_roi}",
                (8, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Right panel: crop grid ────────────────────────────────────────────────
    crop_size = 120
    cols = max(5, len(dets))
    panel_w = crop_size * cols
    panel_h = crop_size + 30
    panel = np.zeros((panel_h, max(panel_w, frame.shape[1]), 3), dtype=np.uint8)

    for i, d in enumerate(dets):
        crop = d.crop
        ch, cw = crop.shape[:2]
        # Scale to fit
        scale = min(crop_size / max(ch, 1), crop_size / max(cw, 1))
        nw, nh = int(cw * scale), int(ch * scale)
        if nw > 0 and nh > 0:
            resized = cv2.resize(crop, (nw, nh))
            px = i * crop_size + (crop_size - nw) // 2
            py = (crop_size - nh) // 2
            panel[py:py+nh, px:px+nw] = resized
        coverage = _symbol_coverage(d.crop)
        die_type = classify_die_type(d, frame.shape)
        cv2.putText(panel, f"#{i+1} {die_type[:5]}",
                    (i*crop_size+2, crop_size+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(panel, f"cov={coverage:.2f}",
                    (i*crop_size+2, crop_size+26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 100), 1)

    # ── Also show the dice mask inside ROI ───────────────────────────────────
    mask_vis = frame.copy()
    if tray_roi:
        roi_crop, (ox, oy) = crop_to_roi(frame, tray_roi)
        mask = _mask_dice(roi_crop)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        rh2 = tray_roi[3]; rw2 = tray_roi[2]
        mask_vis[oy:oy+rh2, ox:ox+rw2] = cv2.addWeighted(
            roi_crop, 0.4, mask_bgr, 0.6, 0)

    # Stack: [annotated | mask] on top, [crops] on bottom
    fw = frame.shape[1]
    top_left  = cv2.resize(vis,      (fw, frame.shape[0]))
    top_right = cv2.resize(mask_vis, (fw, frame.shape[0]))
    top = np.hstack([top_left, top_right])

    bottom_full = cv2.resize(panel, (top.shape[1], 160))
    combined = np.vstack([top, bottom_full])

    cv2.imshow("Debug", combined)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if key == ord('s'):
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"debug_{ts}.jpg")
        cv2.imwrite(path, combined)
        print(f"Saved: debug_{ts}.jpg")
        # Also save individual crops
        for i, d in enumerate(dets):
            cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"debug_{ts}_crop{i+1}.jpg")
            cv2.imwrite(cpath, d.crop)
            coverage = _symbol_coverage(d.crop)
            die_type = classify_die_type(d, frame.shape)
            print(f"  Crop {i+1}: {d.bbox}  coverage={coverage:.3f}  type={die_type}")

cap.release()
cv2.destroyAllWindows()
