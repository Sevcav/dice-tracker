"""
test_tray.py
------------
Quick diagnostic: shows what the tray detection sees in current lighting.
Press S to save, Q to quit.
"""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection.detect_dice import find_tray_roi, _mask_dice

cap = cv2.VideoCapture(0)
cv2.namedWindow("Tray Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tray Test", 1200, 500)

frame_count = 0
tray_roi    = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_count += 1

    # Show red mask — try progressively stricter saturation thresholds
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat_thresh = 150   # notebook S=128-139, tray S=180+ so 150 splits them cleanly
    mask1 = cv2.inRange(hsv, np.array([0,   sat_thresh, 60]),  np.array([10,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, sat_thresh, 60]),  np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    cv2.putText(red_mask, f"S>{sat_thresh}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
    red_bgr  = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)

    # Print HSV of centre-right area (where notebook is) every 60 frames
    if frame_count % 60 == 1:
        h_f, w_f = frame.shape[:2]
        # Sample notebook area (right 20% of frame, middle height)
        sample = hsv[h_f//3 : 2*h_f//3, int(w_f*0.75):]
        print(f"Notebook area HSV mean: H={np.mean(sample[:,:,0]):.0f} "
              f"S={np.mean(sample[:,:,1]):.0f} V={np.mean(sample[:,:,2]):.0f}")

    # Show dice mask inside tray — use smoothed ROI like main.py does
    if frame_count % 60 == 1:
        new_roi = find_tray_roi(frame)
        if new_roi:
            if tray_roi is None:
                tray_roi = new_roi
            else:
                tray_roi = tuple(int(tray_roi[i]*0.9 + new_roi[i]*0.1) for i in range(4))
    roi = tray_roi
    dice_vis = frame.copy()
    if roi:
        rx, ry, rw, rh = roi
        cv2.rectangle(dice_vis, (rx,ry),(rx+rw,ry+rh),(255,100,0),2)
        cv2.putText(dice_vis,"TRAY OK",(rx,ry-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,100,0),2)
        roi_crop = frame[ry:ry+rh, rx:rx+rw]
        dice_mask = _mask_dice(roi_crop)
        dice_bgr  = cv2.cvtColor(dice_mask, cv2.COLOR_GRAY2BGR)
        # paste dice mask back into position
        dice_vis[ry:ry+rh, rx:rx+rw] = cv2.addWeighted(
            roi_crop, 0.5, dice_bgr, 0.5, 0)
    else:
        cv2.putText(dice_vis,"NO TRAY DETECTED",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

    # Stack: original | red mask | dice mask overlay
    h = frame.shape[0]
    s = (400, h)
    p1 = cv2.resize(dice_vis, s)
    p2 = cv2.resize(red_bgr,  s)
    cv2.putText(p1,"Camera + ROI",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1)
    cv2.putText(p2,"Red mask (tray)",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1)
    out = np.hstack([p1, p2])
    cv2.imshow("Tray Test", out)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if key == ord('s'):
        cv2.imwrite("tray_debug.jpg", out)
        print("Saved tray_debug.jpg")

cap.release()
cv2.destroyAllWindows()
