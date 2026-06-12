"""
debug_stability.py
------------------
Shows real-time detection count and centroid stability data so we can
see exactly why the tracker isn't settling.

Press Q to quit.
"""
import cv2
import numpy as np
import sys, os, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.detect_dice import find_tray_roi, detect_dice, SETTLE_FRAMES, SETTLE_MOVE

cap = cv2.VideoCapture(0)
cv2.namedWindow("Stability Debug", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stability Debug", 900, 600)

tray_roi    = None
frame_count = 0
history     = []   # list of centroid lists
count_log   = collections.deque(maxlen=30)  # last 30 frame counts

print("Watching detection stability — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_count += 1

    if frame_count % 60 == 1:
        new_roi = find_tray_roi(frame)
        if new_roi:
            tray_roi = new_roi

    dets = detect_dice(frame, roi=tray_roi)
    centroids = [(d.bbox[0] + d.bbox[2]//2, d.bbox[1] + d.bbox[3]//2)
                 for d in dets]
    history.append(centroids)
    if len(history) > SETTLE_FRAMES + 2:
        history.pop(0)

    count_log.append(len(dets))

    # ── Build display ─────────────────────────────────────────────────────────
    vis = frame.copy()

    # Draw detections
    for i, d in enumerate(dets):
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(vis, str(i+1), (x+4, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

    # Draw tray ROI
    if tray_roi:
        rx, ry, rw, rh = tray_roi
        cv2.rectangle(vis, (rx,ry), (rx+rw,ry+rh), (200,80,0), 1)

    # ── Stats panel ───────────────────────────────────────────────────────────
    panel = np.zeros((600, 300, 3), dtype=np.uint8)
    y = 20
    def pt(text, color=(200,200,200)):
        global y
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += 18

    pt(f"Frame: {frame_count}", (150,150,150))
    pt(f"Dice now: {len(dets)}", (0,255,0) if len(dets) > 0 else (0,0,255))
    pt("")

    # Count stability
    counts = list(count_log)
    if counts:
        modal = max(set(counts), key=counts.count)
        inconsistent = sum(1 for c in counts if c != modal)
        pt(f"Last 30 counts: modal={modal}", (255,200,0))
        pt(f"  inconsistent frames: {inconsistent}",
           (0,255,0) if inconsistent <= 3 else (0,100,255))

    pt("")
    pt(f"Need {SETTLE_FRAMES} stable frames", (200,200,200))

    # Centroid movement
    if len(history) >= 2:
        recent = history[-min(SETTLE_FRAMES, len(history)):]
        modal_count = max(set(len(h) for h in recent),
                         key=lambda x: sum(1 for h in recent if len(h)==x))
        stable = [h for h in recent if len(h) == modal_count]
        pt(f"Stable frames: {len(stable)}/{SETTLE_FRAMES}",
           (0,255,0) if len(stable) >= SETTLE_FRAMES else (0,180,255))

        if len(stable) >= 2 and modal_count > 0:
            pt("")
            pt("Centroid movement (px):", (200,200,200))
            for i in range(modal_count):
                xs = [f[i][0] for f in stable if i < len(f)]
                ys = [f[i][1] for f in stable if i < len(f)]
                if len(xs) >= 2:
                    dx = max(xs) - min(xs)
                    dy = max(ys) - min(ys)
                    ok = dx <= SETTLE_MOVE and dy <= SETTLE_MOVE
                    col = (0,255,0) if ok else (0,80,255)
                    pt(f"  Die {i+1}: dx={dx} dy={dy} {'OK' if ok else 'MOVING'}", col)

    # Count history bar
    y = 520
    pt("Count history (last 30 frames):", (150,150,150))
    for j, c in enumerate(list(count_log)):
        col = (0,255,0) if c == (modal if counts else 0) else (0,80,255)
        cv2.rectangle(panel, (8+j*9, 580-c*8), (15+j*9, 580), col, -1)

    # Combine
    vis_small = cv2.resize(vis, (600, 600))
    combined  = np.hstack([vis_small, panel])
    cv2.imshow("Stability Debug", combined)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
