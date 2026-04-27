"""
debug_settle.py
---------------
Prints detection count every frame for 10 seconds with dice sitting still.
This tells us exactly what's fluctuating and why.
"""
import cv2, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection.detect_dice import find_tray_roi, detect_dice

cap = cv2.VideoCapture(0)
# Let camera settle
for _ in range(30):
    ret, frame = cap.read()

tray_roi = None
new_roi = find_tray_roi(frame)
if new_roi:
    tray_roi = new_roi
    print(f"ROI: {tray_roi}")
else:
    print("NO ROI FOUND")

print("\nRoll dice and leave them still. Watching for 10 seconds...\n")
print("Frame | Count | Centroids")
print("-" * 60)

start = time.time()
frame_num = 0
while time.time() - start < 10:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_num += 1
    dets = detect_dice(frame, roi=tray_roi)
    centroids = [(d.bbox[0]+d.bbox[2]//2, d.bbox[1]+d.bbox[3]//2) for d in dets]
    print(f"  {frame_num:3d}  |   {len(dets)}   | {centroids}")

cap.release()
print("\nDone. Look for frames where count changes or centroids jump.")
