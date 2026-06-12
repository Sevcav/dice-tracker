"""
calibrate_tray_roi.py
---------------------
One-time interactive calibration of the dice tray's pixel coordinates.

Run this whenever the rig geometry changes (camera moved, tray repositioned,
new resolution, etc.). The saved ROI is then loaded by find_tray_roi() at
runtime, so the detector no longer needs to find a red tray every frame.

Workflow:
    1. Run this script with the camera plugged in
    2. A live preview window opens
    3. Click the 4 corners of the dice tray's INNER playing surface
       (where dice actually land — inside the tray walls)
       Order: top-left, top-right, bottom-right, bottom-left
    4. Press SPACE to confirm or R to redo the clicks
    5. The axis-aligned bounding rectangle of those 4 points is saved
       to tray_roi.json

Controls:
    Left click  = place a corner (up to 4)
    R           = reset all corners
    SPACE       = save and exit (only enabled after 4 corners placed)
    Q / ESC     = quit without saving
"""

import cv2
import json
import os
from pathlib import Path

# --- Config ---
CAMERA_INDEX = 0
RESOLUTION   = (1920, 1080)
OUT_PATH     = Path(__file__).parent / "tray_roi.json"

# --- State ---
corners: list[tuple[int, int]] = []
display_scale = 0.6   # window is scaled-down preview; clicks are scaled back up


def on_mouse(event, x, y, flags, param):
    """Record clicks (in display coords) and translate to original-frame coords."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(corners) >= 4:
        return
    # Translate display click -> original frame coords
    fx = int(x / display_scale)
    fy = int(y / display_scale)
    corners.append((fx, fy))
    print(f"  Corner {len(corners)}: ({fx}, {fy})")


def main():
    print("=" * 60)
    print("  Dice Tray ROI Calibration")
    print("=" * 60)
    print(f"Output: {OUT_PATH}")
    print()
    print("Controls:")
    print("  Click 4 corners of the tray's inner playing surface")
    print("  Order: top-left, top-right, bottom-right, bottom-left")
    print("  R     = reset all corners")
    print("  SPACE = save and exit")
    print("  Q/ESC = quit without saving")
    print()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera open at {actual_w}x{actual_h}")
    print()

    win = "Tray Calibration - click 4 corners (TL, TR, BR, BL)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(actual_w * display_scale),
                          int(actual_h * display_scale))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = cv2.resize(frame, None, fx=display_scale, fy=display_scale,
                             interpolation=cv2.INTER_AREA)

        # Draw placed corners
        for i, (fx, fy) in enumerate(corners):
            dx = int(fx * display_scale)
            dy = int(fy * display_scale)
            cv2.circle(display, (dx, dy), 6, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw polygon connecting corners as you go
        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                p1 = (int(corners[i][0]     * display_scale),
                      int(corners[i][1]     * display_scale))
                p2 = (int(corners[i + 1][0] * display_scale),
                      int(corners[i + 1][1] * display_scale))
                cv2.line(display, p1, p2, (0, 255, 0), 2)
            if len(corners) == 4:
                p1 = (int(corners[3][0] * display_scale),
                      int(corners[3][1] * display_scale))
                p2 = (int(corners[0][0] * display_scale),
                      int(corners[0][1] * display_scale))
                cv2.line(display, p1, p2, (0, 255, 0), 2)

        # Status text
        if len(corners) < 4:
            status = f"Click corner {len(corners) + 1} of 4"
            color  = (255, 255, 255)
        else:
            status = "All 4 corners placed - press SPACE to save, R to redo"
            color  = (0, 255, 255)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            print("Quit without saving.")
            break

        if key == ord('r'):
            corners.clear()
            print("Reset corners.")
            continue

        if key == 32 and len(corners) == 4:   # SPACE
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            roi = {
                "x":      x_min,
                "y":      y_min,
                "w":      x_max - x_min,
                "h":      y_max - y_min,
                "frame_width":  actual_w,
                "frame_height": actual_h,
                "corners":      corners,    # full quad for future perspective work
            }
            with open(OUT_PATH, "w") as f:
                json.dump(roi, f, indent=2)
            print(f"Saved ROI to {OUT_PATH}")
            print(f"  Bounding box: x={roi['x']}, y={roi['y']}, "
                  f"w={roi['w']}, h={roi['h']}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
