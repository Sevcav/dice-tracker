"""
align_camera.py
---------------
Show the saved tray-corner reference (from tray_roi.json) as an overlay
on the live camera feed, so you can adjust the rig's camera-arm bolts
until the live view matches the original training geometry.

This is the tool for restoring the camera position after a bump.

Workflow:
    1. Run this script with the camera plugged in
    2. A live preview window opens at 1920x1080
    3. A GREEN quadrilateral shows where the tray corners WERE during
       the original calibration (training-time position)
    4. Loosen the camera-arm bolts, adjust the camera until the actual
       tray edges line up with the green outline, retighten
    5. Press Q to quit

The closer the live tray edges match the green outline, the closer
the camera is to the original training position. Perfect alignment
gives the YOLO model the exact perspective it learned.

Controls:
    Q / ESC = quit
    S       = save current frame to align_snapshots/ for reference

If tray_roi.json doesn't exist, run calibrate_tray_roi.py first.
"""

import cv2
import json
import time
from pathlib import Path

CAMERA_INDEX = 0
RESOLUTION   = (1920, 1080)
ROI_PATH     = Path(__file__).parent / "tray_roi.json"
SNAP_DIR     = Path(__file__).parent / "align_snapshots"
SNAP_DIR.mkdir(exist_ok=True)

DISPLAY_SCALE = 0.5  # the preview window is scaled down


def main():
    if not ROI_PATH.exists():
        print(f"ERROR: {ROI_PATH} not found.")
        print("Run calibrate_tray_roi.py first to save a reference.")
        return

    with open(ROI_PATH) as f:
        roi = json.load(f)

    saved_w = roi.get("frame_width")
    saved_h = roi.get("frame_height")
    corners = roi.get("corners", [])

    if len(corners) != 4:
        print(f"ERROR: tray_roi.json has {len(corners)} corners, expected 4.")
        return

    print("=" * 70)
    print("  Camera Alignment Helper")
    print("=" * 70)
    print(f"Reference: {ROI_PATH}")
    print(f"  saved at {saved_w}x{saved_h}")
    print(f"  TL ({corners[0][0]:4d}, {corners[0][1]:4d})  "
          f"TR ({corners[1][0]:4d}, {corners[1][1]:4d})")
    print(f"  BR ({corners[2][0]:4d}, {corners[2][1]:4d})  "
          f"BL ({corners[3][0]:4d}, {corners[3][1]:4d})")
    print()
    print("Adjust the camera arm until the live tray edges line up")
    print("with the green outline. Q or ESC to quit. S to save snapshot.")
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

    # Scale saved corners if camera reports a different resolution than the
    # saved calibration. This lets the overlay work even if the camera can't
    # honour 1920x1080 for some reason.
    sx = actual_w / saved_w
    sy = actual_h / saved_h
    scaled_corners = [(int(x * sx), int(y * sy)) for x, y in corners]
    if (sx, sy) != (1.0, 1.0):
        print(f"  Scaling reference corners by ({sx:.3f}, {sy:.3f}) "
              "to match current capture resolution")

    win = "Camera Alignment - Match the green outline - Q to quit, S to save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(actual_w * DISPLAY_SCALE),
                          int(actual_h * DISPLAY_SCALE))

    last_save = ""
    last_save_until = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed")
            continue

        # Draw the reference quadrilateral
        pts = [(int(x), int(y)) for x, y in scaled_corners]
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            cv2.line(frame, p1, p2, (0, 255, 0), 3)
            cv2.circle(frame, p1, 8, (0, 255, 0), -1)
            label = ["TL", "TR", "BR", "BL"][i]
            cv2.putText(frame, label, (p1[0] + 12, p1[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # HUD
        cv2.putText(frame, "Align live tray edges to GREEN outline",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, "Align live tray edges to GREEN outline",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Q=quit  S=save snapshot",
                    (15, actual_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, "Q=quit  S=save snapshot",
                    (15, actual_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if time.time() < last_save_until:
            cv2.putText(frame, last_save, (15, actual_h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Scaled-down preview
        display = cv2.resize(frame, None,
                             fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = SNAP_DIR / f"align_{ts}.jpg"
            cv2.imwrite(str(out_path), frame)
            last_save = f"Saved {out_path.name}"
            last_save_until = time.time() + 1.5
            print(last_save)

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
