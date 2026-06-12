"""
capture_frames.py
-----------------
Capture full-frame dice photos for upload to Roboflow.

Labels are NOT applied here — labeling happens in Roboflow's web UI after
upload, where each die in each frame gets its own bounding box + class.

Workflow:
    1. Plug in the Arducam, ensure IR mode is engaged (photoresistor sealed)
    2. Run this script — a live preview window opens
    3. Arrange / roll dice into the tray
    4. Press SPACE to save the current frame
    5. Repeat with varied positions, orientations, and dice types
    6. Press Q to quit
    7. Upload the contents of `capture_sessions/<session>/` to Roboflow

Output:
    capture_sessions/<YYYY-MM-DD_HHMMSS>/frame_NNNN.jpg

Each session goes into its own timestamped folder so you can capture
in batches over time without fear of overwriting old frames.

Capture targets (per the project plan):
    - Block dice first  — pow, push, both_down, player_down, stumble
    - Then BB d6        — 1, 2, 3, 4, 5, bb_logo
    - Then D16          — 1-16
    Aim for ~150 frames per dice type with varied dice arrangements.
"""

import cv2
import os
import time
from pathlib import Path

# --- Config ---
CAMERA_INDEX = 0
RESOLUTION   = (1280, 720)
ROOT         = Path(__file__).parent / "capture_sessions"


def main():
    session_name = time.strftime("%Y-%m-%d_%H%M%S")
    session_dir  = ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Roboflow Frame Capture")
    print("=" * 60)
    print(f"Session: {session_name}")
    print(f"Output:  {session_dir}")
    print()
    print("Controls:")
    print("  SPACE = save current frame")
    print("  Q/ESC = quit")
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

    win = "Capture - SPACE to save, Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    saved_count = 0
    last_saved_until = 0.0
    last_saved_msg   = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed")
            continue

        display = frame.copy()
        # HUD: counter top-left
        cv2.putText(display, f"Saved this session: {saved_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(display, "SPACE = save   Q = quit",
                    (10, actual_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 2)

        # Flash confirmation message after each save
        if time.time() < last_saved_until:
            cv2.putText(display, last_saved_msg,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        if key == 32:   # SPACE
            saved_count += 1
            ts       = time.strftime("%Y%m%d_%H%M%S")
            ms       = int((time.time() % 1) * 1000)
            filename = f"frame_{saved_count:04d}_{ts}_{ms:03d}.jpg"
            out_path = session_dir / filename
            cv2.imwrite(str(out_path), frame)
            last_saved_msg   = f"Saved {filename}"
            last_saved_until = time.time() + 1.5
            print(f"  [{saved_count:04d}] {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print(f"Session complete: {saved_count} frames saved")
    print(f"Folder: {session_dir}")
    print()
    if saved_count > 0:
        print("Next steps:")
        print(f"  1. Zip the folder: {session_dir}")
        print("  2. Upload to your Roboflow project")
        print("  3. Label each die with bounding boxes in the Roboflow UI")
    print("=" * 60)


if __name__ == "__main__":
    main()
