"""
camera.py
---------
Camera feed abstraction.  Supports:
  - USB webcam (default, Phase 1)
  - HTTP MJPEG stream (future Arduino/IP camera, Phase 2)

Usage:
    from detection.camera import Camera
    cam = Camera()          # USB webcam
    cam = Camera(source="http://192.168.1.x/stream")  # IP camera
    frame = cam.read()      # returns BGR numpy array or None
    cam.release()
"""

import cv2
import numpy as np


class Camera:
    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720):
        """
        Parameters
        ----------
        source : int or str
            0 = first USB webcam (default)
            1, 2 ... = other USB cameras if multiple connected
            "http://..." = MJPEG stream URL for IP/Arduino cameras
        width, height : int
            Requested capture resolution. Camera may not support all values.
        """
        self.source = source
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {source}\n"
                "Check that your USB camera is plugged in and not used by another app."
            )

        # Request resolution (not all cameras honour this)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize lag

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened source={source}  resolution={actual_w}x{actual_h}")

    def read(self) -> np.ndarray | None:
        """Read one frame.  Returns BGR numpy array, or None on failure."""
        ret, frame = self._cap.read()
        return frame if ret else None

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()
        print("[Camera] Released.")

    @staticmethod
    def list_usb_cameras(max_test: int = 5) -> list[int]:
        """
        Probe camera indices 0..max_test-1 and return the ones that open.
        Handy for finding your camera index if 0 doesn't work.
        """
        available = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
            cap.release()
        return available


if __name__ == "__main__":
    # Quick test: show live feed, press Q to quit
    print("Available camera indices:", Camera.list_usb_cameras())
    cam = Camera(source=0)
    print("Press Q to quit preview.")
    while True:
        frame = cam.read()
        if frame is None:
            print("No frame received.")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
