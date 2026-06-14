# Raspberry Pi deployment

Move the tracker off the bench PC onto the portable rig. The Pi runs the
**torch-free** path automatically — same app, same logic, light deps.

## What makes the Pi path work

`inference_backend.py` picks the backend at import:
- **PC**: ultralytics + supervision (torch) — unchanged dev experience.
- **Pi**: `onnx_backend.py` — pure onnxruntime + numpy. Custom YOLOv11
  decoder, agnostic NMS, an IoU tracker (ByteTrack-equivalent) and a box
  smoother. Verified **bit-for-bit identical** detections to the
  ultralytics path on banked frames (2026-06-14).

`hardware.py` adds the 4 buttons / 4 LEDs / 2 OLEDs against the locked
pin map (DESIGN.md). It is a silent no-op when gpiozero/luma are absent,
so the same code runs on the PC.

## First bring-up (with an HDMI display attached)

This first portable test uses a monitor on the Pi — you want to watch the
feed while validating the rig, and the camera-alignment overlay + button
input currently go through the OpenCV window. (Fully headless mode =
alignment from the phone, no cv2 window = the documented next step.)

1. On the PC, push the latest code + the deployed model:
   ```
   git push                       # already done from the PC
   ```
2. On the Pi:
   ```
   git clone https://github.com/Sevcav/dice-tracker.git   # or git pull
   cd dice-tracker
   bash deploy/setup_pi.sh
   ```
   The script: apt deps, enables SPI, builds a venv with
   `requirements-pi.txt`, verifies `BACKEND == onnx`, sets the
   `dicetracker.local` mDNS hostname, lists camera devices.
3. Copy the model if not in git (ONNX is gitignored):
   ```
   scp training/models/combined.onnx        sevcav@dicetracker.local:~/dice-tracker/training/models/
   scp training/models/combined.onnx.json   sevcav@dicetracker.local:~/dice-tracker/training/models/
   ```
4. Run:
   ```
   . .venv/bin/activate
   python dice_tracker.py
   ```
   Phone UI: `http://dicetracker.local:5000/`

## Autostart at boot (optional)

```
sudo cp deploy/dice-tracker.service /etc/systemd/system/
sudo systemctl enable --now dice-tracker
journalctl -u dice-tracker -f      # watch logs
```

## Checklist when something's off

- `python -c "import inference_backend as b; print(b.BACKEND)"` must print
  `onnx`. If it prints `ultralytics`, torch leaked into the venv — the Pi
  will try to load torch and choke.
- Buttons/LEDs dead? `python hardware.py` prints GPIO/OLED availability and
  the pin map; check the user is in the `gpio`/`spi` groups.
- OLEDs blank? Confirm SPI is enabled (`ls /dev/spidev*`) and both CE0/CE1
  are wired.
- Camera not found? `v4l2-ctl --list-devices`; the USB camera (not the
  dead CSI ribbon) is the path.
- Reads bad? It's almost always alignment or day-mode — the startup
  self-checks catch both; match the green outline and confirm IR mode.
