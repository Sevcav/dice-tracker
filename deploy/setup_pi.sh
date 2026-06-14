#!/usr/bin/env bash
# setup_pi.sh — one-shot Raspberry Pi setup for the Blood Bowl Dice Tracker.
# Run ON THE PI from the repo root:  bash deploy/setup_pi.sh
#
# Idempotent: safe to re-run. Does NOT install torch/ultralytics/supervision
# (the Pi uses the torch-free onnx backend).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
echo "== BB Dice Tracker Pi setup =="
echo "repo: $REPO"

# 1. System packages (camera/SPI/I2C deps + GPIO/OLED libs via apt) ----------
echo "-- apt packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libatlas-base-dev \
  python3-gpiozero python3-luma.oled avahi-daemon v4l-utils

# 2. Enable SPI (needed for the two SSD1309 OLEDs) ---------------------------
echo "-- enabling SPI"
sudo raspi-config nonint do_spi 0 || true   # 0 = enable

# 3. Python venv with the torch-free deps ------------------------------------
echo "-- python venv"
python3 -m venv --system-site-packages .venv   # system-site so apt gpiozero/luma are visible
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt

# 4. Verify the backend resolves to the torch-free path ----------------------
echo "-- backend self-check"
python - <<'PY'
import inference_backend as b
assert b.BACKEND == "onnx", f"WRONG BACKEND: {b.BACKEND} (torch leaked in?)"
print("backend OK:", b.BACKEND)
from onnx_backend import OnnxModel
m = OnnxModel("training/models/combined.onnx")
print("model OK:", len(m.names), "classes")
PY

# 5. mDNS hostname so the phone can reach http://dicetracker.local:5000 ------
echo "-- mDNS (dicetracker.local)"
sudo raspi-config nonint do_hostname dicetracker || true
sudo systemctl enable --now avahi-daemon

# 6. Camera check -------------------------------------------------------------
echo "-- camera devices"
v4l2-ctl --list-devices || echo "(no v4l2 devices listed — check USB camera)"

cat <<EOF

== Setup complete ==
Run the tracker:
    . .venv/bin/activate
    python dice_tracker.py

Phone UI:  http://dicetracker.local:5000/   (or the LAN IP printed at startup)
Autostart at boot (optional):
    sudo cp deploy/dice-tracker.service /etc/systemd/system/
    sudo systemctl enable --now dice-tracker

First run will show the IR self-check then the camera-alignment overlay —
match the tray to the green outline and press SPACE (alignment is always
step one; the model only knows the calibrated perspective).
EOF
