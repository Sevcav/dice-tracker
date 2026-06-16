# Rig cheat sheet

Quick copy-paste commands. The stuff I always forget.

## SSH into the rig (from the PC)

```bash
ssh sevcav@dicetracker.local
```

User `sevcav`, mDNS hostname `dicetracker.local`. Prompts for the Pi password.
If `dicetracker.local` won't resolve (network blocks mDNS), use the LAN IP:

```bash
ssh sevcav@<pi-ip-address>
```

## Activate the venv (BEFORE running anything by hand)

It's `.venv`, NOT `venv`. luma/gpiozero/onnxruntime live here.

```bash
source .venv/bin/activate
```

Prompt shows `(.venv)` when it's active — if you see that, it's already on,
skip this.

## Run the tracker by hand (dev / debugging)

```bash
cd ~/dice-tracker
git pull
source .venv/bin/activate
python dice_tracker.py
```

## Run with NO PC (store mode) — autostart at boot

Install ONCE (needs SSH this one time):

```bash
sudo cp deploy/dice-tracker.service /etc/systemd/system/
sudo systemctl enable --now dice-tracker
```

After that, every power-on starts the tracker automatically. At the store:
just plug in, wait ~30-60s, then use the phone — no terminal needed.

- Phone alignment: `http://dicetracker.local:5000/align`  (always step one)
- Phone live view + controls: `http://dicetracker.local:5000/`

### systemd controls (when SSH'd in)

```bash
systemctl status dice-tracker      # is it running?
sudo systemctl restart dice-tracker
sudo systemctl stop dice-tracker   # stop it (e.g. to run by hand)
journalctl -u dice-tracker -f      # watch live logs
```

## OLED tests (deploy/)

```bash
python deploy/oled1_test.py       # only CE0/pin24 (P1, left)
python deploy/oled2_test.py       # only CE1/pin26 (P2, right)
python deploy/oled_both_test.py   # BOTH at once, distinct text per screen
python hardware.py                # prints GPIO/OLED availability + pin map
```

## Quick gotchas

- Model `combined.onnx` is gitignored — must already be at
  `~/dice-tracker/training/models/`. If the service crash-loops, check this
  first: `ls ~/dice-tracker/training/models/combined.onnx`
- OLEDs share RST (GPIO25). Only the FIRST ssd1309 device may drive RST;
  the second uses `gpio_RST=None` or it blanks the first panel.
- Backend must be onnx on the Pi:
  `python -c "import inference_backend as b; print(b.BACKEND)"` → `onnx`
