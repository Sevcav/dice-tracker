# Rig cheat sheet

Quick copy-paste commands. The stuff I always forget.

## Keep docs/ off the Pi (sparse-checkout — one-time setup)

The repo holds a `docs/` folder (showcase document + images, ~15 MB) that
the Pi does NOT need. Configure the Pi's clone to pull code only, skipping
`docs/`, so pulls stay small. Run ONCE on the Pi:

```bash
cd ~/dice-tracker
git sparse-checkout init --cone
git sparse-checkout set '/*'          # everything at the repo root...
git sparse-checkout add '!/docs'      # ...except docs/  (cone mode excludes it)
git read-tree -mu HEAD                 # apply: drops docs/ from the working tree
```

If the above cone syntax gives trouble on an older git, use non-cone mode:

```bash
cd ~/dice-tracker
git sparse-checkout init --no-cone
printf '/*\n!/docs/\n' > .git/info/sparse-checkout
git read-tree -mu HEAD
```

Verify: `ls ~/dice-tracker` should show NO `docs/` folder. Normal
`git pull` from then on skips docs/ automatically.

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

## Phone workflows (no PC)

Three tabs: **Align**, **Live**, **Games**. Hard-refresh after a service
restart (phones cache the page).

### Align (startup only)

Alignment runs ONCE at startup, before the live session. The Align tab
shows the camera + green tray outline only during that step; once a
session is live it shows a "session running" notice (the live image is on
the **Live** tab — that's expected, not a bug).

- Adjust the arm so the tray matches the green outline -> **Confirm**.
- Camera moved (e.g. new case)? Tap **Re-set corners**, then tap the
  tray's 4 corners in order **top-left, top-right, bottom-right,
  bottom-left** -> **Save corners** -> **Confirm**. This rewrites
  `tray_roi.json` (overlay quad + the axis-aligned crop the model uses).
  It does NOT dewarp the image — re-square the camera for best accuracy.
- To re-align mid-session: `sudo systemctl restart dice-tracker`, then
  re-do alignment before confirming.

### IR mode (must be IR — always)

Models are trained on IR (monochrome) frames. The rig forces IR via the
photoresistor lens cap / light shield.

- **Day-mode warning ON = a real problem** (shield not sealing / too
  bright). No warning = IR = good.
- Check on the **Live** tab (monochrome = IR) or in the logs:
  `journalctl -u dice-tracker -f` -> `[ok] camera back in IR mode` /
  `[WARNING] camera flipped to DAY mode`.
- The first few frames after camera plug-in are always day-mode color
  while it drops into IR — ignore the startup blip.

### Games (clear test data)

- **Games** tab -> per-game **delete** (removes the game + its rolls), or
  **Clear all games** at the top to wipe everything.
- The game currently being recorded shows **active** instead of delete and
  is protected from both delete and Clear all.
- Clear out bring-up/test games here before a real tournament.

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
