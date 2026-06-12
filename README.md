# Blood Bowl Dice Tracker

A portable dice-roll capture rig for Blood Bowl games and tournaments.
Players roll into a felt tray; a camera reads the dice with a YOLO model;
the rolling player confirms with a button; every roll is logged to a
database. At the end of the game you get a per-player, per-face dice
record (and the statistical proof that your dice really were cursed).

**Target hardware:** Raspberry Pi 4B + Arducam B0205 IR USB camera in a
3D-printed two-tier rig with 4 arcade buttons and 2 OLED displays.
Currently developed and validated on a Windows PC bench rig.

## How it works

1. `dice_tracker.py` watches the tray. A combined 27-class YOLOv11n
   model reads **block dice, d6, and d16 in one inference** — the dice
   type is identified automatically from the detected faces.
2. When the dice settle (stable count + stable labels), the read locks
   and the border flashes green. Low-confidence reads are flagged with
   an orange `?` — nudge that die and it re-reads in place.
3. Confirm logs the roll to SQLite under the active player. Reject saves
   the frame (with your correction) as future training data.
4. The phone web UI (`http://<rig-ip>:5000/`) shows the live read,
   switches dice type manually if ever needed, and serves the post-game
   record: face tallies per player, roll log, corrections, CSV export.

## Quick start (PC bench)

```
pip install -r requirements.txt
python dice_tracker.py            # alignment overlay -> SPACE -> play
python webapp.py                  # standalone review server (no camera)
python eval_harness.py --type block   # measure real per-die accuracy
```

Every camera session starts with two built-in pre-flights: an IR-mode
self-check (the models only know IR frames) and a tray-alignment overlay
(the models only know the calibrated perspective).

## Key documents

- **`DESIGN.md`** — single source of truth: hardware specs, detection
  architecture, locked decisions, lessons learned.
- **`HANDOFF.md`** — current state + next priorities for a new session.
- **`SETUP.txt`** — the original settle-then-confirm UX intent.

## Training

Labeling in Roboflow (free tier), training local via Ultralytics
(`training/train_all.py`, YOLOv11n). `training/merge_datasets.py` builds
the combined 27-class dataset from the per-type projects. Models export
to ONNX and transfer to the Pi by SCP (binaries are not committed).
Retraining policy: a new model must hit **mAP@50 ≥ 92%** on held-out
validation before it replaces production.
