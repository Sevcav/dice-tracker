# Blood Bowl Dice Tracker — Design Document

**Last updated:** June 12, 2026
**Status:** Full pipeline working end-to-end on the PC bench rig. Combined
27-class YOLOv11n deployed — **auto dice-type detection** (no manual
switching; phone/keys remain as override). Settle-race bug fixed
(COUNT_STABLE_FRAMES — missed-dice rolls 32% → 8%). Startup pre-flights:
IR-mode self-check + camera-alignment overlay in both `dice_tracker.py`
and `eval_harness.py`. Real accuracy numbers exist (block per-die ~83-87%;
pow is the weak class at ~60% recall — fix is the tray-crop retrain).
SQLite database + phone web app live: dice-type control, live read
display, post-game corrections (dropdowns), CSV export, and the
**BB3-style end-of-game dice record** with icons drawn from the actual
dice. All committed + pushed. See `HANDOFF.md` for the session hand-off
prompt, and memory entity `BB Dice Tracker Current State 2026-06-12`.

**Retraining policy (locked):** Any retrain (block / d6 / d16) using accumulated rejection frames must still achieve **mAP ≥ ~92%** (block baseline) on the held-out validation set before its ONNX replaces the production model. If retrain regresses below baseline, do not deploy — revert and try a different mix of training data.

**Combined model + auto dice-type detection (deployed 2026-06-11):**
manual dice-type switching proved unworkable in play (players alternate
block/d6 constantly; a wrong-model read logs confident garbage). Cheap
discriminators failed offline (confidence-vote 73%, brightness
inconclusive), so the original single-combined-model plan was revived:
the three datasets were merged (`training/merge_datasets.py`, 1126
images, 27 classes) and one YOLOv11n trained locally. Per-type mAP@50 vs
the separate models: block 92.4% (−3.7, still over the 92% bar), d6
98.4% (−1.1), d16 87.4% (−1.1). `dice_tracker.py` defaults to **auto
mode** (`combined.onnx`): dice type is derived from the detected face
labels per roll; per-type models remain as manual overrides (keys
B/D/X, phone buttons). Pi cost unchanged — one nano inference per frame.

**Session startup rule (locked 2026-06-11):** **Camera alignment is ALWAYS the first step of any camera session.** Moving the rig moves the camera, and the models only know the calibrated tray perspective in `tray_roi.json`. Never score, capture, or log rolls before alignment is confirmed. On the PC (GUI) this is the on-screen green-outline overlay → SPACE; on the **headless Pi** it is the phone alignment screen (`/align`: live feed + green outline → Confirm). Standalone `align_camera.py` remains for bench work.

**Headless rule (locked 2026-06-14):** **The production rig has NO monitor — it is a sealed box and the phone is the only screen.** `dice_tracker.py` runs headless automatically on the Pi (no `$DISPLAY` / torch-free onnx backend): no OpenCV window ever opens; alignment, live read, and Reject/Undo/Confirm are all on the phone, alongside the 4 GPIO buttons. The cv2 preview window is dev-only (`--gui` on a PC). Do not reintroduce a runtime dependency on an attached display.

---

## 1. Project Vision

A portable, self-contained device that captures Blood Bowl dice rolls at gaming
stores and tournaments. Players roll into a tray, the camera reads the dice,
the result is attributed to the rolling player via a dedicated button, and the
roll is logged to a database for later analysis via a web app.

**Not a livestream/replay device.** A **data capture device**.

### Core requirements

- Self-contained — no PC on the table
- Works under variable lighting (gaming store conditions)
- Powered by USB power bank for portability
- Modular 3D-printed rig (lower tier + lid + removable dice tray + power bank shelf)
- Phone connects over WiFi for the web UI

---

## 2. Production Rig Design (Active)

The production rig is a **modular two-tier design** with a removable dice tray.

### Architecture

| Tier | Purpose |
|---|---|
| **Lower tier** | Electronics enclosure — Pi, buttons, LEDs, OLEDs, camera mount |
| **Lid** | Sits on top of lower tier, has tray opening + posts hanging down to bolt to lower tier |
| **Dice tray (removable)** | Drops into the lid's tray opening, felt-lined, removable for felt application and Pi access |
| **Power bank shelf** | External, attaches to rear of rig |

### Outer Dimensions

- **Outer footprint:** 260 × 190mm (W × D) with 55 × 55mm chamfered front corners
- **Lower tier height:** 60mm
- **Lid height:** 60mm
- **Total stack height:** 120mm + tray protrusion

### Material

- **PETG** throughout (heat resistance for hot summer venues)
- 4mm wall thickness on all major shells

---

## 3. Component Status

| Component | Status | Notes |
|---|---|---|
| **Lower tier** | ✅ Modeled | All cuts complete: buttons, LEDs, OLED windows, camera mount |
| **Lid (with posts)** | ✅ Modeled | Tray opening + 4 hanging posts with captive M4 nuts |
| **Dice tray (removable)** | ⏳ To design | Felt-lined insert, drops into lid tray opening |
| **Camera base plate** | ⏳ To design | Replaces MakerWorld Camera_Base bottom; bolts into lower tier rear detent |
| **Power bank shelf** | ⏳ To design | External rear shelf for UGREEN power bank |
| **Camera cradle** | 🔧 Blocked | Need Arducam in hand for pocket sizing |
| **Photoresistor light shield** | ✅ Printed | Black cover over the B0205 photoresistor forcing IR mode. NOTE: frame analysis 2026-06-11 found the May 13 *morning* session ran in day mode (color cast) while the afternoon ran in IR — verify the shield fully seals against bright-room light |
| **Bench prototype tray cradle** | ✅ Printed | Standalone test, validated soft-tray support concept |
| **Bench prototype arm foot** | ✅ Printed | Bridges over cradle back posts; captive nuts; friction-fit |
| **Camera_Base + Camera_Link** | ✅ Printed | From MakerWorld 627829, will be re-used for production rig |

---

## 4. Production Rig — Detailed Specs

### 4.1 Lower Tier

| Spec | Value |
|---|---|
| Outer footprint | 260 × 190mm with 55mm front chamfers |
| Wall thickness | 4mm |
| Height | 60mm |
| Front face: button holes | 4 × Ø27.78mm |
| Front face: LED holes | 4 × Ø8mm (for 7.5mm snap-in bezels) |
| Button positions (X) | ±56.7 and ±18.9mm |
| Button vertical (Z) | ~33mm centered |
| LED positions (X) | Same as buttons |
| LED vertical (Z) | ~9mm |
| Both chamfered corners | OLED window (62.53 × 40.28mm) + PCB recess (3mm deep) |
| Rear face: camera mount detent | 50 × 50 × 3mm pocket on outside |
| Rear inside wall: nut pockets | 4× hex pockets, 0.25mm deep, on inside of rear wall |
| Rear bolt clearance | 4 × Ø4.5mm holes through rear wall |
| Camera mount bolt path | Outside (camera plate) → through 4.5mm clearance → into captive nut on inside |

### 4.2 Lid

| Spec | Value |
|---|---|
| Outer footprint | 251.44 × 181.44mm with chamfered front corners (matches lower tier outline) |
| Wall thickness | 4mm |
| Height | 60mm |
| Tray opening (visible from top) | 160mm wide × 130mm deep |
| Tray opening lip | 4mm wide step around perimeter, 2mm deep recess |
| Tray opening biased toward rear | Yes |
| Posts (4) | Round Ø10mm columns hanging from lid underside |
| Post length | 54mm (engages 2mm into lower tier floor pocket) |
| Post positions | Front: (±110, -30); Rear: (±115, +85) |
| Post captive nut | M4 hex pocket at bottom of each post (3.4mm deep) |
| Post bolt clearance | Ø4.5mm hole through center of each post |
| Bolt entry | From bottom of lower tier (countersunk) |

### 4.3 Dice tray (still to design)

| Spec | Value |
|---|---|
| Outer flange (sits on lid lip) | 159.7 × 138.7mm × 2mm tall |
| Tray body (drops through lid) | 153.7 × 132.7mm |
| Total height | ~52mm |
| Wall thickness | 4mm |
| Floor thickness | 3mm |
| Inner draft | 5° outward (top wider) |
| Felt lining | Self-adhesive felt sheet, applied after print |

### 4.4 Camera base plate (still to design)

| Spec | Value |
|---|---|
| Plate size | ~50 × 50mm (sized to fit in 50×50mm rear detent of lower tier) |
| Plate thickness | 3mm (recessed flush into detent) |
| Top mating | Hinge clevis matching Camera_Link's M4 hinge |
| Bottom mating | 4× M4 clearance holes matching lower tier captive nut pattern |

### 4.5 Power bank shelf (still to design)

| Spec | Value |
|---|---|
| Power bank dimensions | 160.5 × 81 × 26.5mm (UGREEN Nexode) |
| Shelf attachment | Snap-fits or bolts to rear face of lower tier |
| Cable management | Routes USB-C power to rear of rig into Pi |

---

## 5. Detection Software

### Pipeline (current — YOLO architecture)

1. **Frame capture** — `capture_frames.py` saves raw 1280×720 frames in IR mode
2. **Stability tracker** — wait for dice to settle (no motion) — to be carried over from old code
3. **YOLO inference** — single ONNX model produces per-die `(class, x, y, w, h, conf)` boxes
4. **Player attribution** — handled at the rig (see Player attribution section)

The legacy rule-based pipeline (mask → contours → watershed → CNN classifier)
has been **archived** under `archive/legacy_rule_based_detector/`.
See that folder's README.md for why it was abandoned.

### Dice scope

| Die | Detection | Classes | Status |
|---|---|---|---|
| Block dice (cream) | YOLO | 5 (pow, push, both_down, player_down, stumble) | ✅ Labeled, training overnight 5/4 |
| BB d6 (black) | YOLO | 6 (1-5, bb_logo) | ⏳ After block validates |
| D16 (cream, trapezohedron) | YOLO | 16 (1-16) | ⏳ After d6 — injury rolls |
| D8 (scatter) | 📱 Manual entry on phone | — | — |
| D3 | 📱 Manual entry on phone | — | — |

**Single combined model is preferred** — one YOLO model with all classes vs.
separate models per die type. Easier deployment, single inference call. We
will train block-only first to validate the workflow, then expand the same
model to include d6 then d16.

### Camera angle

Set **empirically** by functional constraints (full tray visible + arm clear
of rolling area), not to a fixed degree number. As-built geometry comes out
shallow / near-overhead — top faces dominate each die crop, side faces are
minimized. The "~35° forward bank shot" figure from earlier docs was an
estimate that did not match the as-built rig.

### Lighting / IR

Production lighting strategy is **forced IR mode** on the Arducam B0205.
This eliminates color/white-balance variability across gaming store
conditions, and IR LED illumination dominates ambient light from windows
and store fluorescents — making the dice appearance consistent regardless
of time of day.

The B0205's IR mode is **photoresistor-controlled only** (no software
toggle available). A 3D-printed light shield over the photoresistor will
force IR mode on regardless of room brightness — see Section 3 component
list.

### Roboflow + training workflow

- **Dataset:** Roboflow Public/Free workspace `bbdicetracker`
- **Project:** "My First Project" (Object Detection, public)
- **Capture script:** `capture_frames.py` saves 1280×720 frames to
  `capture_sessions/<timestamp>/`
- **Labeling:** Roboflow web UI, manual bounding boxes per die
- **Preprocessing:** Auto-Orient, Resize 640×640
- **Augmentation:** Horizontal flip, ±15° rotation, ±20% brightness,
  ≤1px blur, ≤1% noise
- **Train/val/test split:** 70/20/10 (112/32/16 of the 160 frames)
- **Model:** Roboflow 3.0 Object Detection (Fast variant)
- **Checkpoint:** Fine-tune from MS COCO
- **Deployment:** ONNX export → `classifier/` folder → ONNX Runtime on Pi
  (already validated)

### Player attribution

- 4 buttons on rig: **P1 Confirm**, **P2 Confirm**, **Reject**, **Undo**
- Whoever pressed **their** confirm button = the rolling player
- No turn tracking on rig — that lives in the web app
- Phone web app selects dice type (Block / D6 / D16) before each roll

### Correction flow (LOCKED 2026-05-13)

The rig has **4 buttons total**. There is **no keyboard, no number-pad,
no menu navigation** during play. Players will abandon any correction
tool that takes more than a button press. This drives a hard design
principle: **prevention over correction**.

**Pre-confirm correction tools (real-time, frequent use):**

1. **Nudge to re-read** — player physically nudges a misread die.
   `dice_tracker.py` detects per-die motion of more than
   `NUDGE_PIXEL_THRESHOLD = 20px` between consecutive frames and
   automatically releases the settle lock back to "watching" so the
   model re-evaluates. The rest of the roll is preserved; only the
   nudged die gets re-read. Built and validated.
2. **OLED uncertainty markers** *(planned)* — low-confidence labels
   shown with a `?` so the player's attention is drawn to questionable
   dice. High-confidence labels show clean. The OLED's role is to
   make wrong-reads obvious BEFORE the player confirms.
3. **Reject button** — explicit "this read is wrong" before confirm.
   Saves the frame to `retrain_candidates/<type>/` for future model
   improvement.

**Post-confirm correction tools (emergency, rare use):**

4. **Undo button** — removes the LAST logged roll. Player must catch
   the mistake immediately (within ~1 roll). If they realize N rolls
   later, Undo would lose the rolls in between.
5. **Phone web app post-game session review** *(planned, post-MVP)* —
   the phone app's session export/review screen will allow editing
   historical mis-logged rolls. This is the only path for corrections
   discovered late. It is explicitly NOT a real-time tool — it's run
   after the game when speed doesn't matter.

**What we deliberately did NOT build:**

- No keyboard-driven edit flow (`E` to enter manual values for last
  roll) — violates the 4-button constraint and the "no menus" rule.
- No mid-game phone-based correction — adds a context switch that
  slows play; players abandon the tool.
- No "undo by roll number" — too complex for buttons; ambiguous which
  intermediate rolls to preserve.

The cost of this constraint: occasional wrong rolls slip into the log
and aren't caught until later. That is acceptable as long as the
model + OLED-warning + nudge UX keeps the slip rate low. If real-world
play shows the slip rate is too high, the fix is more training data
(retrain to a higher mAP) or richer OLED warning — NOT a richer
correction UI on the rig.

### Accuracy evaluation (added 2026-06-11)

`eval_harness.py` measures **real per-die accuracy** against keyed-in ground
truth, using the exact production detection stack from `dice_tracker.py`.
Per roll: settle → console prints the read (dice numbered left→right,
matching `[n]` overlays in the window) → user types truth (ENTER = correct)
→ clear tray → repeat; `q` prints + saves the report
(`eval_sessions/eval_<type>_<ts>.json/.csv`). Misread frames are saved to
`retrain_candidates/<type>/eval_miss_*.jpg/.json` **with ground truth
attached**, ready for relabeling. The report includes per-die accuracy with
a 95% Wilson CI, roll-level accuracy, per-class recall, confusion pairs,
detection-count errors, and how many rolls were captured in day mode.

Target: ≥50 rolls per dice type (~±3-4% CI at 3 dice/roll). The previous
"86% confirm rate" was 6/7 rolls — 95% CI of 49-97%, not a usable number.

**First block eval (2026-06-11, 50 rolls, 144 dice, all IR):**

- Per-die accuracy 86/96 = **89.6%** (95% CI 81.9–94.2%) on count-correct
  rolls; roll-level 79.4%.
- **Dominant failure: missed dice** — 16/50 rolls lost dice (23 dice never
  read). ROOT CAUSE FOUND: a settle-logic race, not model blindness —
  14/16 saved miss-frames had ALL dice detectable at standard settings on
  the very frame that settled. A detection flickering in/out at the conf
  threshold let the roll settle during an "out" frame. **Fixed** by
  requiring the detection count to hold constant for
  `COUNT_STABLE_FRAMES = 10` consecutive frames before settle (restores
  the count-stability requirement the legacy DiceStabilityTracker had).
  Needs re-eval to confirm.
- Weakest class: **pow, 64% recall** (7/11), confused with stumble ×3.
  More pow training data wanted at next retrain.
- **Confidence threshold 0.85 validated for OLED `?` markers**: 8/10
  wrong reads fell below 0.85; only 18/86 correct reads did.
- The 2 miss-frames not recovered at conf 0.40 full-frame WERE recovered
  by a tray-ROI crop — keep tray-crop inference in the back pocket as an
  accuracy lever.

**Re-eval after settle fix (2026-06-11, 25 rolls, 75 dice):**

- **Count errors collapsed: 16/50 → 2/25** (32% → 8%). Settle-race fix
  validated.
- Per-die accuracy 57/69 = 82.6% (CI 72.0–89.8%) — *lower* than session 1
  because the hard flickery dice that previously vanished as count errors
  now stay in the read and get scored. This is the honest number.
  Combined across both sessions: 143/165 = 86.7%.
- **pow recall 57%** (8/14); combined 15/25 = 60%. pow confusions go both
  directions (read as both_down ×3, stumble ×2). Every other class is
  86–97%.
- **Misread-frame experiment:** re-reading the same frames full-frame fixed
  0/10 (errors are systematic, not flicker). Tray-crop re-read flipped pow
  to CORRECT in 4 rolls — pow is distinguishable at 2× resolution — but
  broke other dice because the model wasn't trained on crops.
- **Conclusion / retrain plan:** the pow fix is RESOLUTION, i.e. retrain
  on tray-cropped data and run inference on crops. Next capture session:
  pow-heavy + wall/corner positions; build the dataset tray-cropped
  (training is local now, we control preprocessing); fold in the banked
  `retrain_candidates/block/` frames with ground truth. ≥92% mAP bar
  applies. Until then the current model stays in production — the 0.85
  OLED `?` threshold catches ~80% of its wrong reads.

Both `dice_tracker.py` and `eval_harness.py` now run two startup
pre-flights, in order:

1. **IR-mode self-check** — refuse-to-start prompt in day mode;
   `dice_tracker.py` also re-checks every ~3s during play with a red HUD
   warning if the camera flips to day mode mid-session.
2. **Camera alignment check** — overlays the saved tray-corner reference
   (`tray_roi.json`) on the live feed; adjust the camera arm until the
   tray matches the green outline, SPACE to continue. The rig moves
   between sessions and the model only knows the calibrated perspective,
   so this always runs first (standalone `align_camera.py` still exists
   for bench work).

### Database + web app (IMPLEMENTED 2026-06-11)

**`db.py`** — SQLite (`dice_tracker.db`, gitignored), short-lived
connections (safe across tracker + Flask threads):

```
games:  id, started_at, ended_at, player1_name, player2_name, notes
rolls:  id, game_id FK, roll_no, timestamp, player (P1|P2), dice_type,
        results JSON, confidences JSON, rejected, edited, raw_image_path
```

`dice_tracker.py` creates the game row lazily on the first logged roll,
inserts on confirm AND reject (rejected=1), deletes on undo, closes the
game on quit. `face_tallies()` produces THE end-of-game record:
per-player, per-dice-type face counts over confirmed rolls.

**`webapp.py`** — Flask phone UI, mobile-first, single file:

- **Live control** (`/`): dice-type buttons (Block/D6/D16) and active
  player buttons; requests flow to the tracker loop through a
  thread-safe `WebControl` mailbox; 1s status poll (state, last roll,
  roll count, day-mode warning).
- **Game review** (`/games`, `/games/<id>`): roll log with confidences +
  reject/edit flags, the face-tally record, per-roll edit (flagged
  `edited` — this is the post-game correction path from the locked
  correction flow), delete, CSV export.
- Started automatically in a daemon thread by `dice_tracker.py` on port
  5000 (`--no-web` disables). Standalone review mode: `python webapp.py`.

**HUD uncertainty markers (OLED precursor, threshold 0.85 from eval
data):** stable+confident = yellow; stable but conf < 0.85 = ORANGE with
`?` and a NUDGE hint; still settling = gray. The same three-state logic
will drive the physical OLEDs.

---

## 6. Hardware

### On hand

- Raspberry Pi 4B in TH3D aluminum case (91 × 65 × 33mm)
- Pi 3B (spare)
- Pi Camera v2.1 with 22" ribbon (CSI port non-functional on this Pi 4 — moved to USB path)
- USB camera (test only, not the production camera)
- Bambu P1S 3D printer
- M3 + M4 hardware in stock (M4 lengths: 8/12/16/20mm)
- M3 + M4 nuts in stock
- 32GB microSD (current, getting full)
- HiLetgo SPI 2.42" OLED 128×64 (verified dimensions): PCB 71×43mm, glass 62.25×40mm, header zone 14mm at top

### On order / shipping

- 64GB A2 microSD
- Arducam 1080P Day/Night IR USB camera (OV2710 sensor)
- Bambu LED Lamp Kit 001 ×2 (USB 5V, built-in PC diffuser)
- WMYCONGCONG arcade buttons (verified: thread Ø26.25mm, dial Ø33mm, body length 62mm — using Ø27.78mm holes for print clearance)
- 5mm pre-wired LEDs in 7.5mm snap-in bezels (using Ø8mm holes)
- UGREEN Power Bank (160.5 × 81 × 26.5mm)
- Pre-crimped JST pigtails (5 red, 5 black, ~500mm, pins both ends)
- M4 brass heat-set inserts (production future use, not yet needed)

### Wiring notes

- **OLEDs (SSD1309, SPI, 7-pin):** All 7 pins used. Two OLEDs share CLK/MOSI/DC/RES/VCC/GND; differentiated by CE0 (GPIO 8) and CE1 (GPIO 7). Requires ~20 female-to-female dupont jumper wires — **not yet on hand**.
- **LEDs:** Pre-wired with resistor inline under heatshrink on red lead. Connect red → GPIO, black → GND directly. No additional resistors needed.
- **Arcade button switches:** PCB-mount microswitch (pin legs, not spade). Use COM + NO only (ignore NC). Solder wires direct to pins or friction-fit female dupont. 2 wires per button × 4 buttons = 8 connections.
- **GPIO allocation:** OLEDs ×2 = 6 pins; Buttons ×4 = 4 pins; LEDs ×4 = 4 pins. Total = 14 of 28 available GPIOs.
- **Library:** luma.oled (pip install luma.oled) — native SSD1309 SPI support.

#### LOCKED GPIO pin map (2026-06-13, BCM numbering)

Verified software-clean (no SPI-bus collision; all plain GPIO with usable
internal pull-ups). OLEDs share the SPI0 bus, distinguished by chip-select.

| Function | BCM GPIO | Physical pin | Wiring |
|---|---|---|---|
| OLED SCLK (both) | GPIO 11 | 23 | SPI0 SCLK, shared |
| OLED MOSI/DIN (both) | GPIO 10 | 19 | SPI0 MOSI, shared |
| OLED DC (both) | GPIO 9 | 21 | shared |
| OLED RES/RST (both) | GPIO 25 | 22 | shared |
| OLED-1 CS (P1 side) | GPIO 8 | 24 | SPI0 CE0 |
| OLED-2 CS (P2 side) | GPIO 7 | 26 | SPI0 CE1 |
| OLED VCC (both) | 3V3 (NOT 5V) | 1 | shared; SSD1309 is a 3.3V part |
| OLED GND (both) | GND | 6 (any GND) | shared rail |
| Button: P1 Confirm | GPIO 17 | 11 | switch → GND, internal pull-up |
| Button: P2 Confirm | GPIO 27 | 13 | switch → GND, internal pull-up |
| Button: Reject | GPIO 22 | 15 | switch → GND, internal pull-up |
| Button: Undo | GPIO 23 | 16 | switch → GND, internal pull-up |
| LED: P1 | GPIO 5 | 29 | red → GPIO, black → GND (inline R) |
| LED: P2 | GPIO 6 | 31 | red → GPIO, black → GND (inline R) |
| LED: Reject | GPIO 13 | 33 | red → GPIO, black → GND (inline R) |
| LED: Undo | GPIO 19 | 35 | red → GPIO, black → GND (inline R) |

Buttons: COM/NO only (NC unused). LEDs already have inline resistors —
no external resistor. Convenient grounds (all the same rail): physical
pins 6, 9, 14, 20, 25, 30, 34, 39.

OLED VCC = **3V3 (pin 1)**, never 5V. GND = any ground pin above.

#### Ground consolidation (decided 2026-06-14, final)

~10 ground wires (2 OLED + 4 button + 4 LED) all share one rail. All are
**26–28 AWG Dupont-terminated**, and the constraints are: case is already
set/printed (NO HAT — a GPIO terminal HAT stands too tall and covers the
header), stay plug-in (no cutting Dupont ends, no soldering), plenty of
internal room.

- **ORDERED 2026-06-14: REXQualis/Ambberdr 400-point self-adhesive
  breadboard** (Amazon's Choice, ~$7.19, 6-pack; 3.22×2.12×0.35"). This
  is the committed ground/3V3-bus part. Listing confirms
  the three musts: "2 Positive & Negative Power Lines on both ends" (the
  common bus), "Self-Adhesive Tape on the Back" (peel-stick mount), and
  "20-29 AWG" wire support (covers the 26–28 AWG Dupont). 170-point minis
  were REJECTED — they have no power rails (only 5-hole tie columns), so a
  10-wire bus would need bridging multiple columns.
  Wiring: stick it inside the rig; jumper any Pi GND pin
  (6/9/14/20/25/30/34/39) into a `-` power line to make it common ground;
  plug all 10 ground Duponts into it. One Pi GND pin used. The `+` line
  buses the two OLED **3V3** wires the same way (jumper from pin 1).
  Note: 400-pt rails are usually split into 2 segments/side — one short
  jumper joins the halves if you ever need >1 segment (you won't for 10).
  NEED ONE F/M jumper for the Pi-pin→rail link (female onto the Pi male
  header pin, male into the breadboard).
- **Rejected:**
  - GPIO screw-terminal HAT — covers the 40-pin header and stands too
    tall for the finished case; also needs Dupont ends cut to bare wire.
  - WAGO 221 lever-nuts — rated 24 AWG min; Dupont (26–28 AWG) too thin
    to seat. (WAGO 243 micro fits but needs cutting.)
  - 2.54mm common-bus PIN board — would work and stays plug-in, but the
    mini breadboard is cheaper/more flexible and we have the room.
- Portability caveat: breadboard rails are light friction tie-points —
  after seating, a dab of hot glue over the inserted wires (NOT solder,
  not on the rail clips) stops them backing out when the rig travels.
  Reversible with a peel.

NOTE on the generated wiring images (wiring/): the labels are verified
against THIS table, but the `wiring/gpio_wiring_diagram.png` v2 image has
one hallucinated label — "GND (Pin 7)". Physical **pin 7 is GPIO4, not a
ground.** Ignore that label and its dangling wire. This table is the
authoritative source, not the image.

### Captive nut pocket dimensions (PETG, 0.28mm gap)

| Nut | Circumscribed dia (Fusion polygon) | Flat-to-flat | Depth |
|---|---|---|---|
| M4 | 8.32mm | 7.56mm | 3.4mm |
| M5 | 9.80mm | 8.56mm | 4.1mm |

- Camera arm friction joints use M5×25mm SHCS + M5 nyloc nuts (M5×20 button heads too short — 5mm deficit at 16.5mm joint span)
- M5 bolt clearance hole: 5.7mm

---

## 7. Software State

### Pi-side stack (validated working)

- Python 3.13.5
- OpenCV 4.13.0
- ONNX Runtime 1.25.1
- Flask 3.1.1

### Code repo

- GitHub: github.com/Sevcav/dice-tracker
- Pi clones from this repo
- ONNX model transferred via SCP (binary, not committed to git)

### Repo layout (current, 2026-06-12)

```
Dice Code/
├── DESIGN.md                # This document — single source of truth
├── HANDOFF.md               # Session hand-off prompt + current priorities
├── README.md
├── SETUP.txt                # Original UX intent (settle→confirm) — still canon
├── requirements.txt
├── dice_tracker.py          # PRODUCTION app: YOLO + supervision stack +
│                            #   settle/confirm UX + DB logging + web thread
├── eval_harness.py          # Ground-truth per-die accuracy measurement
├── db.py                    # SQLite layer (games/rolls, face tallies)
├── webapp.py                # Phone web UI: live control + game record/review
├── dice_types.py            # Shared face vocabularies / type mapping
├── align_camera.py          # Standalone camera-alignment overlay
├── calibrate_tray_roi.py    # (Re)writes tray_roi.json reference corners
├── capture_frames.py        # Frame capture for labeling sessions
├── scan_capture.py          # Pre-upload capture QA (Roboflow ingest checks)
├── tray_roi.json            # Calibrated tray corners (may4 backup alongside)
├── training/                # Local Ultralytics training (RTX 4080)
│   ├── train_all.py         #   block / d6 / d16 / combined configs
│   ├── merge_datasets.py    #   builds the 27-class combined dataset
│   ├── models/              #   ONNX outputs (gitignored — SCP to Pi)
│   ├── datasets/, runs/     #   gitignored artifacts
│   └── live_detect*.py …    #   dev/debug scripts
├── capture_sessions/        # gitignored — raw frames per session
├── retrain_candidates/      # gitignored — reject/miss frames + ground truth
├── eval_sessions/           # gitignored — eval reports
├── archive/legacy_rule_based_detector/   # old pipeline, recoverable
├── Dice Images/             # Reference photos of every die face
└── Stls/                    # Rig CAD files (committed)
```

### Networking

- iPhone hotspot saved as `iphone-hotspot` connection (for game stores)
- Home WiFi (Knickerbocker on Deco mesh) saved separately as `home-knickerbocker`
- NetworkManager (`nmcli`) controls WiFi on Bookworm/Trixie
- mDNS broadcasting needed for `dicetracker.local` access from phone (TODO)
- Pi default IP on iPhone hotspot: 172.20.10.x (Apple subnet)
- Pi default IP on home network: 192.168.68.88 (DHCP, may shift)

### Tested / validated

- ✅ Pi 4 boots, SSH works
- ✅ All Python libraries import cleanly on the 64GB card
- ✅ USB Arducam captures at 1920×1080 on the PC
- ✅ Arducam IR mode engages when photoresistor is darkened
- ✅ iPhone hotspot connection saved (`iphone-hotspot`)
- ✅ Home WiFi connection saved (`home-knickerbocker`, SSID has trailing space)
- ✅ 160 frames captured of block dice in IR mode (`capture_sessions/2026-05-04_201259/`)
- ✅ All 160 frames manually labeled in Roboflow with 5 block-die classes
- ⏳ First Roboflow YOLO model training overnight 2026-05-04 → 2026-05-05

### Next up (priority order, 2026-06-12)

1. **Live eval of the combined model** per dice type
   (`python eval_harness.py --type block --model combined`, then d6/d16)
   — confirms auto mode's real per-die accuracy vs the dedicated models.
2. **Pow-heavy capture session + tray-crop retrain** — pow recall is ~60%;
   proven to be a resolution problem (tray-crop re-reads fixed 4 misreads).
   Build the dataset tray-cropped; fold in `retrain_candidates/` frames
   (they carry ground truth). ≥92% mAP bar applies.
3. **Wiring + GPIO code** for buttons / OLEDs / LEDs (luma.oled, SSD1309
   SPI ×2). OLED rendering = the three-state uncertainty logic already
   proven on the HUD and phone (yellow / orange-? / gray, threshold 0.85).
4. **Pi port**: pure onnxruntime inference path (ultralytics needs torch —
   too heavy for the Pi), mDNS broadcast for `dicetracker.local`, DHCP
   reservation or hotspot config.
5. D8/D3 manual entry flow in the web app.

---

## 8. Open Questions / Deferred Decisions

| Topic | Status |
|---|---|
| Lighting strategy | B0205 IR mode is photoresistor-only (no software control) — Bambu lamps will be the consistent-illumination strategy |
| Camera angle | Set empirically to satisfy "full tray visible + arm clear of rolling area" — near-overhead shallow angle in as-built rig |
| D16 detection | Deferred until block + d6 working end-to-end |
| OLED retention method | Friction / hot glue / M2 screws — TBD when OLEDs in hand |
| Phone web UI scaffold | Not started |
| Database schema | Drafted, not implemented |
| Camera cradle for Arducam | Blocked — need camera in hand for measurements |
| Cable routing strategy | External clips for prototype, internal channels for production |
| Aesthetic direction | Utilitarian for prototype; organic curves for production v2 |
| Power bank shelf attachment method | Snap-fit, screw, or magnetic — TBD |
| **Optional table touchscreen (accessory, not in-case)** | **PLAN RESOLVED 2026-06-14: a STANDALONE screen on a stand at the table, ADDITIVE to the locked design (OLEDs + buttons + phone all stay).** This sidesteps every earlier blocker: no case cutout (it's a separate desktop unit), two-player ergonomics unaffected (it's a shared "scoreboard"; per-player OLEDs still do per-player results), and battery is opt-in (unplug to fall back to phone). Candidate: **ELECROW 7" 1024×600 IPS capacitive, with acrylic stand (~$52.99)** — the "with stand" SKU, not the bare $45.99 one. Connects via 3 ports on the board's left edge: Display→Pi HDMI, Touch→Pi USB, and a DEDICATED Power port → power it from the power bank directly (NOT through the Pi, so it can't brown out the Pi/camera). SOFTWARE = ~free: a boot kiosk launcher opening a browser to `localhost:5000` (the existing web app: live read, dice switching, BB3 roll record). Does NOT touch dice/detection code. Build it AFTER the rig is proven end-to-end with OLEDs+phone. |

---

## 9. Known Lessons / Misconceptions

- **Cradle ≠ production base.** The 200×200 cradle was a standalone bench
  test, not the foundation of the full rig.
- **Pi camera ribbon hardware appears dead** on the current Pi 4 — using USB
  camera path instead. Will not pursue ribbon further.
- **PyTorch is too big for the Pi.** ONNX Runtime is the runtime path; PyTorch
  stays on Windows for training only.
- **Apostrophes and `$` in WiFi credentials** broke `nmcli` — needed single
  quotes in shell, and full key-mgmt declaration.
- **wpa_supplicant.conf is ignored on Bookworm/Trixie** — NetworkManager is
  in charge. All WiFi config goes through `nmcli`.
- **STL external data** — PyTorch ONNX export creates a `.onnx.data` sidecar
  for large models. Both files must be transferred together.
- **Deco mesh routers can block new devices** — single SSID for both bands
  caused initial connectivity issues; ethernet was the most reliable path
  for initial setup.
- **Captive nuts beat heat-set inserts** when inserts haven't shipped yet,
  and captive nuts are stronger anyway (steel vs brass).
- **Print pause at the right layer** is critical for captive nuts — too late
  and the pocket caps over; too early and pocket walls haven't formed yet.
- **OLED PCB has a header strip** at the top — total height includes 14mm
  header zone above the active glass area. Account for this when sizing
  display windows.
- **OLED requires inside-out mounting** — PCB sits on inside of shell wall,
  glass protrudes outward through the through-window. Solder pins on PCB
  back protrude 2mm and need clearance.
- **Bumped lower tier height to 60mm** because OLED at 39.78mm tall plus
  4mm floor + 4mm top wouldn't fit in original 50mm box.
- **PCB recess depth dictates wall thickness** — initially considered 6mm
  walls for outside pocket, but moving to inside-mounting kept walls at 4mm.
- **OLED through-window goes glass-only** — header strip stays inside the
  box, only the glass + active display protrudes through the window cut.
- **Arcade button switches are PCB-mount** — pin legs not spade tabs. Solder direct to COM + NO. NC unused.
- **LEDs have inline resistor** under heatshrink on red lead — no external resistors needed.
- **Female-to-female dupont jumpers still needed** for OLED wiring — pre-crimped JST pigtails are wrong connector type for OLED pin strip.
- **M5 captive nut pocket** — circumscribed 9.80mm, flat-to-flat 8.56mm, depth 4.1mm at 0.28mm gap. Confirmed by test print.
- **Camera arm requires M5×25mm** — M5×20mm button heads are 3.3mm too short for the 16.5mm friction joint. Use M5×25 SHCS instead.
- **B0205 (Arducam UC-A53) IR is photoresistor-controlled only** — no software/UVC control of the IR LEDs or IR-cut filter. The 6 IR LEDs only activate in genuinely dark ambient conditions. Software-controlled IR is **not possible** on this unit. Bambu LED lamps + day-mode is the lighting strategy instead.
- **B0205 spec lists 1m minimum focus** — but the lens is in fact adjustable to focus at ~12 inches once the locking ring is loosened. Confirmed empirically by getting a sharp image at the actual rig working distance.
- **B0205 lens locking ring is partially obstructed by the IR LED ring** — adjustment is possible but tight. Do NOT force it. Loosen with care; replacement lens would require desoldering LEDs.
- **Camera angle is empirical, not a target** — earlier "~35°" figure was wrong for the as-built rig. The functional constraints (full tray + arm clear of rolling area) drive the angle, and what comes out is shallow / near-overhead. Always measure as-built rather than declaring a degree number.
- **Wide-angle lens has heavy barrel distortion + edge softness** — center is sharp, edges warped. Acceptable for CNN training because per-die crops are small relative to the distortion field, but training data must include dice in many tray positions so the CNN sees the full distortion range.
- **35° camera angle would have made side faces visible enough to confuse the CNN** — at the actual shallow angle, top faces dominate each die crop. This was a lucky outcome of letting functional constraints drive the angle instead of picking a number.
- **Rule-based detector failed in IR mode** — `find_tray_roi` required a saturated red tray; `_mask_dice` had a polarity bug that flooded the combined mask with white. Tuning thresholds for a new lighting/camera/distance was a losing fight. Pivoted to a YOLO object detector trained via Roboflow. Old code is in `archive/legacy_rule_based_detector/`, recoverable.
- **YOLO replaces both detection AND classification** — one model outputs labeled bounding boxes per die in a single inference. The previous architecture (rule-based detector → per-die crop → MobileNetV3 classifier) is gone. Simpler runtime, less code, more robust.
- **Tray ROI calibration is still useful but not strictly required** — `calibrate_tray_roi.py` writes a JSON file with the 4 tray corners. YOLO can search the full frame, but cropping to the tray region speeds inference and reduces false positives outside the tray. Currently archived; may be revived later as an inference-time optimization.
- **Roboflow Public/Free workspace is fine for this project** — $60/mo credits, 250k image cap, free fine-tuning of small YOLO variants, free ONNX export. Public Universe sharing is acceptable since the data is dice on a felt tray. **Frames are uploaded full-frame** (background workspace visible at edges); for any future privacy concern, crop to tray ROI before upload.
- **Always check class count BEFORE clicking "Create Version"** — a single typo class (`Pow` vs `pow`, etc.) splits training data across two labels and silently degrades model accuracy. Caught one such typo in the first labeling session via the "Source Images: Classes: 6" mismatch on the Versions wizard.
- **Roboflow defaults are not optimal** — first wizard suggested 100% train / 0% valid / 0% test (no overfitting check), and 512×512 resize (YOLOv8 native is 640×640). Always edit the train/test split and the resize step before clicking Create.
- **Manual labeling 160 frames in one session is feasible** — took the user one sitting. SAM auto-labeling was *not* used because it labels generic objects without class semantics; manual labeling was actually faster end-to-end given that you'd have to re-confirm every box anyway.
- **Single combined YOLO model > separate models per dice type** — block / d6 / d16 will all share one model with combined classes (5 + 6 + 16 = 27 classes when fully expanded). Easier deployment (one ONNX file), single inference call, retrains include all dice types together.
- **IR mode must be VERIFIED, not assumed, even with the light shield printed.**
  Frame analysis (2026-06-11) of the May 13 reject frames: the morning frames
  were day-mode (warm color cast), the afternoon frame matched the IR
  training frames. The camera silently flipped lighting regimes mid-day.
  The models were trained on IR frames only — day-mode frames are
  out-of-distribution and degrade accuracy. Now guarded by a self-check in
  `dice_tracker.py` + `eval_harness.py` (see below).
- **IR check calibration (measured in-tray, 2026-06-11):** mean of |R−G|,
  |G−B|, |R−B| inside the tray ROI: true IR = 4.0-5.2 (chroma noise on the
  tray's high-contrast graphics), day mode = ~10.8-11. Threshold 8.0.
  Measure INSIDE the tray (via `tray_roi.json`, scaled to capture
  resolution) — ambient lamp spill on the floor inflates the global-frame
  number (an IR frame measured 5.4 global with a lamp on, and 25 during the
  transitional state right after plug-in). Auto-exposure takes ~2s to
  settle after camera open and reads high during settling — warm up 2s and
  take the median of 5 probes before judging.
- **The B0205 takes a few seconds to drop into IR mode after plug-in** —
  the first frames out of a freshly connected camera can be full day-mode
  color even in a dim room. Never judge lighting from the first frame.

---

## 10. Decisions Locked

### Hardware geometry
- ✅ Lower tier 260 × 190mm with 55mm front chamfers, 60mm tall, 4mm walls
- ✅ Lid matches lower tier outer footprint, 60mm tall, 4mm walls
- ✅ Tray opening in lid: 160 × 130mm with 2mm × 3-4mm wide perimeter lip
- ✅ Lid mounted to lower tier with 4× Ø10mm posts hanging down + M4 captive nuts + bolts from below
- ✅ Removable dice tray drops into lid pocket (felt-lined)
- ✅ Camera base plate replaces MakerWorld Camera_Base; mounts in 50×50mm rear detent
- ✅ Camera arm: re-use Camera_Link from MakerWorld 627829
- ✅ Power bank external on rear shelf

### Detection
- ✅ Block dice + BB d6 + D16 (single combined YOLO model); D8/D3 manual entry
- ✅ D16 needed for injury rolls (not kickoff)
- ✅ No turn tracking on rig — pure data capture
- ✅ Camera angle: set empirically by functional constraints (full tray visible, arm clear of rolling area) — not a fixed degree target
- ✅ Production rig direction: utilitarian for prototype, organic v2 later
- ✅ **Architecture: YOLO object detector via Roboflow** (replaces former rule-based detector + MobileNetV3 classifier). Trained on labeled IR-mode frames; ONNX-deployed on Pi.
- ✅ Lighting strategy: forced IR mode (photoresistor light shield). No software toggle on B0205, so a 3D-printed shield permanently darkens the photoresistor.
- ✅ Capture/labeling workflow: `capture_frames.py` → upload to Roboflow → manual bounding boxes → ONNX export → SCP to Pi.
- ✅ Train block first, then expand same model to d6, then d16.

### Bench prototype (validation only — not production)
- ✅ Tray cradle 200×200×5mm with 4 corner posts (15×15×50mm)
- ✅ Arm foot bridges back two posts of cradle (230 × 40 × 15mm)
- ✅ Both validated by printing
