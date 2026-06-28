# Session Hand-off — Blood Bowl Dice Tracker

*Last updated: 2026-06-27 (rig is LIVE and reading dice; this session was
field-readiness + UX polish + connectivity + docs. See "⭐ SESSION
2026-06-27" below for the current state and what's open. The older
"LIVE RIG BRING-UP" section remains for deeper background.)*

---

## ⭐ SESSION 2026-06-27 — CURRENT STATE (read this first)

The rig is operational and reading dice live. This session focused on
field readiness: connectivity, UX, the 7" screen, and documentation.

### Connectivity — THREE ways to reach the rig (all working)

| Path | Use when | Address |
|---|---|---|
| Phone over WiFi/hotspot | normal | `http://dicetracker.local:5000` |
| Surface over **Bluetooth PAN** | no WiFi at venue | `http://192.168.44.1:5000` |
| Phone via **Tailscale** | remote admin/SSH anywhere | `100.72.200.96` |

- **Hotspot:** Pi auto-joins the iPhone hotspot ("Ken's iPhone (2)", saved
  profile). The fix when it won't connect: iPhone **Maximize Compatibility
  ON** (forces 2.4GHz the Pi can see) + `sudo nmcli device wifi rescan`
  (plain rescan errors "not authorized" — needs sudo). On the hotspot the
  Pi's IP is `172.20.10.x`; use `dicetracker.local`, not the home IP.
- **Tailscale** (installed this session): mesh VPN, Pi = `100.72.200.96`,
  tailnet `chapman_thor@yahoo.com`. SSH from the phone via Termius (username
  must be lowercase `sevcav`). iOS allows only ONE VPN at a time — toggle
  Tailscale on / regular VPN off to use it. PC can't reach `100.x` unless
  Tailscale is installed on the PC too.
- **Bluetooth PAN** (other session built it, set up + TESTED this session):
  paired KENS-SURFACE (`68:F7:D8:C4:C1:A6`, trusted). Daily recurring step
  on the Surface: Devices & Printers → right-click *dicetracker* →
  **Connect using → Access point**, then browse `192.168.44.1:5000`. Pi side
  is automatic (bt-nap + dnsmasq services). See `deploy/BLUETOOTH_PAN.md`.

### 7" touchscreen (optional add-on, now in use)

ELECROW/Mediatrix MPI7002 over HDMI, runs at 1920×1080 → desktop UI is
microscopic. **Fix: display scale 2× via labwc autostart** —
`~/.config/labwc/autostart` has `wlr-randr --output HDMI-A-1 --scale 2 &`
(survives reboot). View the app on the Pi at `http://127.0.0.1:5000`
(the `http://` is required — Chromium searches otherwise). cups-browsed
auth-popup nuisance was disabled (`systemctl disable --now cups-browsed`).

### Software changes shipped this session (all pushed)

- **P1 OLED fix** — both OLEDs share RST (GPIO25); building the 2nd ssd1309
  with `gpio_RST=25` reset the 1st, blanking P1. Fix: 2nd device uses
  `gpio_RST=None` (in `hardware.py` AND the test scripts). Root cause of
  "Player 1 OLED not working" — was NOT the wiring.
- **OLED confirmed-state** — after a confirm the OLED shows
  "Last roll CONFIRMED / Player X / watching (roll...)" instead of leaving
  the dice up (was confusing).
- **Phone tap-4-corners re-alignment** — `/align` has a "Re-set corners"
  tool: tap the tray's 4 corners (TL,TR,BR,BL) → Save → rewrites
  `tray_roi.json` (overlay quad + axis-aligned crop). Does NOT perspective-
  warp (model only knows the rectangular crop). Alignment is startup-only;
  once live, `/align` shows a "session running" notice (live read is on the
  Live tab — that's expected, not a bug).
- **Games tab** — pinned editable latest roll at top; Started/Export moved
  to bottom; per-game delete + Clear-all (active game protected); a Player
  Names form; and **Load from Game Sheets** (imports coach/team names from
  the league's TourPlay schedule via the Cloudflare Worker proxy —
  `tourplay.py`, default URL pre-filled + auto-loads, see DEFAULT_SHEET_URL).
- **Showcase doc + BOM** — `docs/BB-Dice-Tracker-Showcase.docx` (styled,
  images embedded; regen with `node docs/build_showcase.js`). DESIGN.md §6
  rewritten to the as-built BOM (Dupont+splitters in; lamps/JST/heat-set
  inserts NOT used; 7" screen optional).
- **Repo hygiene** — wiring PNGs (~23MB) removed from git (bloated Pi
  pulls); `docs/` tracked but meant to be skipped on the Pi via
  sparse-checkout (see `deploy/CHEATSHEET.md`).

### OPEN ITEMS / risks (not yet done)

1. **OVERHEATING — real tournament risk.** The Pi thermally shut down once
   this session (sealed case, sustained inference). Recovered on cooldown,
   but a sealed rig WILL overheat across an all-day event. Needs heatsinks /
   a fan / vents. UNADDRESSED.
2. **Sparse-checkout NOT set up on the Pi** — so `git pull` on the rig still
   drags the ~30MB `docs/` folder. Commands are in `deploy/CHEATSHEET.md`;
   run them on the Pi before the next pull.
3. **Day-mode on startup** — rig came up at color-deviation 8.9 (just over
   the 8.0 IR threshold). Reseat the photoresistor lens cap so it forces IR;
   the camera was also clicking (IR-cut filter oscillating at threshold).
4. **Pi password is in chat transcripts** — change with `passwd` when convenient.

---

## Hand-off prompt

You are picking up the **Blood Bowl Dice Tracker** project at
`C:\Users\chapm\Dice Code`. Before doing anything else:

1. Read `DESIGN.md` — single source of truth (header has current status).
2. Read `SETUP.txt` — original settle→confirm UX intent; still canon.
3. Search knowledge-graph memory for: `BB Dice Tracker Project`,
   `BB Dice Tracker Current State 2026-06-12`,
   `BB Dice Tracker Detection Architecture`, `BB Dice Tracker Hardware`,
   `BB Dice Tracker Lessons Learned`, `BB Dice Tracker Networking`.

**Operating rules (locked — do not relitigate):**

- **Camera alignment is ALWAYS the first step of any camera session.**
  Both `dice_tracker.py` and `eval_harness.py` enforce it at startup.
- **Never trust day-mode frames.** Models are trained on IR only; the
  startup self-check + live HUD warning guard this. Verify, don't assume.
- **Retrain quality bar:** mAP@50 ≥ 92% on held-out validation or the new
  ONNX does not replace production.
- **Correction flow:** 4 buttons only, prevention over correction. No
  keyboard/menu correction UI on the rig — late fixes happen in the
  phone web app's game review.
- **No guessing.** Measure with the eval harness / real data, or say so.

## ⭐ LIVE RIG BRING-UP (2026-06-15) — RESUME HERE

The portable Pi rig is wired and the software is RUNNING ON IT headless.
Dice read live off the camera through the full torch-free stack. We are
in the polish/debug phase of the first real run.

**The Pi:** hostname `dicetracker`, user `sevcav`, **IP 192.168.68.85**
(DHCP — may shift; re-find via router or `ping dicetracker.local`).
SSH: `ssh sevcav@192.168.68.85`.

**Bluetooth fallback for the web UI (no-WiFi venues):** when hotspot/cellular
WiFi is unreliable at a store/tournament, the Surface Pro can reach the phone
web UI over **Bluetooth PAN** instead — same Align/Live/Games pages, zero app
code changes. Pi runs a Bluetooth NAP; the Surface joins as a PAN client and
browses the *fixed* URL `http://192.168.44.1:5000/` (static, so it never moves
like the WiFi DHCP IP). One-time setup + pairing + daily-use steps in
`deploy/BLUETOOTH_PAN.md` (configs: `deploy/pan0.sh`, `deploy/dnsmasq-pan.conf`,
`deploy/bt-nap.service`).

**Run the rig — ALWAYS these 3 lines in order, every restart.** The
`. .venv/bin/activate` is MANDATORY; without it you get
`ModuleNotFoundError: No module named 'cv2'`. The venv does NOT persist
across `cd` or a new SSH session, so re-activate every time:
```
cd ~/dice-tracker
. .venv/bin/activate
python dice_tracker.py
```
Headless: it prints the web URL, runs the IR self-check, then waits for
phone alignment. **On the phone (same WiFi):** open the rig URL — the nav
bar has **Align / Live / Games** buttons. Tap **Align**, match tray to the
green outline, **Confirm**, then it goes live. **Games** = the BB3 record
(per-die counts + per-roll fix dropdown).

**To deploy a code change:** PC commits+pushes → on Pi: stop the tracker,
`git pull`, restart, re-align. (Pi pulls from github.com/Sevcav/dice-tracker.)

**To STOP the running tracker:** Ctrl-C (now reliable — installs a SIGINT
handler; prints `[stopping] Ctrl-C received`). If it ever hangs, the
always-works backup is a **second SSH session** → `pkill -f dice_tracker.py`.

**Verified working LIVE on the real Pi:** torch-free backend (`backend OK:
onnx`, 27 classes), all 4 buttons (fire correct names), all 4 LEDs, both
OLEDs (isolation tests `deploy/oled1_test.py`/`oled2_test.py`), USB camera
`/dev/video0`, **live dice reading INCLUDING d16 settling to ONE value.**

**Fixes pushed 2026-06-15 — last few NEED A LIVE CONFIRM on next restart:**
1. **d16 settle** ✅ CONFIRMED LIVE — now settles + shows ONE top value.
   (Was gating on flickering glyph-box count; now gates on DICE/cluster
   count + ≥2 stable faces. `DICE_DEBUG=1` prints `[d16-dbg]` lines if it
   ever won't settle again.)
2. **d16 single value** ✅ — settle display + logged roll + `/games` tally
   show ONE deduced top value per die, not 3 glyph faces. (Recognition was
   always right — rolled 1 → faces {8,2,1} is the correct ring triple.)
3. **Settle speed — FPS-ADAPTIVE (pending live confirm).** Settle felt
   ~2s on the Pi because `COUNT_STABLE_FRAMES=10` is a FRAME count and the
   Pi runs ~5fps vs PC ~30fps. Startup now measures real capture fps and
   sets `settle_frames = max(4, round(SETTLE_SECONDS * fps))`; **tune feel
   via `SETTLE_SECONDS` (=0.5) at top of `dice_tracker.py`.** Startup
   prints `Capture ~X fps -> settle needs N frames` — READ THAT BACK. If
   it settles while a die still wobbles, raise SETTLE_SECONDS to 0.6–0.7.
4. **Ctrl-C stop (pending live confirm)** — SIGINT handler; see "To STOP"
   above.
5. **LED dimming (pending confirm)** — `hardware.py` `LED_BRIGHTNESS=0.2`
   (PWMLED). Edit 0.1–0.35 to taste.

**Open items (after the above):** OLED physical-swap check (which screen
faces which player — both show identical content, doesn't block); case
wire-bulge (cage needs a relief cut or reprint to close — wiring is
heavier than designed); a full ~25-roll live d16 + d6 eval once settle is
solid; the ELECROW 7" touchscreen accessory (standalone, kiosk the web app
at localhost:5000 — DESIGN.md §8, post-bring-up).

**Deploy gotchas seen (so next session doesn't relearn):** `training/
models/` is gitignored so `git pull` won't create it — `mkdir -p` then
`scp combined.onnx` + `combined.onnx.json` from the PC. A stale Pi local
edit may block `git pull` (`git checkout -- <file>` to clear). `setup_pi.sh`
fixed for Trixie (libatlas→libopenblas0). Windows-committed `.sh` may need
`sed -i 's/\r$//'`.

---

**Prior milestone — synthetic-data work stream (2026-06-13):** the tray-crop retrain SHIPPED and the
d16 full-rotation iteration is LIVE-VALIDATED. Production `combined.onnx`
is the full-rotation crop-trained 27-class YOLOv11n (synthetic copy-paste
pipeline in `training/synth_dice.py` et al., full story in
`training/SYNTH_RETRAIN_PLAN.md` STATUS section). Inference crops to the
tray ROI via the `combined.onnx.json` sidecar (pad 8 / pad_bottom 40 —
bottom-wall dice project below the ROI rect). LIVE eval results
(eval_sessions/): **block 66/66 = 100% per-die, pow 18/18** (was 83–90% /
60% pow); **d16 22/26 = 85% value accuracy** (was 12% baseline → 65%
first crop → 85% full-rotation), 0 count-mismatch. d16 eval scores the
ROLLED VALUE via the verified adjacency layer (`d16_geometry.py`, two
consecutive rings 1-8/9-16, deduction + impossible-read warnings live).
Uncertainty thresholds are per-type (`CONF_UNCERTAIN`: block/d6 0.60,
d16 0.80 — crop model confidences run lower than the old model). Backups:
`models/combined_crop_v1_20260612.onnx` (first crop model),
`models/combined_fullframe_backup_20260611.onnx` (pre-crop).

**d16 remaining error mode (know this before iterating):** all live d16
misses are now OFF-BY-ONE RING NEIGHBORS (12→11, 14→13, 1→2) — the model
lands one rotational step from the true top, not wildly wrong. Confidence
is flat (correct 0.82 vs wrong 0.82), so the 0.80 "?" marker won't catch
them and the geometry can't either (an off-by-one neighbor is a legal
face). The weak point is top-vs-side disambiguation, not glyph
recognition. Mitigation in play: nudge-to-re-read; 85% is table-usable.

**Next priorities, in order:**

1. **d16 full-rotation retrain — DONE + LIVE-VALIDATED 2026-06-13.**
   `ROT_D16_MAX=180`, passed val gates (block 93.6 / d6 99.2 / d16 92.7)
   and live eval (85%, see Current state). Deployed. Needed a sliver-box
   fix in `write_yolo_boxes` (a 0px-tall glyph crashed Ultralytics
   augmentation on Windows; `training/_scan_labels.py` finds them).
2. **d6 perspective-tilt retrain — DONE + LIVE-CONFIRMED 2026-06-13.**
   First d6 eval was 89% (both misses 4pip-as-neighbor from tilted dice
   the flat captures never showed). Added perspective-TILT augmentation
   to `synth_dice.py` (block/d6 only; `P_TILT`/`TILT_FRAC`), retrained,
   deployed. Re-eval: **40/40 = 100% over 22 rolls, 4pip 6/6** (was 1/3),
   tilted rolls included. The tilt-augmented model is production
   `combined.onnx`; prior full-rotation model kept as
   `models/combined_crop_fullrot_20260613.onnx`. The synthetic-data work
   stream is COMPLETE — live scoreboard: block 100%, d6 100%, d16 85%,
   pow 60%→100%, zero new dice captured.
3. **GPIO + OLED + Pi port — CODE DONE 2026-06-14, NOW LIVE ON THE PI
   (2026-06-15, see top "LIVE RIG BRING-UP" section — this is the active
   work).** Built:
   - `onnx_backend.py` — torch-free inference (custom YOLOv11 decode,
     agnostic NMS, IoU tracker + smoother). Bit-for-bit parity with the
     ultralytics path on banked frames; full Pi pipeline (detect→track→
     smooth→d16 geometry) verified via `DICE_BACKEND=onnx`.
   - `inference_backend.py` — auto-selects ultralytics (PC) vs onnx (Pi);
     `dice_tracker`/`eval_harness` refactored onto it, PC path unchanged.
   - `hardware.py` — buttons (17/27/22/23) + LEDs (5/6/13/19) + dual
     SSD1309 (SPI0 CE0/CE1) against the DESIGN.md locked map; no-op stub
     off-Pi. Wired into the loop: a player's confirm button = set active
     + confirm (one press); OLEDs mirror the live read.
   - `requirements-pi.txt`, `deploy/setup_pi.sh`,
     `deploy/dice-tracker.service`, `deploy/README.md`.
   - **HEADLESS (the design — sealed box, phone is the only screen):** on
     the Pi dice_tracker runs with NO cv2 window automatically (no
     `$DISPLAY` / onnx backend; `--gui` forces preview, `--headless`
     forces off). Alignment is phone-driven — `webapp` `/align` streams
     the live feed + green tray outline (MJPEG), operator taps Confirm
     (`alignment_check_web`). Reject/Undo/Confirm available from the phone
     (`/api/action`) and the GPIO buttons; both share the synthetic-key
     dispatch. Day-mode prompt auto-proceeds headless (flag stays live).
   First bring-up: follow `deploy/README.md` (git pull → setup_pi.sh →
   scp model → `python dice_tracker.py` → align on phone).
5. When convenient: relabel the 6 bad d16 frames in Roboflow
   (`training/crop_common.py` EXCLUDE_STEMS) + the manual queue
   (`training/datasets/auto_labeled/manual_queue.txt`).

**Gotchas that will bite you** (full list in DESIGN.md §9 + memory):
OpenCV window titles/HUD must be ASCII-only on Windows; `input()` freezes
cv2 windows (use `pumped_input` in eval_harness); supervision must stay
<0.30 until ByteTrack migration; Ultralytics on Windows needs
`workers=0`; lower NMS IoU keeps MORE boxes; DHCP moves IPs — trust
`lan_ip()` printed at startup, not yesterday's address.
