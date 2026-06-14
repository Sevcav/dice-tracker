# Session Hand-off — Blood Bowl Dice Tracker

*Last updated: 2026-06-13 (synthetic-data work stream COMPLETE: live
block 100% / d6 100% / d16 85%, pow 60%→100%, zero new dice captured).
Paste the prompt below into a new session, or just point the assistant
at this file.*

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

**Current state (2026-06-13):** the tray-crop retrain SHIPPED and the
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
3. **GPIO + OLED + Pi port — CODE DONE 2026-06-14, awaiting first rig
   bring-up.** Built:
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
   **FIRST BRING-UP NEEDS AN HDMI DISPLAY on the Pi** — the alignment
   overlay + button input still go through the OpenCV window. Follow
   `deploy/README.md`. **KNOWN NEXT STEP: fully-headless mode** (phone
   alignment, no cv2 window) for tournament use without a monitor.
5. When convenient: relabel the 6 bad d16 frames in Roboflow
   (`training/crop_common.py` EXCLUDE_STEMS) + the manual queue
   (`training/datasets/auto_labeled/manual_queue.txt`).

**Gotchas that will bite you** (full list in DESIGN.md §9 + memory):
OpenCV window titles/HUD must be ASCII-only on Windows; `input()` freezes
cv2 windows (use `pumped_input` in eval_harness); supervision must stay
<0.30 until ByteTrack migration; Ultralytics on Windows needs
`workers=0`; lower NMS IoU keeps MORE boxes; DHCP moves IPs — trust
`lan_ip()` printed at startup, not yesterday's address.
