# Session Hand-off — Blood Bowl Dice Tracker

*Last updated: 2026-06-13 (d16 full-rotation live-validated at 85%; d6
perspective-tilt retrain deployed, live-verify pending). Paste the prompt
below into a new session, or just point the assistant at this file.*

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
2. **d6 live sanity — PROVISIONAL DEPLOY, NEEDS LIVE VERIFY.** First d6
   live eval (2026-06-13, full-rotation model): 16/18 = 89%, both misses
   were 4pip read as off-by-one neighbors. Root cause: captures posed
   FLAT, real rolls land tilted (a tilted 4 reads as 3/5). Fix shipped:
   perspective-TILT augmentation in `synth_dice.py` (block/d6 only;
   `P_TILT`/`TILT_FRAC`). Retrained + DEPLOYED as production
   `combined.onnx` (val 4pip AP 0.93→0.98, no regression; fixed 1 of 2
   banked 4pip misses — the other is a hard occlusion case tilt can't
   cover). **This deploy is provisional: roll ~15 d6 next rig session**
   (`python eval_harness.py --type d6 --model combined`) to confirm 4pip
   improves live. If it doesn't: roll back to
   `models/combined_crop_fullrot_20260613.onnx`, then either capture ~30
   tilted d6 (covers occlusion too) or lower d6's `?` threshold to 0.75.
   Also recalibrate the d6 0.60 threshold from the live data.
3. **GPIO + OLED**: wire 4 arcade buttons, 4 LEDs, 2× SSD1309 SPI OLEDs
   (luma.oled; pins in DESIGN.md §6). OLED rendering = the three-state
   uncertainty logic on HUD/phone (now per-type thresholds).
4. **Pi port**: pure-onnxruntime inference path (ultralytics needs torch —
   too heavy for Pi), mDNS for `dicetracker.local`.
5. When convenient: relabel the 6 bad d16 frames in Roboflow
   (`training/crop_common.py` EXCLUDE_STEMS) + the manual queue
   (`training/datasets/auto_labeled/manual_queue.txt`).

**Gotchas that will bite you** (full list in DESIGN.md §9 + memory):
OpenCV window titles/HUD must be ASCII-only on Windows; `input()` freezes
cv2 windows (use `pumped_input` in eval_harness); supervision must stay
<0.30 until ByteTrack migration; Ultralytics on Windows needs
`workers=0`; lower NMS IoU keeps MORE boxes; DHCP moves IPs — trust
`lan_ip()` printed at startup, not yesterday's address.
