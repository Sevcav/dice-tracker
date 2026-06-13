# Session Hand-off — Blood Bowl Dice Tracker

*Last updated: 2026-06-12 evening (synthetic-retrain work stream landed).
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

**Current state (2026-06-12 evening):** the tray-crop retrain SHIPPED.
Production `combined.onnx` is now the crop-trained 27-class YOLOv11n
(synthetic copy-paste pipeline in `training/synth_dice.py` et al., full
story in `training/SYNTH_RETRAIN_PLAN.md` STATUS section). Inference
crops to the tray ROI via the `combined.onnx.json` sidecar (pad 8 /
pad_bottom 40 — bottom-wall dice project below the ROI rect). LIVE eval
results (eval_sessions/, 2026-06-12): **block 66/66 = 100% per-die, pow
18/18** (was 83–90% / 60% pow); **d16 65% value accuracy vs 12%
baseline** (~88% correct-or-flagged with the verified adjacency layer —
`d16_geometry.py`, two consecutive rings 1-8/9-16, deduction + impossible-
read warnings live). d16 eval scores the ROLLED VALUE now. Uncertainty
thresholds are per-type (`CONF_UNCERTAIN`: block/d6 0.60, d16 0.80 —
the crop model's confidence scale runs lower than the old one). Old
full-frame model kept at `models/combined_fullframe_backup_20260611.onnx`.

**Next priorities, in order:**

1. **d16 full-rotation retrain** (no rig time needed): the d16 capture
   was POSED — glyphs mostly upright — so live rolls show digits at
   angles the model half-knows. Regenerate synth with `ROT_D16_MAX` =
   180 (whole-unit spin is valid near-overhead), retrain, score offline
   against the ~38 value-truth frames in `retrain_candidates/d16/` with
   `training/score_banked_d16.py` BEFORE asking for rig time. Target:
   push d16 live value accuracy from 65% toward block's level.
2. **d6 live sanity** (~10 rolls, first play session is fine):
   `python eval_harness.py --type d6 --model combined` — d6 has never
   been live-evaled on the crop model; also recalibrate its 0.60
   uncertainty threshold from that data.
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
