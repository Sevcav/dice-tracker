# Session Hand-off — Blood Bowl Dice Tracker

*Last updated: 2026-06-12. Paste the prompt below into a new session, or
just point the assistant at this file.*

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

**Current state (2026-06-12):** everything works end-to-end on the PC
bench rig. Combined 27-class YOLOv11n provides auto dice-type detection
(`auto` mode default; per-type models = manual override). Settle-race bug
fixed (COUNT_STABLE_FRAMES=10). Real accuracy numbers: block per-die
~83–87%, pow weakest at ~60% recall (proven to be a resolution problem —
tray-crop fixes it). SQLite DB (`db.py`) + phone web app (`webapp.py`)
deliver live control, the BB3-style end-of-game face record, corrections,
and CSV export. All committed and pushed to github.com/Sevcav/dice-tracker.

**Next priorities, in order:**

1. **Live eval of the combined model**:
   `python eval_harness.py --type block --model combined` (then d6, d16).
   ~25 rolls each. Compare per-die accuracy vs the dedicated-model evals
   in `eval_sessions/`.
2. **Pow-heavy capture + tray-crop retrain**: capture session weighted
   toward pow faces and wall/corner positions; build the dataset cropped
   to the tray ROI; fold in `retrain_candidates/` frames (ground truth
   attached); retrain under the 92% bar.
3. **GPIO + OLED**: wire 4 arcade buttons, 4 LEDs, 2× SSD1309 SPI OLEDs
   (luma.oled; pins in DESIGN.md §6). OLED rendering = the three-state
   uncertainty logic already proven on HUD/phone (threshold 0.85).
4. **Pi port**: pure-onnxruntime inference path (ultralytics needs torch —
   too heavy for Pi), mDNS for `dicetracker.local`.

**Gotchas that will bite you** (full list in DESIGN.md §9 + memory):
OpenCV window titles/HUD must be ASCII-only on Windows; `input()` freezes
cv2 windows (use `pumped_input` in eval_harness); supervision must stay
<0.30 until ByteTrack migration; Ultralytics on Windows needs
`workers=0`; lower NMS IoU keeps MORE boxes; DHCP moves IPs — trust
`lan_ip()` printed at startup, not yesterday's address.
