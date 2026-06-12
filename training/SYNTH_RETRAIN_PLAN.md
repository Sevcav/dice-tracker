# Hand-off: Synthetic training data + POW/D16 fix

*Work-stream hand-off, written 2026-06-12. For general project context
read `HANDOFF.md` and `DESIGN.md` first, and search memory for the
`BB Dice Tracker` entities. This file scopes ONE work stream.*

## Mission

Eliminate manual capture-and-label work by generating training data from
real pixels (copy-paste compositing), retrain the combined 27-class model
tray-cropped, and fix the two measured weaknesses: **POW recall (~60%)**
and **D16** (never live-evaluated; 88.5% val mAP). Implement the D16
adjacency-deduction layer.

## Evidence already established — do not re-derive

- **pow failures are a RESOLUTION problem.** Re-running misread frames
  cropped to the tray ROI (~2x die pixels) correctly identified pow in
  4/10 misread rolls. Frames + ground truth:
  `retrain_candidates/block/eval_miss_*.{jpg,json}`; session reports:
  `eval_sessions/eval_block_20260611_*.json`.
- **Naive crop inference with the current full-frame-trained model breaks
  other classes** — the crop must be trained in, not bolted on.
- Camera is near-overhead: a block/d6 die crop rotated to ANY angle and
  pasted at ANY tray position is physically valid.
- **D16 dice must be composited as WHOLE units** — they are labeled 3
  boxes per die (top + 2 visible sides) and the faces are geometrically
  married. Modest rotations only.
- Combined 27-class model is production (`training/models/combined.onnx`,
  block 92.4 / d6 98.4 / d16 87.4 mAP@50). Datasets:
  `training/datasets/{block,d6,d16}`, merge logic in
  `training/merge_datasets.py` (block ids 0-4, d6 5-10, d16 11-26).
- `tray_roi.json` is saved in 1920x1080 coordinates; frames are 1280x720
  — scale by width ratio (see `color_deviation`/`alignment_check` usage
  in `dice_tracker.py`).

## Deliverables, in order

1. **`training/auto_label_rejects.py`** — convert banked
   `retrain_candidates/*` frames into YOLO labels. The reject/miss JSONs
   carry predicted boxes + (for eval misses) ground-truth labels; for
   non-count-mismatch rolls, boxes sorted left-to-right pair with the
   truth list in order. Skip count-mismatch frames (boxes incomplete) or
   queue them for manual labeling.
2. **`training/synth_dice.py`** — the compositor:
   - harvest labeled die crops from the existing datasets
   - build clean tray backgrounds by inpainting dice out of real frames
     (`cv2.inpaint` with label boxes as mask)
   - paste crops: random rotation (any angle for block/d6; whole-die,
     small-angle for d16), random position incl. wall/corner bias,
     brightness matched to the local background (vignette)
   - weight generation toward pow faces and wall positions
   - emit YOLO labels; build everything **tray-cropped**
3. **Retrain** combined YOLOv11n on real + auto-labeled + synthetic,
   tray-cropped. Recipe (`training/train_all.py`): imgsz=640, batch=16,
   epochs=100, patience=20, device=0, **workers=0 (Windows hard
   requirement)**.
4. **Quality bar (locked):** per-type mean mAP@50 >= 92% on held-out
   validation or the new ONNX does not replace production. Keep the
   current `combined.onnx` until then.
5. **Inference must match training:** when deploying the crop-trained
   model, `dice_tracker.py` and `eval_harness.py` must crop frames to the
   tray ROI before `predict`.
6. **D16 adjacency layer** (new module or `dice_types.py`): lookup
   `(side_A, side_B) -> top`; opposite faces sum to 17; both of the
   user's D16s share one adjacency map (same manufacturer, one white one
   black). Derive the table from the physical dice — **verify with the
   user, do not assume a standard layout**. Use it to (a) deduce the top
   face when top confidence is low, (b) flag geometrically impossible
   face triples as misreads before the player confirms.
7. **Validation at the rig (user time, ~25 rolls per type):**
   - d16 FIRST with the CURRENT model — there is no live d16 baseline
   - then block + d16 with the new model:
     `python eval_harness.py --type block --model <new>`
   - compare against `eval_sessions/` baselines (block per-die 82.6-89.6%).

## Rules

- Camera alignment is always the first step of a live session (built in).
- Train only on IR frames (color deviation < 6; helper in dice_tracker).
- No guessing — every accuracy claim comes from the eval harness.
