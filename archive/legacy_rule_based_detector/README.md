# Legacy Rule-Based Detector — Archived

This folder contains the original rule-based dice detection and classification
pipeline. It has been **retired** in favour of a YOLO-based object detector
trained via Roboflow.

## Why this was archived

- Detection relied on detecting a saturated red tray colour. In IR mode
  (the chosen production lighting strategy), the tray loses its red colour
  and `find_tray_roi` returns nothing.
- `_mask_dice` had a polarity bug that flooded the combined mask with white,
  collapsing all dice into a single oversized contour. Empirically validated
  in `debug_detect_pipeline.py` against a real IR frame.
- Filter parameters (`MIN_AREA`, `MAX_AREA`, `EDGE_EXCLUSION_PX`, etc.) were
  tuned for the original camera at the original working distance. The new
  Arducam B0205 at 12 in produces dice at a different pixel scale, requiring
  a re-tune of every threshold.
- Maintaining hand-tuned thresholds for each new lighting condition is the
  failure mode this codebase kept hitting. A trained CNN handles variability
  in lighting / glare / shiny dice in a way no fixed threshold can.

## Replacement architecture

- **Detection + classification:** single YOLOv8n model trained on labelled
  frames. Outputs `(class, x, y, w, h, confidence)` per die in one pass.
- **Labelling:** Roboflow web UI with SAM-assisted auto-suggestion.
- **Training:** Roboflow hosted training (free tier) or local Ultralytics
  if needed.
- **Deployment:** export ONNX → drop into `classifier/` → ONNX Runtime on Pi.
- **Tray ROI:** still calibrated once via clicked-corner script (the
  YOLO model itself can search the full frame, but cropping to the tray
  region speeds inference and reduces false positives outside the tray).

## What's in this archive

| File | Purpose |
|---|---|
| `detect_dice.py` | Mask + contours + watershed split (rule-based) |
| `classify_die_type.py` | Heuristic to guess "block die" vs other types |
| `block_dice_classifier.py` | PyTorch MobileNetV3 classifier wrapper |
| `block_dice_classifier_onnx.py` | ONNX Runtime classifier wrapper |
| `export_onnx.py` | PyTorch → ONNX export script |
| `prepare_training_data.py` | Train/val split + augmentation script |
| `train_block_dice.py` | PyTorch training loop |
| `label_live_dice.py` | Live-feed per-die labelling tool |
| `debug_detect_pipeline.py` | Stage-by-stage detector debug dump |
| `debug_detection.py`, `debug_settle.py`, `debug_stability.py` | Various detector debug scripts |
| `test_tray.py`, `test_watershed.py` | Detector unit tests |
| `main.py` | Top-level loop wiring detector + classifier + UI |
| `reading/` | Higher-level "read this die" wrappers |
| `calibrate_tray_roi.py` | Tray-corner calibration tool (still useful — may be revived under YOLO) |
| `tray_roi.json` | Saved 4-corner tray calibration |

## Recovery

If the YOLO approach fails or proves unsuitable, individual files can be
moved back to their original locations. Nothing here was deleted — only
relocated.
