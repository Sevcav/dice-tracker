"""
crop_common.py
--------------
Shared plumbing for the tray-crop retrain pipeline (synth_dice.py,
auto_label_rejects.py, build_crop_dataset.py).

Established geometry facts (verified 2026-06-12 by ORB feature matching
between raw capture frames and the Roboflow exports — see
SYNTH_RETRAIN_PLAN.md for the work-stream context):

- Raw capture frames are 1280x720 (capture_sessions/<ts>/*.jpg).
- The Roboflow 640x640 export is a PURE CENTER CROP of the raw frame:
      export = raw[40:680, 320:960]        (scale 1.0, no resize)
  so export coords map to raw coords by adding (320, 40).
- Labels are POLYGONS around the top-face symbol/glyph (block: the symbol
  ring, d6: the pip cluster, d16: one circle per visible number glyph,
  3 per die). Ultralytics converts polygons to boxes for detection
  training, so emitted labels must be the polygon's bounding box.
- d6/d16 exports contain Roboflow-augmented copies (3 per frame). Only
  the unaugmented original (pixel-identical to the raw center crop) is
  used here; Ultralytics applies its own augmentation at train time.
- tray_roi.json is stored in 1920x1080 coords; scale by width ratio to
  the working resolution (same convention as dice_tracker.py).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT      = Path(__file__).resolve().parent          # training/
PROJECT   = ROOT.parent                              # Dice Code/
DATASETS  = ROOT / "datasets"
ASSETS    = ROOT / "synth_assets"
CAPTURES  = PROJECT / "capture_sessions"

RAW_W, RAW_H = 1280, 720
EXPORT_OFF_X, EXPORT_OFF_Y = 320, 40   # export = raw[40:680, 320:960]
EXPORT_SIZE = 640

SOURCES = ["block", "d6", "d16"]       # combined-class offset order
                                       # (must match merge_datasets.py)

# D16 frames whose labels are geometrically impossible (found by
# derive_d16_adjacency.py: the glyph triple violates the ring structure
# supported by 450+ correctly-labeled dice — e.g. {5,6,6} where only
# {5,6,7} is physical). ~7 mislabeled glyphs across these 6 frames.
# Excluded from training/val and from synth harvest until relabeled in
# Roboflow.
EXCLUDE_STEMS = {
    "frame_0093_20260505_200328_211",
    "frame_0096_20260505_200358_577",
    "frame_0132_20260505_201108_677",
    "frame_0002_20260506_060405_712",
    "frame_0004_20260506_060422_448",
    "frame_0125_20260506_062108_824",
}

# Train only on IR frames. 8.0 is the production in-tray day-mode
# threshold (dice_tracker.DAY_MODE_DEVIATION, calibrated 2026-06-11:
# true IR sits at 4.0-5.2 in-tray, day mode at ~10.8-11). The plan's
# "< 6" figure predates the in-tray recalibration and would discard the
# whole d6 capture session (median 7.2, clearly not day mode).
IR_MAX_DEVIATION = 8.0


# ── Tray geometry ───────────────────────────────────────────────────────────
def tray_rect() -> tuple[int, int, int, int]:
    """Tray ROI (x, y, w, h) in raw 1280x720 coords."""
    d = json.loads((PROJECT / "tray_roi.json").read_text())
    sx = RAW_W / d["frame_width"]
    sy = RAW_H / d["frame_height"]
    return (int(d["x"] * sx), int(d["y"] * sy),
            int(d["w"] * sx), int(d["h"] * sy))


def tray_quad() -> np.ndarray:
    """Tray corner quad (4x2 float32, TL TR BR BL) in raw coords."""
    d = json.loads((PROJECT / "tray_roi.json").read_text())
    sx = RAW_W / d["frame_width"]
    sy = RAW_H / d["frame_height"]
    return np.array([[cx * sx, cy * sy] for cx, cy in d["corners"]],
                    dtype=np.float32)


def color_deviation_tray(frame: np.ndarray) -> float:
    """IR-mode check on a raw frame: mean abs channel deviation inside the
    tray ROI (same math as dice_tracker.color_deviation)."""
    x, y, w, h = tray_rect()
    small = frame[y:y + h, x:x + w][::4, ::4].astype(np.float32)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return float((np.abs(r - g).mean() + np.abs(g - b).mean()
                  + np.abs(r - b).mean()) / 3)


# ── Combined 27-class vocabulary ────────────────────────────────────────────
def combined_classes() -> tuple[list[str], dict[str, int]]:
    """(names, offsets) for the combined 27-class space, built from the
    source data.yamls exactly like merge_datasets.py so class ids match
    the production combined model."""
    offsets: dict[str, int] = {}
    names: list[str] = []
    for src in SOURCES:
        with open(DATASETS / src / "data.yaml") as f:
            src_names = yaml.safe_load(f)["names"]
        offsets[src] = len(names)
        names.extend(src_names)
    return names, offsets


def source_classes(src: str) -> list[str]:
    with open(DATASETS / src / "data.yaml") as f:
        return yaml.safe_load(f)["names"]


# ── Original-copy index ─────────────────────────────────────────────────────
def _raw_frames_by_stem() -> dict[str, str]:
    return {p.stem: str(p) for p in CAPTURES.glob("*/*.jpg")}


def originals_index(rebuild: bool = False) -> dict:
    """For every (source, split, stem): the dataset image/label pair that is
    the UNAUGMENTED original, plus its raw capture frame. Cached to
    synth_assets/originals_index.json (the scan diffs every export copy
    against the raw center crop, ~1-2 min).

    Returns {src: {split: [{stem, raw, label}, ...]}}.
    """
    ASSETS.mkdir(exist_ok=True)
    cache = ASSETS / "originals_index.json"
    if cache.exists() and not rebuild:
        return json.loads(cache.read_text())

    raw_by_stem = _raw_frames_by_stem()
    index: dict = {}
    for src in SOURCES:
        index[src] = {}
        for split in ["train", "valid", "test"]:
            img_dir = DATASETS / src / split / "images"
            lbl_dir = DATASETS / src / split / "labels"
            groups: dict[str, list[Path]] = {}
            for img in sorted(img_dir.glob("*")):
                stem = img.name.split("_jpg.rf.")[0]
                groups.setdefault(stem, []).append(img)
            entries = []
            for stem, copies in sorted(groups.items()):
                raw_path = raw_by_stem.get(stem)
                if raw_path is None:
                    print(f"  [warn] {src}/{split}/{stem}: no raw frame — skipped")
                    continue
                raw = cv2.imread(raw_path)
                ref = raw[EXPORT_OFF_Y:EXPORT_OFF_Y + EXPORT_SIZE,
                          EXPORT_OFF_X:EXPORT_OFF_X + EXPORT_SIZE]
                best, best_d = None, 1e9
                for c in copies:
                    im = cv2.imread(str(c))
                    d = float(np.abs(im.astype(np.float32)
                                     - ref.astype(np.float32)).mean())
                    if d < best_d:
                        best_d, best = d, c
                if best_d > 6.0:
                    print(f"  [warn] {src}/{split}/{stem}: no original copy "
                          f"(best diff {best_d:.1f}) — skipped")
                    continue
                lbl = lbl_dir / (best.stem + ".txt")
                if not lbl.exists():
                    print(f"  [warn] {src}/{split}/{stem}: missing label — skipped")
                    continue
                entries.append({"stem": stem, "raw": raw_path,
                                "label": str(lbl)})
            index[src][split] = entries
            print(f"  {src}/{split}: {len(entries)} originals")
    cache.write_text(json.dumps(index, indent=1))
    return index


# ── Per-session tray alignment ──────────────────────────────────────────────
# The tray (or camera) sat slightly differently in some capture sessions —
# the May 5-6 d16 sessions are ~15-25px low vs the calibrated tray_roi
# perspective (verified visually 2026-06-12). Frames from those sessions
# are warped into the reference tray space before cropping, so training
# geometry matches deployment. Reference = the eval-session background
# (June 11 frames, captured after camera alignment against tray_roi.json).
ALIGN_REF = "eval_block"


def session_alignments(rebuild: bool = False) -> dict[str, np.ndarray]:
    """{session_name: 2x3 affine} mapping that session's frame coords into
    the reference (tray_roi-calibrated) frame coords. Computed by ECC on
    the median-stack backgrounds over the padded tray window; cached to
    synth_assets/alignments.json. Run synth_dice.build_backgrounds first."""
    cache = ASSETS / "alignments.json"
    if cache.exists() and not rebuild:
        return {k: np.array(v, dtype=np.float32)
                for k, v in json.loads(cache.read_text()).items()}

    bg_dir = ASSETS / "backgrounds"
    bgs = {p.stem[3:]: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
           for p in sorted(bg_dir.glob("bg_*.png"))}
    if ALIGN_REF not in bgs:
        raise RuntimeError(f"reference background bg_{ALIGN_REF}.png missing "
                           "— run synth_dice.py (build_backgrounds) first")

    tx, ty, tw, th = tray_rect()
    pad = 50
    x0, y0 = max(0, tx - pad), max(0, ty - pad)
    x1 = min(RAW_W, tx + tw + pad)
    y1 = min(RAW_H, ty + th + pad)
    ref = bgs[ALIGN_REF][y0:y1, x0:x1].astype(np.float32)

    out: dict[str, np.ndarray] = {}
    for name, g in bgs.items():
        if name == ALIGN_REF:
            out[name] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            continue
        mov = g[y0:y1, x0:x1].astype(np.float32)
        # Two seeds: identity (right when the offset is small) and a
        # phase-correlation shift (rescues big offsets like May 6, but
        # can latch onto a false peak). Keep the seed whose refined ECC
        # correlation is best.
        hann = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
        (sx, sy), _ = cv2.phaseCorrelate(ref, mov, hann)
        best_cc, warp = -1.0, None
        for seed in (np.eye(2, 3, dtype=np.float32),
                     np.array([[1, 0, sx], [0, 1, sy]], dtype=np.float32)):
            try:
                cc, w = cv2.findTransformECC(
                    ref, mov, seed.copy(), cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     300, 1e-7), None, 5)
            except cv2.error:
                continue
            if cc > best_cc:
                best_cc, warp = cc, w
        if warp is None:
            print(f"  [warn] ECC failed for session {name} — using identity")
            out[name] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            continue
        # ECC's warp maps ref-local -> session-local. We want
        # session-full -> ref-full, i.e. the inverse, lifted to full-frame.
        A = warp[:, :2]
        t = warp[:, 2]
        Ainv = np.linalg.inv(A)
        o = np.array([x0, y0], dtype=np.float32)
        # p_ref_local = Ainv @ (p_sess_local - t)
        # full coords: p_sess_local = p_sess_full - o; p_ref_full = p_ref_local + o
        M = np.zeros((2, 3), dtype=np.float32)
        M[:, :2] = Ainv
        M[:, 2] = -Ainv @ t - Ainv @ o + o
        out[name] = M
        shift = M[:, 2] + M[:, :2] @ np.array([RAW_W / 2, RAW_H / 2]) \
            - np.array([RAW_W / 2, RAW_H / 2])
        print(f"  alignment {name}: center shift "
              f"({shift[0]:+.1f}, {shift[1]:+.1f}) px")

    cache.write_text(json.dumps({k: v.tolist() for k, v in out.items()},
                                indent=1))
    return out


def warp_to_ref(frame: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(frame, M, (RAW_W, RAW_H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def warp_pts(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def is_identity(M: np.ndarray) -> bool:
    return bool(np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-3))


# ── Label parsing ───────────────────────────────────────────────────────────
def read_polygons_raw(label_path: str | Path) -> list[tuple[int, np.ndarray]]:
    """Parse a YOLO polygon label file from a 640x640 export and return
    [(class_id, Nx2 float32 polygon)] in RAW 1280x720 pixel coords."""
    out = []
    for line in Path(label_path).read_text().splitlines():
        v = line.split()
        if len(v) < 7:        # class + at least 3 points
            continue
        cls = int(v[0])
        pts = np.array(
            [[float(v[i]) * EXPORT_SIZE + EXPORT_OFF_X,
              float(v[i + 1]) * EXPORT_SIZE + EXPORT_OFF_Y]
             for i in range(1, len(v) - 1, 2)], dtype=np.float32)
        out.append((cls, pts))
    return out


def poly_bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def write_yolo_boxes(path: Path, boxes: list[tuple[int, float, float, float, float]],
                     img_w: int, img_h: int):
    """boxes = [(cls, x1, y1, x2, y2)] in pixel coords of the emitted image.
    Boxes are clamped to the image and sliver boxes (<3px a side, e.g. a
    glyph clipped to nothing at the crop edge) are dropped — degenerate
    instances are a known source of flaky augmentation crashes."""
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        x1 = min(max(x1, 0.0), float(img_w))
        x2 = min(max(x2, 0.0), float(img_w))
        y1 = min(max(y1, 0.0), float(img_h))
        y2 = min(max(y2, 0.0), float(img_h))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
