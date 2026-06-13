"""
synth_dice.py
-------------
Copy-paste synthetic training data from REAL pixels. No rendering, no
GANs: real die cutouts pasted onto real empty-tray backgrounds, weighted
toward the measured weaknesses (pow faces, wall/corner positions).

Pipeline:
  1. Harvest die cutouts from the TRAIN-split original frames (train
     split only, so nothing from the held-out valid/test frames leaks
     into synthetic training images).
       block/d6: one labeled symbol polygon per die. The die body extends
       well beyond the symbol; the cutout is a feathered disc of
       DIE_RADIUS_FACTOR x symbol radius around the symbol center.
       d16: three glyph polygons per die, geometrically married — the
       die is cut and pasted as a WHOLE unit (small rotations only).
  2. Build clean tray backgrounds by per-pixel MEDIAN over each capture
     session (fixed camera + moving dice -> the median is an empty tray
     with no inpainting artifacts). Day-mode frames are excluded.
  3. Composite: rotation (any angle for block/d6 — near-overhead camera
     makes that physically valid; +/-12 deg for d16 units), placement
     inside the tray quad with wall/corner bias, per-die brightness gain
     matched to the local background, feathered alpha blend.
  4. Emit tray-cropped images + YOLO labels in the combined 27-class id
     space (same ids as the production combined model).

Output:
  training/datasets/synth/images/synth_<type>_<n>.jpg   (tray-cropped)
  training/datasets/synth/labels/synth_<type>_<n>.txt
  training/synth_assets/backgrounds/*.png               (cached step 2)
  training/datasets/synth/qc/*.jpg                      (--qc)

Usage:
    python synth_dice.py                          # default counts
    python synth_dice.py --block 1200 --d6 250 --d16 700 --qc 8
    python synth_dice.py --rebuild-backgrounds
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from crop_common import (
    ASSETS, CAPTURES, DATASETS, EXCLUDE_STEMS, IR_MAX_DEVIATION, PROJECT,
    SOURCES, color_deviation_tray, combined_classes, is_identity,
    originals_index, poly_bbox, read_polygons_raw, session_alignments,
    tray_quad, tray_rect, warp_to_ref, write_yolo_boxes,
)

OUT      = DATASETS / "synth"
BG_DIR   = ASSETS / "backgrounds"

# Search-window radius around the labeled symbol when looking for the die
# body via background subtraction (the die extends well beyond the symbol).
WINDOW_FACTOR = {"block": 2.8, "d6": 2.4}
D16_WINDOW    = 85            # px; D16 body is ~110px across, and the glyph
                              # centroid sits off the die center
PATCH_MARGIN  = 16            # extra pixels of source kept around the mask
DIFF_THRESH   = 12            # |frame - median bg| level that counts as die
AREA_RANGE = {                # plausible die hull areas (px^2) — outside
    "block": (2500, 12000),   # this range the mask grabbed a neighbor or
    "d6":    (2200, 11000),   # lost the die (white-on-white failure)
    "d16":   (2500, 18000),
}

# Compositing distributions
N_DICE_P = {
    "block": ([1, 2, 3],    [0.15, 0.35, 0.50]),
    "d6":    ([1, 2, 3, 4], [0.20, 0.35, 0.30, 0.15]),
    "d16":   ([1, 2],       [0.60, 0.40]),
}
POW_WEIGHT    = 3.0    # pow crops drawn 3x as often as other block faces
P_WALL        = 0.50   # fraction of dice placed in the wall band
P_CORNER      = 0.30   # of wall placements, fraction forced into corners
WALL_BAND_PX  = 35
WALL_INSET    = 25     # tray_roi corners trace the wall RIM; the floor
                       # starts ~this many px further in (sloped walls)
# Full-circle d16 rotation (2026-06-12, second iteration): the capture
# sessions were POSED so glyphs are biased upright, but live rolls land
# digits at any spin — the first crop model misread rotated digits
# (9<->16, 8<->1). Spinning the WHOLE unit is physically valid from the
# near-overhead camera and keeps the three glyph boxes married.
ROT_D16_MAX   = 180.0
SCALE_JITTER  = (0.95, 1.06)
GAIN_CLAMP    = (0.70, 1.40)
CROP_JITTER   = 12     # tray-crop origin jitter, px


# ── Geometry helpers ────────────────────────────────────────────────────────
def edge_distances(quad: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distance from point p to each quad edge (positive inside for a
    clockwise-in-image-coords quad)."""
    d = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        e = b - a
        d.append(float(np.cross(e, p - a)) / (float(np.hypot(*e)) + 1e-9))
    return np.array(d)


def transform_pts(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


# ── Asset harvest ───────────────────────────────────────────────────────────
def harvest_units(index: dict, bgs: dict[str, np.ndarray]
                  ) -> dict[str, list[dict]]:
    """Cut die units from train-split original frames. The die silhouette
    comes from BACKGROUND SUBTRACTION against the session's median-stack
    background (fixed camera -> the diff is exactly the die + its contact
    shadow), so the cutout carries no source-background ring and no
    clipped die edges.

    Returns {type: [unit]} where unit = dict(patch, alpha, R, faces);
    faces = [(combined_cls, polygon-relative-to-patch-origin)]."""
    _, offsets = combined_classes()
    units: dict[str, list[dict]] = {t: [] for t in SOURCES}
    skipped = defaultdict(int)

    for src in SOURCES:
        off = offsets[src]
        for entry in index[src]["train"]:
            if entry["stem"] in EXCLUDE_STEMS:
                skipped[f"{src}: excluded (bad labels)"] += 1
                continue
            session = Path(entry["raw"]).parent.name
            bg = bgs.get(session)
            if bg is None:
                skipped[f"{src}: no background for session"] += 1
                continue
            frame = cv2.imread(entry["raw"])
            if frame is None:
                skipped[f"{src}: unreadable"] += 1
                continue
            if color_deviation_tray(frame) >= IR_MAX_DEVIATION:
                skipped[f"{src}: day-mode frame"] += 1
                continue
            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))
            polys = read_polygons_raw(entry["label"])

            if src in ("block", "d6"):
                centers = [p.mean(axis=0) for _, p in polys]
                radii   = [float(np.hypot(*(p - c).T).mean())
                           for (_, p), c in zip(polys, centers)]
                for i, (cls, pts) in enumerate(polys):
                    c, sym_r = centers[i], radii[i]
                    win_r = WINDOW_FACTOR[src] * sym_r
                    near = [j for j in range(len(polys)) if j != i
                            and np.hypot(*(centers[j] - c))
                            < win_r + radii[j]]
                    if near:
                        skipped[f"{src}: neighbor in window"] += 1
                        continue
                    u = _cut_unit(frame, diff, c, win_r,
                                  AREA_RANGE[src], [(cls + off, pts)])
                    if isinstance(u, str):
                        skipped[f"{src}: {u}"] += 1
                    else:
                        units[src].append(u)
            else:  # d16 — cluster glyphs into whole-die units
                clusters = _cluster_d16(polys)
                cluster_cs = [np.mean([p.mean(axis=0) for _, p in cl],
                                      axis=0) for cl in clusters]
                for ci, cluster in enumerate(clusters):
                    if len(cluster) != 3:
                        skipped[f"d16: cluster of {len(cluster)}"] += 1
                        continue
                    c = cluster_cs[ci]
                    near = [j for j in range(len(clusters)) if j != ci
                            and np.hypot(*(cluster_cs[j] - c))
                            < D16_WINDOW + 55]
                    if near:
                        skipped["d16: neighbor in window"] += 1
                        continue
                    faces = [(cls + off, pts) for cls, pts in cluster]
                    u = _cut_unit(frame, diff, c, D16_WINDOW,
                                  AREA_RANGE["d16"], faces)
                    if isinstance(u, str):
                        skipped[f"d16: {u}"] += 1
                    else:
                        units["d16"].append(u)

    for t in SOURCES:
        per_cls = defaultdict(int)
        for u in units[t]:
            for cls, _ in u["faces"]:
                per_cls[cls] += 1
        print(f"  harvested {t}: {len(units[t])} units "
              f"({len(per_cls)} classes, min/class "
              f"{min(per_cls.values()) if per_cls else 0})")
    if skipped:
        print("  harvest skips:", dict(skipped))
    return units


def _cut_unit(frame, diff, center, win_r, area_range, faces):
    """Extract one die: convex hull of background-subtraction pixels
    within win_r of the symbol center. Returns a unit dict, or a reason
    string when the mask is implausible."""
    H, W = frame.shape[:2]
    fg = (diff > DIFF_THRESH).astype(np.uint8)
    # OPEN first: tray-print edges fire as 1-3px lines in the diff (the
    # paper tray shifts slightly between frames); the die is a thick blob
    # and survives. CLOSE then consolidates the blob.
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          np.ones((7, 7), np.uint8), iterations=2)
    # only sizeable components near the symbol count as die pixels
    n_comp, comp, stats, cents = cv2.connectedComponentsWithStats(fg)
    pts_list = []
    for k in range(1, n_comp):
        if stats[k, cv2.CC_STAT_AREA] < 150:
            continue
        if np.hypot(cents[k][0] - center[0], cents[k][1] - center[1]) >= win_r:
            continue
        ys, xs = np.nonzero(comp == k)
        keep = (np.hypot(xs - center[0], ys - center[1]) < win_r)
        if keep.any():
            pts_list.append(np.stack([xs[keep], ys[keep]], axis=1))
    if not pts_list:
        return "no diff pixels"
    pts = np.concatenate(pts_list).astype(np.int32)
    hull = cv2.convexHull(pts)
    area = float(cv2.contourArea(hull))
    if not (area_range[0] <= area <= area_range[1]):
        return f"hull area {int(area)} out of range"

    hc = hull.reshape(-1, 2).mean(axis=0)          # hull centroid
    R = float(np.max(np.hypot(*(hull.reshape(-1, 2) - hc).T)))
    side = int(2 * (R + PATCH_MARGIN))
    x0 = int(round(hc[0] - side / 2))
    y0 = int(round(hc[1] - side / 2))
    if x0 < 0 or y0 < 0 or x0 + side > W or y0 + side > H:
        return "patch out of frame"

    patch = frame[y0:y0 + side, x0:x0 + side].copy()
    mask = np.zeros((side, side), np.uint8)
    cv2.fillConvexPoly(mask, hull - np.array([x0, y0]), 255)
    # erode: when the die sat on tray graphics, the hull rim carries a halo
    # of source print that would ghost onto the new background
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (9, 9), 2.5)
    rel_faces = [(cls, p - np.array([x0, y0], dtype=np.float32))
                 for cls, p in faces]
    return {"patch": patch, "alpha": alpha, "R": R, "faces": rel_faces}


def _cluster_d16(polys):
    """Union-find glyph polygons into per-die clusters (intra-die glyph
    centers are ~35-45px apart; different dice are further)."""
    centers = [p.mean(axis=0) for _, p in polys]
    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if np.hypot(*(centers[i] - centers[j])) < 55:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, (cls, pts) in enumerate(polys):
        groups[find(i)].append((cls, pts))
    return list(groups.values())


# ── Backgrounds ─────────────────────────────────────────────────────────────
def build_backgrounds(rebuild: bool = False) -> dict[str, np.ndarray]:
    """{group_name: clean tray image} — group name is the capture-session
    folder name (used to pair each harvested frame with ITS background)."""
    BG_DIR.mkdir(parents=True, exist_ok=True)
    cached = sorted(BG_DIR.glob("*.png"))
    if cached and not rebuild:
        return {p.stem[3:]: cv2.imread(str(p)) for p in cached}

    groups: dict[str, list[Path]] = {}
    for d in sorted(CAPTURES.iterdir()):
        if d.is_dir() and list(d.glob("*.jpg")):
            groups[d.name] = sorted(d.glob("*.jpg"))
    # the eval-session bank reflects the CURRENT rig lighting
    bank = sorted((PROJECT / "retrain_candidates" / "block")
                  .glob("eval_miss_*.jpg"))
    if len(bank) >= 8:
        groups["eval_block"] = bank

    bgs: dict[str, np.ndarray] = {}
    for name, files in groups.items():
        # cap the stack; evenly spaced frames see all dice positions
        if len(files) > 80:
            files = files[:: len(files) // 80][:80]
        stack = []
        for f in files:
            im = cv2.imread(str(f))
            if im is None or im.shape[:2] != (720, 1280):
                continue
            if color_deviation_tray(im) >= IR_MAX_DEVIATION:
                continue
            stack.append(im)
        if len(stack) < 8:
            print(f"  background group {name}: only {len(stack)} IR frames "
                  f"— skipped")
            continue
        med = np.median(np.stack(stack), axis=0).astype(np.uint8)
        out = BG_DIR / f"bg_{name}.png"
        cv2.imwrite(str(out), med)
        print(f"  background {out.name}: median of {len(stack)} frames")
        bgs[name] = med
    if not bgs:
        raise RuntimeError("no usable backgrounds")
    return bgs


# ── Compositing ─────────────────────────────────────────────────────────────
def place_dice(rng: random.Random, quad: np.ndarray, dice_R: list[float]):
    """Choose centers for each die: inside the tray quad, wall/corner
    biased, non-overlapping. Returns one entry per requested die: the
    center as np.array, or None when placement kept colliding."""
    qx0, qy0 = quad.min(axis=0)
    qx1, qy1 = quad.max(axis=0)
    placed: list[tuple[np.ndarray, float]] = []
    spots: list[np.ndarray | None] = []
    for R in dice_R:
        want_wall = rng.random() < P_WALL
        want_corner = want_wall and rng.random() < P_CORNER
        ok = None
        base = R * 0.85 + WALL_INSET
        for _ in range(300):
            p = np.array([rng.uniform(qx0, qx1), rng.uniform(qy0, qy1)])
            d = edge_distances(quad, p)
            if d.min() < base:              # die body must sit on the floor
                continue
            near = np.sort(d)[:2]           # distance to 2 nearest walls
            if want_corner:
                if not (near[0] <= base + WALL_BAND_PX
                        and near[1] <= base + WALL_BAND_PX):
                    continue
            elif want_wall:
                if not near[0] <= base + WALL_BAND_PX:
                    continue
            if all(np.hypot(*(p - q)) >= (R + R2) * 0.92
                   for q, R2 in placed):
                ok = p
                break
        if ok is not None:
            placed.append((ok, R))
        spots.append(ok)
    return spots


def paste_unit(canvas: np.ndarray, unit: dict, target: np.ndarray,
               angle: float, scale: float):
    """Rotate/scale the unit patch + its silhouette alpha, brightness-match
    the die to the local background, alpha-blend onto the canvas.
    Returns [(cls, x1, y1, x2, y2)] face boxes in canvas coords."""
    patch = unit["patch"]
    side  = patch.shape[0]
    pc    = (side / 2.0, side / 2.0)

    M = cv2.getRotationMatrix2D(pc, angle, scale)
    warped = cv2.warpAffine(patch, M, (side, side), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(unit["alpha"], M, (side, side),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    x0 = int(round(target[0] - side / 2))
    y0 = int(round(target[1] - side / 2))
    H, W = canvas.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + side > W or y0 + side > H:
        return None    # placement guard should prevent this
    region = canvas[y0:y0 + side, x0:x0 + side]

    # brightness gain: compare the SOURCE background ring just outside the
    # silhouette with the same ring on the target — compensates lighting
    # drift between sessions / vignette across the tray.
    a_bin = (alpha > 0.5).astype(np.uint8)
    ring = (cv2.dilate(a_bin, np.ones((25, 25), np.uint8)) > 0) & (alpha < 0.05)
    if ring.sum() < 50:
        return None
    src_med = float(np.median(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)[ring]))
    dst_med = float(np.median(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)[ring]))
    gain = np.clip(dst_med / max(src_med, 1.0), *GAIN_CLAMP)
    lit = np.clip(warped.astype(np.float32) * gain, 0, 255)

    a = alpha[..., None]
    canvas[y0:y0 + side, x0:x0 + side] = \
        (a * lit + (1 - a) * region.astype(np.float32)).astype(np.uint8)

    boxes = []
    for cls, pts in unit["faces"]:
        tp = transform_pts(M, pts) + np.array([x0, y0])
        boxes.append((cls, *poly_bbox(tp)))
    return boxes


def synthesize(units, bgs, counts: dict[str, int], qc: int,
               rng: random.Random):
    names, _ = combined_classes()
    pow_id = names.index("pow")

    img_out = OUT / "images"
    lbl_out = OUT / "labels"
    qc_out  = OUT / "qc"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    if qc:
        qc_out.mkdir(exist_ok=True)

    quad = tray_quad()
    tx, ty, tw, th = tray_rect()
    cls_counts = defaultdict(int)

    # paste canvases live in the calibrated tray space — warp any session
    # whose tray sat off the reference position (May 5-6 d16 sessions)
    alignments = session_alignments()
    bg_list = [bg if is_identity(alignments.get(name, np.eye(2, 3)))
               else warp_to_ref(bg, alignments[name])
               for name, bg in bgs.items()]
    for dtype, n_images in counts.items():
        pool = units[dtype]
        if not pool:
            print(f"  !! no units for {dtype} — skipped")
            continue
        weights = None
        if dtype == "block":
            weights = [POW_WEIGHT if any(c == pow_id for c, _ in u["faces"])
                       else 1.0 for u in pool]
        rot_max = ROT_D16_MAX if dtype == "d16" else 180.0

        made = 0
        for i in range(n_images):
            canvas = bg_list[rng.randrange(len(bg_list))].copy()
            n_dice = rng.choices(*N_DICE_P[dtype])[0]
            chosen = rng.choices(pool, weights=weights, k=n_dice)
            scales = [rng.uniform(*SCALE_JITTER) for _ in chosen]
            spots  = place_dice(rng, quad,
                                [u["R"] * s for u, s in zip(chosen, scales)])
            boxes = []
            for p, u, s in zip(spots, chosen, scales):
                if p is None:
                    continue
                angle = rng.uniform(-rot_max, rot_max)
                b = paste_unit(canvas, u, p, angle, s)
                if b:
                    boxes.extend(b)
            if not boxes:
                continue

            # jittered tray crop
            jx = tx + rng.randint(-CROP_JITTER, CROP_JITTER)
            jy = ty + rng.randint(-CROP_JITTER, CROP_JITTER)
            jx = max(0, min(jx, canvas.shape[1] - tw))
            jy = max(0, min(jy, canvas.shape[0] - th))
            crop = canvas[jy:jy + th, jx:jx + tw]
            out_boxes = [(c, x1 - jx, y1 - jy, x2 - jx, y2 - jy)
                         for c, x1, y1, x2, y2 in boxes]

            stem = f"synth_{dtype}_{i:05d}"
            cv2.imwrite(str(img_out / f"{stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo_boxes(lbl_out / f"{stem}.txt", out_boxes, tw, th)
            for c, *_ in out_boxes:
                cls_counts[names[c]] += 1
            made += 1

            if qc and i < qc:
                vis = crop.copy()
                for c, x1, y1, x2, y2 in out_boxes:
                    cv2.rectangle(vis, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 1)
                    cv2.putText(vis, names[c], (int(x1), int(y1) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 255), 1)
                cv2.imwrite(str(qc_out / f"{stem}.jpg"), vis)
        print(f"  synthesized {dtype}: {made}/{n_images} images")

    print("  per-class face counts:")
    for n in names:
        if cls_counts[n]:
            print(f"    {n:<14} {cls_counts[n]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=1200)
    ap.add_argument("--d6",    type=int, default=250)
    ap.add_argument("--d16",   type=int, default=700)
    ap.add_argument("--qc",    type=int, default=0,
                    help="write the first N composites per type with "
                         "label overlays")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--rebuild-backgrounds", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("Backgrounds (per-session median stacks):")
    bgs = build_backgrounds(rebuild=args.rebuild_backgrounds)
    print(f"  {len(bgs)} backgrounds ready")

    print("Harvesting die units from train-split originals:")
    index = originals_index()
    units = harvest_units(index, bgs)

    print("Compositing:")
    counts = {"block": args.block, "d6": args.d6, "d16": args.d16}
    synthesize(units, bgs, counts, args.qc, rng)
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
