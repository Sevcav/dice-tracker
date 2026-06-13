"""
eval_harness.py
---------------
Measure the REAL per-die accuracy of the dice models against ground truth.

Why: the only accuracy figure to date ("86% confirm rate") came from a
single 7-roll session — its 95% confidence interval is roughly 42-99%,
i.e. we don't actually know how good the models are. This harness produces
a trustworthy per-die accuracy number for each dice type, plus the
confusion pairs that tell us WHAT to fix if it's low.

It runs the exact same detection stack as dice_tracker.py (YOLO ONNX ->
ByteTrack -> DetectionsSmoother -> LabelStabilizer -> settle), so the
number it produces is the number production gets.

Flow per roll:
    1. Roll dice into the tray.
    2. Wait for the settled read (green border + numbered boxes).
    3. The console prints the model's read, dice numbered LEFT -> RIGHT
       (matching the [1] [2] [3] overlays in the window).
    4. Type the ground truth in the console. ENTER alone = model is right.
    5. Clear the tray, roll again. Type q to finish and get the report.

Ground-truth entry (labels separated by space or comma, LEFT -> RIGHT):
    block:  pow/po/w  push/pu/u  both_down/bd/b  player_down/pd  stumble/st/s
    d6:     1 2 3 4 5 6          (6 = the BB logo face)
    d16:    the ROLLED VALUE, one number per die (usually one die, so
            just e.g. "12"). The harness clusters the model's face boxes
            into dice and runs the d16_geometry adjacency layer to get a
            VALUE read — you score the value, not individual face boxes.
            Face-level detail is saved automatically for retraining.
    ENTER   accept the model's read as fully correct (fast path)
    r       discard this roll (cocked die, die against wall, bad roll)
    q       finish session, print + save the report

Output:
    eval_sessions/eval_<type>_<ts>.json   summary + full per-roll record
    eval_sessions/eval_<type>_<ts>.csv    tabular per-roll record
    retrain_candidates/<type>/eval_miss_<ts>.jpg/.json
        every misread frame, with ground truth attached — ready for
        relabeling in Roboflow and retraining (>=92% mAP bar applies).

Usage:
    python eval_harness.py --type block
    python eval_harness.py --type d6
    python eval_harness.py --type d16 --camera 1

Aim for at least 50 rolls per dice type: at ~3 dice/roll that gives a
95% CI of about +/-3-4% on per-die accuracy.
"""

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

import d16_geometry

from dice_tracker import (
    CAMERA_INDEX, CONF_THRESHOLD, COUNT_STABLE_FRAMES, DAY_MODE_DEVIATION,
    MODELS_DIR, RESOLUTION, RETRAIN_DIR, SMOOTHER_LENGTH,
    LabelStabilizer, alignment_check, color_deviation, draw_detections,
    hud_line, load_model_meta, make_tracker, predict_detections,
    tray_crop_rect,
)
from dice_types import TYPE_FACES

ROOT     = Path(__file__).parent
EVAL_DIR = ROOT / "eval_sessions"
EVAL_DIR.mkdir(exist_ok=True)


# ── Console input that keeps the video window alive ────────────────────────
def pumped_input(prompt: str) -> str:
    """input() replacement: pumps the OpenCV GUI event queue between
    keystrokes so the video window doesn't go 'Not Responding' while the
    harness waits for ground truth. Windows-only (msvcrt); falls back to
    plain input() elsewhere."""
    try:
        import msvcrt
    except ImportError:
        return input(prompt)
    print(prompt, end="", flush=True)
    chars: list[str] = []
    while True:
        cv2.waitKey(50)                      # keep the GUI responsive
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                print()
                return "".join(chars)
            if ch in ("\x00", "\xe0"):       # arrow/function key prefix
                msvcrt.getwch()              # swallow the second byte
                continue
            if ch == "\x08":                 # backspace
                if chars:
                    chars.pop()
                    print("\b \b", end="", flush=True)
                continue
            if ch == "\x03":                 # Ctrl-C
                raise KeyboardInterrupt
            chars.append(ch)
            print(ch, end="", flush=True)


# ── Ground-truth parsing ────────────────────────────────────────────────────
def build_alias_map(dice_type: str, class_names: list[str]) -> dict[str, str]:
    """Map user shorthand -> exact model class name."""
    aliases = {name.lower(): name for name in class_names}
    if dice_type == "block":
        extra = {"po": "pow", "w": "pow",
                 "pu": "push", "u": "push",
                 "bd": "both_down", "b": "both_down",
                 "pd": "player_down",
                 "st": "stumble", "s": "stumble"}
        for k, v in extra.items():
            if v in class_names:
                aliases[k] = v
    else:
        # Numeric dice: bare number -> class containing that number
        # ("3" -> "3pip", "6" -> "6BB", "12" -> "D16_12").
        for name in class_names:
            nums = re.findall(r"\d+", name)
            if nums:
                aliases[str(int(nums[-1]))] = name
    return aliases


def parse_truth(raw: str, aliases: dict[str, str]) -> list[str]:
    tokens = raw.replace(",", " ").split()
    out = []
    for t in tokens:
        key = t.strip().lower()
        if key not in aliases:
            raise ValueError(f"unrecognized label {t!r}")
        out.append(aliases[key])
    return out


def prompt_truth(pred: list[str], aliases: dict[str, str]):
    """Returns a list of true labels, or 'discard' / 'quit'."""
    valid = sorted(set(aliases.values()))
    while True:
        raw = pumped_input(
            "  truth (ENTER=all correct, r=discard, q=quit): ").strip()
        low = raw.lower()
        if low == "":
            return list(pred)
        if low == "r":
            return "discard"
        if low == "q":
            return "quit"
        try:
            truth = parse_truth(raw, aliases)
        except ValueError as e:
            print(f"  {e} — valid labels: {valid}")
            continue
        if len(truth) != len(pred):
            ok = pumped_input(f"  model saw {len(pred)} dice, you entered "
                              f"{len(truth)} - record as detection-count "
                              f"error? [Y/n] ").strip().lower()
            if ok == "n":
                continue
        return truth


# ── Reporting ───────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_report(rolls: list[dict], discarded: int, dice_type: str) -> dict:
    paired   = [r for r in rolls if not r["count_mismatch"]]
    mismatch = [r for r in rolls if r["count_mismatch"]]

    n_dice = sum(len(r["correct"]) for r in paired)
    k_dice = sum(sum(r["correct"]) for r in paired)
    lo, hi = wilson_ci(k_dice, n_dice)
    rolls_ok = sum(1 for r in paired if all(r["correct"]))

    confusions = Counter()
    per_class  = defaultdict(lambda: [0, 0])   # truth -> [correct, total]
    conf_correct, conf_wrong = [], []
    for r in paired:
        for p, t, c, ok in zip(r["predicted"], r["truth"],
                               r["confidences"], r["correct"]):
            per_class[t][1] += 1
            if ok:
                per_class[t][0] += 1
                conf_correct.append(c)
            else:
                confusions[(t, p)] += 1
                conf_wrong.append(c)

    day_rolls = sum(1 for r in rolls
                    if r["color_deviation"] >= DAY_MODE_DEVIATION)

    return {
        "dice_type":        dice_type,
        "rolls_scored":     len(rolls),
        "rolls_discarded":  discarded,
        "count_mismatch_rolls": len(mismatch),
        "dice_total":       n_dice,
        "dice_correct":     k_dice,
        "per_die_accuracy": round(k_dice / n_dice, 4) if n_dice else None,
        "per_die_ci95":     [round(lo, 4), round(hi, 4)],
        "roll_level_accuracy": (round(rolls_ok / len(paired), 4)
                                if paired else None),
        "per_class": {t: {"correct": c, "total": n,
                          "recall": round(c / n, 4)}
                      for t, (c, n) in sorted(per_class.items())},
        "confusions": {f"{t} read as {p}": n
                       for (t, p), n in confusions.most_common()},
        "mean_conf_when_correct": (round(sum(conf_correct)
                                         / len(conf_correct), 3)
                                   if conf_correct else None),
        "mean_conf_when_wrong":   (round(sum(conf_wrong)
                                         / len(conf_wrong), 3)
                                   if conf_wrong else None),
        "day_mode_rolls": day_rolls,
    }


def print_report(rep: dict):
    print()
    print("=" * 70)
    print(f"  EVAL REPORT — {rep['dice_type']}")
    print("=" * 70)
    print(f"  Rolls scored: {rep['rolls_scored']} "
          f"(+{rep['rolls_discarded']} discarded)")
    if rep["count_mismatch_rolls"]:
        print(f"  Detection-count errors (missed/extra die): "
              f"{rep['count_mismatch_rolls']} rolls")
    if rep["dice_total"]:
        lo, hi = rep["per_die_ci95"]
        print(f"  PER-DIE accuracy: {rep['dice_correct']}/{rep['dice_total']} "
              f"= {rep['per_die_accuracy']*100:.1f}%  "
              f"(95% CI {lo*100:.1f}-{hi*100:.1f}%)")
        print(f"  Roll-level (all dice right): "
              f"{rep['roll_level_accuracy']*100:.1f}%")
        print()
        print("  Per-class recall:")
        for t, d in rep["per_class"].items():
            print(f"    {t:<14} {d['correct']}/{d['total']} "
                  f"= {d['recall']*100:.0f}%")
        if rep["confusions"]:
            print("  Confusions:")
            for desc, n in rep["confusions"].items():
                print(f"    {desc} x{n}")
        print(f"  Mean confidence when correct: "
              f"{rep['mean_conf_when_correct']}  |  when wrong: "
              f"{rep['mean_conf_when_wrong']}")
    if rep["day_mode_rolls"]:
        print(f"  !! {rep['day_mode_rolls']} rolls captured in DAY mode — "
              f"those reads don't reflect IR accuracy")
    print("=" * 70)


def save_results(rolls: list[dict], rep: dict, dice_type: str,
                 ts_session: str):
    base = EVAL_DIR / f"eval_{dice_type}_{ts_session}"
    base.with_suffix(".json").write_text(
        json.dumps({"report": rep, "rolls": rolls}, indent=2))
    with open(base.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roll_id", "timestamp", "dice_type", "predicted",
                    "truth", "correct", "confidences", "count_mismatch",
                    "color_deviation"])
        for r in rolls:
            w.writerow([
                r["roll_id"], r["timestamp"], r["dice_type"],
                "|".join(r["predicted"]), "|".join(r["truth"]),
                "|".join(str(c) for c in r["correct"]),
                "|".join(str(c) for c in r["confidences"]),
                r["count_mismatch"], r["color_deviation"],
            ])
    print(f"Saved: {base}.json / .csv")


# ── Main loop ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Per-die accuracy evaluation")
    ap.add_argument("--type", choices=["block", "d6", "d16"], default="block")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--model", default=None,
                    help="ONNX stem override (e.g. 'combined' to eval the "
                         "combined model against this dice type's truth)")
    args = ap.parse_args()
    dice_type = args.type
    model_stem = args.model or dice_type

    print("=" * 70)
    print(f"  Dice model evaluation — {dice_type} (model: {model_stem})")
    print("=" * 70)

    model = YOLO(str(MODELS_DIR / f"{model_stem}.onnx"), task="detect")
    model_meta = load_model_meta(model_stem)
    if model_meta.get("tray_crop"):
        print("Model is tray-crop-trained — inference crops to the tray ROI")
    class_names = list(model.names.values())
    # Restrict ground-truth shorthand to this dice type's faces — the
    # combined model carries all 27 classes and bare-number aliases would
    # collide across types (e.g. "1" -> 1pip vs D16_1).
    vocab = ([n for n in class_names if n in TYPE_FACES[dice_type]]
             or class_names)
    aliases = build_alias_map(dice_type, vocab)
    if dice_type == "d16":
        # value mode: truth is the rolled value 1..16, one per die
        aliases = {str(n): str(n) for n in range(1, 17)}
        print("Truth vocabulary: rolled value 1-16, one number per die")
    else:
        print(f"Truth vocabulary: {vocab}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize stale frames after input()
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_rect = tray_crop_rect(actual_w, actual_h)

    # IR-mode self-check (same as dice_tracker): warm up ~2s for
    # auto-exposure, then median of 5 probes.
    t_warm = time.time()
    while time.time() - t_warm < 2.0:
        cap.read()
    devs = []
    for _ in range(5):
        ret, probe = cap.read()
        if ret:
            devs.append(color_deviation(probe))
    if devs:
        dev0 = sorted(devs)[len(devs) // 2]
        is_day = dev0 >= DAY_MODE_DEVIATION
        print(f"Lighting self-check: color deviation {dev0:.1f} -> "
              f"{'DAY mode (!!)' if is_day else 'IR mode'}")
        if is_day:
            print("WARNING: day mode — results will NOT reflect IR accuracy.")
            if input("Continue anyway? [y/N] ").strip().lower() != "y":
                cap.release()
                return

    # Camera alignment pre-flight — always run before scoring rolls.
    if not alignment_check(cap, actual_w, actual_h):
        cap.release()
        print("Aborted at alignment check.")
        return

    tracker    = make_tracker()
    smoother   = sv.DetectionsSmoother(length=SMOOTHER_LENGTH)
    label_stab = LabelStabilizer()

    rolls: list[dict] = []
    discarded = 0
    state = "watching"        # watching -> (settle: console input) -> clearing
    count_hist: deque = deque(maxlen=COUNT_STABLE_FRAMES)
    ts_session = time.strftime("%Y%m%d_%H%M%S")
    quit_requested = False

    # ASCII-only title — non-ASCII creates duplicate ghost windows on Windows
    win = f"Eval {dice_type} - ground truth goes in the CONSOLE"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, RESOLUTION[0], RESOLUTION[1])

    print()
    print("Roll dice. When the read settles, enter ground truth in THIS")
    print("console window. ENTER = model correct. q = finish + report.")
    print()

    while not quit_requested:
        ret, frame = cap.read()
        if not ret:
            continue

        detections = predict_detections(model, frame, model_meta, crop_rect)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        if (detections.tracker_id is not None and len(detections) > 0
                and detections.class_id is not None):
            label_states = label_stab.update(
                detections.tracker_id.tolist(),
                detections.class_id.tolist(),
                detections.confidence.tolist(),
            )
        else:
            label_states = []

        n_det = len(detections)
        if dice_type == "d16":
            # d16 settle gates on STABLE boxes only. A die shows 2-3 small
            # glyph boxes and the marginal one flickers at the conf
            # threshold — requiring the raw count to hold 10 frames means
            # the roll never settles. Transient boxes are ignored (they
            # reset nothing and are excluded from the read); a box that
            # persists long enough to win the label vote joins the stable
            # set, resets the window once, and settles WITH the read.
            # 2-face reads are safe: the adjacency layer deduces the top.
            stable_ids = [i for i, (_c, _cf, s) in enumerate(label_states)
                          if s]
            n_units = (len(d16_geometry.cluster_faces(
                [list(map(float, detections.xyxy[i])) for i in stable_ids]))
                if stable_ids else 0)
            all_stable = len(stable_ids) > 0
            count_hist.append((len(stable_ids), n_units))
        else:
            stable_ids = list(range(n_det))
            all_stable = (n_det > 0
                          and all(s for _, _, s in label_states))
            # Require the detection COUNT to hold steady before settling,
            # so a die flickering in/out at the conf threshold can't be
            # silently dropped from the read (16/50 rolls, first eval).
            count_hist.append(n_det)
        count_stable = (len(count_hist) == COUNT_STABLE_FRAMES
                        and len(set(count_hist)) == 1)

        annotated = draw_detections(frame, detections, model, label_states)
        hud_line(annotated,
                 f"Eval {dice_type}   rolls scored: {len(rolls)}   "
                 f"state: {state}", 25)

        if state == "watching" and all_stable and count_stable:
            # Sort dice left->right so console order matches the overlay.
            # (d16: stable boxes only — transients are not part of the read)
            dice = sorted(
                ((float(detections.xyxy[i][0] + detections.xyxy[i][2]) / 2,
                  float(detections.xyxy[i][1] + detections.xyxy[i][3]) / 2,
                  model.names.get(label_states[i][0],
                                  str(label_states[i][0])),
                  label_states[i][1],
                  [int(v) for v in detections.xyxy[i]])
                 for i in stable_ids),
                key=lambda d: (d[0], d[1]))
            pred  = [d[2] for d in dice]
            confs = [round(d[3], 3) for d in dice]
            boxes = [d[4] for d in dice]
            faces_detail = None
            d16_notes: list[str] = []
            if dice_type == "d16":
                # VALUE scoring: the per-face-box protocol was unusable at
                # the tray (glyph arcs don't run left-right and the rolled
                # VALUE is what the product must get right). Cluster the
                # face boxes into dice, run the verified adjacency layer,
                # and score one value per die. Face-level detail is kept
                # in the record for retraining.
                faces_detail = {"labels": pred, "confidences": confs,
                                "boxes": boxes}
                verdicts = d16_geometry.analyze_roll(
                    pred, [list(map(float, b)) for b in boxes], confs)
                verdicts.sort(key=lambda v: sum(
                    (boxes[i][0] + boxes[i][2]) / 2 for i in v["indices"])
                    / len(v["indices"]))
                die_pred, die_confs, die_boxes = [], [], []
                for v in verdicts:
                    bs = [boxes[i] for i in v["indices"]]
                    die_boxes.append([min(b[0] for b in bs),
                                      min(b[1] for b in bs),
                                      max(b[2] for b in bs),
                                      max(b[3] for b in bs)])
                    die_pred.append(str(v["top"]))
                    die_confs.append(round(max(confs[i]
                                               for i in v["indices"]), 3))
                    fs = ",".join(str(d16_geometry.face_value(
                        faces_detail["labels"][i])) for i in v["indices"])
                    note = f"faces {fs}"
                    if v["status"] == "deduced":
                        note += "; top deduced from sides"
                    elif v["status"] == "impossible":
                        note += "; IMPOSSIBLE combo - likely misread"
                    d16_notes.append(note)
                pred, confs, boxes = die_pred, die_confs, die_boxes
            dev   = color_deviation(frame)

            # Freeze the display before input() blocks the UI loop. Built
            # from the RAW frame — the live view's label bars cover the
            # dice and make the glyphs unreadable exactly when the user
            # needs to read them. Thin boxes + [n] beside each die; the
            # model's read is in the console.
            frozen = frame.copy()
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b
                cv2.rectangle(frozen, (x1, y1), (x2, y2), (0, 255, 0), 1)
                mx = x2 + 6 if x2 + 60 < frozen.shape[1] else x1 - 52
                cv2.putText(frozen, f"[{i+1}]", (mx, y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frozen, (5, 5),
                          (frozen.shape[1] - 5, frozen.shape[0] - 5),
                          (0, 255, 0), 6)
            hud_line(frozen, "SETTLED - enter ground truth in console",
                     50, color=(0, 255, 255))
            cv2.imshow(win, frozen)
            cv2.waitKey(30)
            cv2.waitKey(30)

            if dice_type == "d16":
                print(f"Roll {len(rolls) + 1} - die VALUE read "
                      "(LEFT->RIGHT):  "
                      + "   ".join(f"[{i+1}] {p} ({int(c*100)}%; {n})"
                                   for i, (p, c, n)
                                   in enumerate(zip(pred, confs,
                                                    d16_notes))))
                print("  enter the ROLLED VALUE per die "
                      "(e.g. 12) - not the side faces")
            else:
                print(f"Roll {len(rolls) + 1} - model read (LEFT->RIGHT):  "
                      + "   ".join(f"[{i+1}] {p} ({int(c*100)}%)"
                                   for i, (p, c)
                                   in enumerate(zip(pred, confs))))
            truth = prompt_truth(pred, aliases)

            if truth == "quit":
                quit_requested = True
            elif truth == "discard":
                discarded += 1
                print("  discarded.\n")
            else:
                count_mismatch = len(truth) != len(pred)
                correct = ([] if count_mismatch
                           else [p == t for p, t in zip(pred, truth)])
                roll = {
                    "roll_id":        len(rolls) + 1,
                    "timestamp":      time.time(),
                    "dice_type":      dice_type,
                    "predicted":      pred,
                    "confidences":    confs,
                    "boxes":          boxes,
                    "truth":          truth,
                    "correct":        correct,
                    "count_mismatch": count_mismatch,
                    "color_deviation": round(dev, 2),
                }
                if faces_detail is not None:
                    # d16 value mode: per-glyph detail for retraining; the
                    # top-level boxes are die unions paired with VALUES,
                    # so auto_label_rejects must not pair them with truth
                    roll["truth_mode"] = "value"
                    roll["faces"] = faces_detail
                if count_mismatch or not all(correct):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    miss = RETRAIN_DIR / dice_type / f"eval_miss_{ts}.jpg"
                    cv2.imwrite(str(miss), frame)
                    miss.with_suffix(".json").write_text(
                        json.dumps(roll, indent=2))
                    roll["raw_image_path"] = str(miss)
                    if count_mismatch:
                        print(f"  COUNT ERROR recorded "
                              f"(saw {len(pred)}, actual {len(truth)}) — "
                              f"frame saved.\n")
                    else:
                        wrong = [f"[{i+1}] {t} read as {p}"
                                 for i, (p, t, ok)
                                 in enumerate(zip(pred, truth, correct))
                                 if not ok]
                        print(f"  {sum(correct)}/{len(correct)} correct — "
                              f"{'; '.join(wrong)} — frame saved.\n")
                else:
                    print(f"  all {len(correct)} correct.\n")
                rolls.append(roll)

            state = "clearing"
            if not quit_requested:
                print("  (pick the dice up out of the tray to arm "
                      "the next roll)")

        elif state == "clearing":
            hud_line(annotated, "Clear the tray, then roll again...",
                     50, color=(200, 100, 200))
            if n_det == 0:
                # tray is empty — reset the pipeline for the next roll
                tracker    = make_tracker()
                smoother   = sv.DetectionsSmoother(length=SMOOTHER_LENGTH)
                label_stab.reset()
                count_hist.clear()
                state = "watching"
                print("  (tray clear - roll when ready)")

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            quit_requested = True

    cap.release()
    cv2.destroyAllWindows()

    if not rolls:
        print("No rolls scored — nothing to report.")
        return
    rep = build_report(rolls, discarded, dice_type)
    print_report(rep)
    save_results(rolls, rep, dice_type, ts_session)


if __name__ == "__main__":
    main()
