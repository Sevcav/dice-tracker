"""
label_live_dice.py
------------------
Interactive tool to capture live die crops and label them for retraining.
Run this BEFORE retraining to add real in-game images to the training set.

Controls in the LIVE FEED window:
    C       Capture current frame and label each detected die
    D       Delete old small samples (< MIN_KEEP_SIZE px) before capturing
    Q       Quit and show summary

Controls in the DIE CROP window (shown per die):
    1       both_down
    2       pow
    3       push
    4       player_down
    5       stumble
    6       d6_bb_logo
    R       Recapture — reject this crop, try again (don't save)
    S       Skip this die (don't save)

Usage:
    python label_live_dice.py
Then retrain:
    python classifier/prepare_training_data.py
    python classifier/train_block_dice.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection.detect_dice import find_tray_roi, detect_dice

CLASSES = {
    # Block dice
    ord('1'): 'both_down',
    ord('2'): 'pow',
    ord('3'): 'push',
    ord('4'): 'player_down',
    ord('5'): 'stumble',
    ord('6'): 'd6_bb_logo',
    # d6 pip faces — F-row keys to avoid conflicting with R=recapture/S=skip
    ord('f'): 'd6_1',
    ord('g'): 'd6_2',
    ord('h'): 'd6_3',
    ord('j'): 'd6_4',
    ord('k'): 'd6_5',
}

BASE     = os.path.dirname(os.path.abspath(__file__))
TRAIN    = os.path.join(BASE, 'training_data')
LIVE_DIR = os.path.join(BASE, 'live_samples')
os.makedirs(LIVE_DIR, exist_ok=True)

# Samples smaller than this on their shortest side are considered "old/small"
# and will be removed when you press D
MIN_KEEP_SIZE = 60   # px — raise if camera is higher, lower if closer

saved  = {v: 0 for v in CLASSES.values()}
deleted = {v: 0 for v in CLASSES.values()}


def _sample_sizes(cls_name: str) -> dict:
    """Return {filename: (w,h)} for all live_* samples in a class folder."""
    d = os.path.join(TRAIN, cls_name)
    result = {}
    if not os.path.isdir(d):
        return result
    for f in os.listdir(d):
        if not f.startswith('live_'):
            continue
        path = os.path.join(d, f)
        img  = cv2.imread(path)
        if img is not None:
            result[path] = (img.shape[1], img.shape[0])
    return result


def print_sample_summary():
    print("\nCurrent live sample inventory:")
    for cls in CLASSES.values():
        sizes = _sample_sizes(cls)
        small = sum(1 for (w,h) in sizes.values() if min(w,h) < MIN_KEEP_SIZE)
        large = len(sizes) - small
        print(f"  {cls:<16}: {len(sizes):3d} total  "
              f"({large} good >={MIN_KEEP_SIZE}px,  {small} small <{MIN_KEEP_SIZE}px)")


def delete_small_samples():
    """Remove live_* samples smaller than MIN_KEEP_SIZE on shortest side."""
    print(f"\nDeleting live samples smaller than {MIN_KEEP_SIZE}px...")
    total = 0
    for cls in CLASSES.values():
        sizes = _sample_sizes(cls)
        for path, (w, h) in sizes.items():
            if min(w, h) < MIN_KEEP_SIZE:
                os.remove(path)
                deleted[cls] = deleted.get(cls, 0) + 1
                total += 1
    print(f"Deleted {total} small samples.")
    print_sample_summary()


def label_die(crop: np.ndarray, die_num: int) -> str | None:
    """
    Show crop in a window, wait for keypress label.
    Returns class name, 'RECAPTURE', 'QUIT', or None (skip).
    """
    h, w = crop.shape[:2]
    # Scale up so the crop is at least 200px wide for easy viewing
    scale = max(1, 200 // max(h, w))
    big   = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Add size info to window title
    win = f'Die {die_num} ({w}x{h}px) — 1:BD 2:POW 3:Push 4:PD 5:Stbl 6:BB | F:d6-1 G:d6-2 H:d6-3 J:d6-4 K:d6-5 | R:redo S:skip'
    cv2.imshow(win, big)
    cv2.moveWindow(win, 100 + die_num * 220, 100)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in CLASSES:
            cv2.destroyWindow(win)
            return CLASSES[key]
        if key in (ord('r'), ord('R')):
            cv2.destroyWindow(win)
            return 'RECAPTURE'
        if key in (ord('s'), ord('S')):
            cv2.destroyWindow(win)
            return None
        if key in (ord('q'), 27):
            cv2.destroyWindow(win)
            return 'QUIT'


def save_crop(crop: np.ndarray, class_name: str):
    dst = os.path.join(TRAIN, class_name)
    os.makedirs(dst, exist_ok=True)
    # Find next available index
    existing = [f for f in os.listdir(dst) if f.startswith('live_')]
    idx  = len(existing)
    path = os.path.join(dst, f'live_{class_name}_{idx:04d}.jpg')
    cv2.imwrite(path, crop)
    cv2.imwrite(os.path.join(LIVE_DIR, f'{class_name}_{idx:04d}.jpg'), crop)
    saved[class_name] += 1
    h, w = crop.shape[:2]
    print(f'  Saved -> {class_name}  ({w}x{h}px)  '
          f'[total this session: {saved[class_name]}]')


def main():
    print('=' * 60)
    print('  Blood Bowl Die Labeller')
    print('=' * 60)
    print(f'MIN_KEEP_SIZE = {MIN_KEEP_SIZE}px  '
          f'(edit this value if camera height changed)')
    print()
    print('Controls in LIVE FEED:')
    print('  C = capture & label dice')
    print('  D = delete old small samples (< MIN_KEEP_SIZE px)')
    print('  Q = quit')
    print()
    print('Controls in DIE CROP window:')
    print('  Block dice:  1=BD  2=POW  3=Push  4=PD  5=Stbl  6=BB-logo')
    print('  d6 pips:     F=1pip  G=2pip  H=3pip  J=4pip  K=5pip')
    print('  R=recapture (reject & retake)  S=skip')
    print()

    print_sample_summary()
    print()
    print('Roll dice into tray, press C to capture & label.')
    print('Press D first to clear out old small-camera samples.')
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Cannot open camera.')
        return

    cv2.namedWindow('Live Feed — C=capture  D=delete-small  Q=quit', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Feed — C=capture  D=delete-small  Q=quit', 700, 530)

    tray_roi    = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 30 == 1:
            new_roi = find_tray_roi(frame)
            if new_roi:
                tray_roi = new_roi

        dets = detect_dice(frame, roi=tray_roi)

        # Draw preview
        preview = frame.copy()
        if tray_roi:
            rx, ry, rw, rh = tray_roi
            cv2.rectangle(preview, (rx, ry), (rx+rw, ry+rh), (200, 80, 0), 1)
        for i, d in enumerate(dets):
            x, y, w, h = d.bbox
            cv2.rectangle(preview, (x, y), (x+w, y+h), (30, 220, 30), 2)
            cv2.putText(preview, f'Die {i+1} ({w}x{h})', (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 220, 30), 1)

        status = (f'Dice: {len(dets)}  |  '
                  f'Saved: {dict((k,v) for k,v in saved.items() if v>0)}  |  '
                  f'Min size: {MIN_KEEP_SIZE}px')
        cv2.putText(preview, status, (8, preview.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.imshow('Live Feed — C=capture  D=delete-small  Q=quit', preview)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        if key == ord('d'):
            delete_small_samples()

        if key == ord('c') and dets:
            print(f'\nCapturing {len(dets)} dice...')
            quit_requested = False
            i = 0
            while i < len(dets):
                d = dets[i]
                label = label_die(d.crop, i + 1)
                if label == 'QUIT':
                    quit_requested = True
                    break
                if label == 'RECAPTURE':
                    print(f'  Die {i+1} — recapturing, roll again and press C.')
                    # Break out of the labelling loop so user can roll again
                    break
                if label:
                    save_crop(d.crop, label)
                else:
                    print(f'  Die {i+1} skipped.')
                i += 1
            if quit_requested:
                break
            print()

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print('\n' + '=' * 60)
    print('Session complete.')
    print('\nSaved this session:')
    for cls, count in saved.items():
        if count > 0:
            print(f'  {cls}: {count}')
    if deleted:
        print('\nDeleted this session:')
        for cls, count in deleted.items():
            if count > 0:
                print(f'  {cls}: {count} small samples removed')
    print()
    print_sample_summary()
    if any(v > 0 for v in saved.values()):
        print('\nNow retrain with:')
        print('  python classifier/prepare_training_data.py')
        print('  python classifier/train_block_dice.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
