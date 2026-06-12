"""
prepare_training_data.py
------------------------
Crops the reference face images from /Dice Images and saves them into
training_data/<class>/ folders ready for train_block_dice.py.

Run once before training:
    python classifier/prepare_training_data.py
"""

import os
import shutil
import cv2
import numpy as np

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC         = os.path.join(BASE, "Dice Images")
TRAIN_DIR   = os.path.join(BASE, "training_data")

# Map source filenames  ->  class folder
# We use every angle shot to maximise training variety
FILE_CLASS_MAP = {
    # Block dice faces (cream/tan coloured die)
    "Both Down Face.jpg":      "both_down",
    "POW Face.jpg":            "pow",
    "POW Angle 1.jpg":         "pow",
    "POW Angle 2.jpg":         "pow",
    "POW Angle 3.jpg":         "pow",
    "POW Angle 4.jpg":         "pow",
    "Push Face.jpg":           "push",
    "Push Angle 1.jpg":        "push",
    "Push Angle 2.jpg":        "push",
    "Push Angle 3.jpg":        "push",
    "Push Angle 4.jpg":        "push",
    "Player Down Face.jpg":    "player_down",
    "Player Down Angle 1.jpg": "player_down",
    "Player Down Angle 2.jpg": "player_down",
    "Player Down Angle 3.jpg": "player_down",
    "Player Down Angle 4.jpg": "player_down",
    "Stumble Face.jpg":        "stumble",
    "Stumble Angle 1.jpg":     "stumble",
    "Stumble Angle 2.jpg":     "stumble",
    "Stumble Angle 3.jpg":     "stumble",
    "Stumble Angel 4.jpg":     "stumble",   # note: typo in original filename
    # d6 Blood Bowl logo face (black die)
    "6 Sided Top.jpg":         "d6_bb_logo",
    "6 sided angle 3.jpg":     "d6_bb_logo",
    "6 Sided angle 1.jpg":     "d6_bb_logo",
    "6 sidesd angle 2.jpg":    "d6_bb_logo",  # note: typo in original filename
    "6 sided angle 4.jpg":     "d6_bb_logo",
}

TARGET_SIZE = (128, 128)


def crop_die_face(img: np.ndarray) -> np.ndarray:
    """
    Attempt to isolate the die face from the background.
    Falls back to a centre crop if detection fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold: die is either very dark (black d6) or light (cream block dice)
    _, thresh_dark  = cv2.threshold(blurred, 80,  255, cv2.THRESH_BINARY_INV)
    _, thresh_light = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    best_box = None
    best_area = 0

    for thresh in [thresh_dark, thresh_light]:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h
            if 0.5 < aspect < 2.0 and area > best_area:
                best_area = area
                best_box = (x, y, w, h)

    if best_box:
        x, y, w, h = best_box
        # add 10% padding
        pad = int(max(w, h) * 0.10)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        crop = img[y:y+h, x:x+w]
    else:
        # fallback: centre 60% crop
        h, w = img.shape[:2]
        m = 0.20
        crop = img[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]

    return cv2.resize(crop, TARGET_SIZE)


def augment(img: np.ndarray):
    """
    Yield original + rotations + flips + RED TINT versions.
    The red tint simulates how the dice look under the red NAF tray lighting,
    which is the actual camera environment they'll be classified in.
    """
    variants = [img]
    # Rotations
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (128, 128)))
    # Flips
    variants.append(cv2.flip(img, 1))
    variants.append(cv2.flip(img, 0))

    # Red tint augmentations — simulate the NAF tray lighting environment
    # The tray casts a strong red/pink colour cast on the cream dice
    for v in list(variants):
        tinted = v.astype(np.float32).copy()
        tinted[:, :, 2] = np.clip(tinted[:, :, 2] * 1.35, 0, 255)  # boost red
        tinted[:, :, 1] = np.clip(tinted[:, :, 1] * 0.80, 0, 255)  # reduce green
        tinted[:, :, 0] = np.clip(tinted[:, :, 0] * 0.75, 0, 255)  # reduce blue
        variants.append(tinted.astype(np.uint8))

    for v in variants:
        yield v


def main():
    copied = 0
    skipped = 0

    for filename, class_name in FILE_CLASS_MAP.items():
        src_path = os.path.join(SRC, filename)
        if not os.path.exists(src_path):
            print(f"  [MISSING] {filename}")
            skipped += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"  [UNREADABLE] {filename}")
            skipped += 1
            continue

        face = crop_die_face(img)
        dst_dir = os.path.join(TRAIN_DIR, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        base_name = os.path.splitext(filename)[0]
        for i, aug in enumerate(augment(face)):
            out_path = os.path.join(dst_dir, f"{base_name}_aug{i}.jpg")
            cv2.imwrite(out_path, aug)
            copied += 1

    print(f"\nDone. {copied} images written, {skipped} skipped.")
    print(f"Training data saved to: {TRAIN_DIR}")


if __name__ == "__main__":
    main()
