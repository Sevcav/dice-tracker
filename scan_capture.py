"""
scan_capture.py
---------------
Scan a capture_sessions folder for files that might break Roboflow ingest.

Checks each .jpg for:
  - Decodes cleanly with OpenCV
  - Decodes cleanly with PIL (cross-check, catches different bug classes)
  - Expected dimensions (matches the most common size in the folder)
  - Expected mode/channels
  - File size sanity (not zero, not absurdly small)
  - EXIF orientation (causes Roboflow to silently rotate)
  - Color profile / ICC presence

Reports any outliers to stdout.

Usage:
    python scan_capture.py [folder]
    Default folder: capture_sessions/2026-05-04_201259
"""

import os
import sys
from collections import Counter
from pathlib import Path

import cv2

try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except ImportError:
    print("WARN: PIL not installed; skipping cross-check.")
    HAS_PIL = False


DEFAULT_FOLDER = Path(__file__).parent / "capture_sessions" / "2026-05-04_201259"


def scan(folder: Path) -> int:
    if not folder.exists():
        print(f"ERROR: folder does not exist: {folder}")
        return 1

    jpgs = sorted(folder.glob("*.jpg"))
    if not jpgs:
        print(f"No .jpg files in {folder}")
        return 1

    print(f"Scanning {len(jpgs)} files in {folder}")
    print("-" * 70)

    cv_failures      = []
    pil_failures     = []
    size_outliers    = []
    too_small        = []
    exif_rotated     = []
    weird_modes      = []
    has_icc          = []

    sizes = Counter()
    modes = Counter()

    for path in jpgs:
        # File size
        st = path.stat()
        if st.st_size < 1024:
            too_small.append((path.name, st.st_size))
            continue

        # OpenCV decode
        cv_img = cv2.imread(str(path))
        if cv_img is None:
            cv_failures.append(path.name)
            continue
        h, w = cv_img.shape[:2]
        sizes[(w, h)] += 1

        # PIL decode
        if HAS_PIL:
            try:
                with Image.open(path) as pim:
                    pim.load()
                    modes[pim.mode] += 1
                    if pim.mode not in ("RGB", "L"):
                        weird_modes.append((path.name, pim.mode))

                    # EXIF orientation
                    exif = pim._getexif() if hasattr(pim, "_getexif") else None
                    if exif:
                        for tag, val in exif.items():
                            tag_name = ExifTags.TAGS.get(tag, tag)
                            if tag_name == "Orientation" and val not in (1, None):
                                exif_rotated.append((path.name, val))

                    # ICC profile
                    if "icc_profile" in pim.info and pim.info["icc_profile"]:
                        has_icc.append(path.name)
            except Exception as e:
                pil_failures.append((path.name, type(e).__name__, str(e)[:80]))

    # Most common size = canonical
    if sizes:
        canon_size, canon_count = sizes.most_common(1)[0]
        for sz, count in sizes.items():
            if sz != canon_size:
                # find filenames for the outlier size
                for path in jpgs:
                    img = cv2.imread(str(path))
                    if img is not None and (img.shape[1], img.shape[0]) == sz:
                        size_outliers.append((path.name, sz))

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"Total scanned     : {len(jpgs)}")
    print(f"OpenCV failures   : {len(cv_failures)}")
    if HAS_PIL:
        print(f"PIL failures      : {len(pil_failures)}")
    print(f"Size distribution : {dict(sizes)}")
    if HAS_PIL:
        print(f"PIL mode dist     : {dict(modes)}")
    print(f"Files <1 KB       : {len(too_small)}")
    print(f"EXIF rotated      : {len(exif_rotated)}")
    print(f"Has ICC profile   : {len(has_icc)}")
    print(f"Weird color modes : {len(weird_modes)}")
    print(f"Size outliers     : {len(size_outliers)}")

    print()
    flagged = False

    if cv_failures:
        flagged = True
        print("OPENCV FAILURES:")
        for n in cv_failures:
            print(f"  {n}")
        print()

    if pil_failures:
        flagged = True
        print("PIL FAILURES:")
        for n, et, msg in pil_failures:
            print(f"  {n}  [{et}] {msg}")
        print()

    if too_small:
        flagged = True
        print("TINY FILES:")
        for n, sz in too_small:
            print(f"  {n}  {sz} bytes")
        print()

    if exif_rotated:
        flagged = True
        print("EXIF-ROTATED IMAGES (Roboflow may silently rotate, breaking annotations):")
        for n, val in exif_rotated:
            print(f"  {n}  orientation={val}")
        print()

    if weird_modes:
        flagged = True
        print("UNEXPECTED COLOR MODES:")
        for n, m in weird_modes:
            print(f"  {n}  mode={m}")
        print()

    if size_outliers:
        flagged = True
        print(f"SIZE OUTLIERS (canonical = {canon_size}):")
        for n, sz in size_outliers[:20]:
            print(f"  {n}  {sz}")
        if len(size_outliers) > 20:
            print(f"  ... and {len(size_outliers)-20} more")
        print()

    if has_icc:
        # Not a failure, just informational
        print(f"NOTE: {len(has_icc)} files carry an ICC color profile. "
              f"Some ML pipelines strip these silently. Usually harmless.")

    if not flagged:
        print("No anomalies detected. All images decode cleanly, "
              "uniform size, no EXIF rotation, standard RGB mode.")
        return 0
    return 1


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FOLDER
    sys.exit(scan(folder))
