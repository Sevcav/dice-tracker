"""
merge_datasets.py
-----------------
Merge the block / d6 / d16 YOLO datasets into one combined 27-class
dataset so a SINGLE model can read any dice type — which eliminates the
manual dice-type switching on the rig (the model's class labels identify
the dice type for free, at no extra inference cost on the Pi).

Class id mapping (each source keeps its internal order, offset applied):
    block  0-4   -> 0-4     (both_down, player_down, pow, push, stumble)
    d6     0-5   -> 5-10    (1pip..5pip, 6BB)
    d16    0-15  -> 11-26   (D16_1, D16_10, ... Roboflow's lexical order)

Output: training/datasets/combined/{train,valid,test}/{images,labels}
        + data.yaml with absolute paths (Windows/Ultralytics gotcha).

Filenames get a source prefix (block_/d6_/d16_) to avoid collisions.
Safe to re-run: wipes and rebuilds the combined folder.
"""

import shutil
from pathlib import Path

import yaml

ROOT     = Path(__file__).parent
DATASETS = ROOT / "datasets"
OUT      = DATASETS / "combined"

SOURCES = ["block", "d6", "d16"]
SPLITS  = ["train", "valid", "test"]


def main():
    # Build the combined class list from the source data.yamls
    offsets: dict[str, int] = {}
    combined_names: list[str] = []
    for src in SOURCES:
        with open(DATASETS / src / "data.yaml") as f:
            names = yaml.safe_load(f)["names"]
        offsets[src] = len(combined_names)
        combined_names.extend(names)
    print(f"Combined classes ({len(combined_names)}):")
    for src in SOURCES:
        print(f"  {src}: offset {offsets[src]}")

    if OUT.exists():
        shutil.rmtree(OUT)

    counts: dict[str, int] = {}
    for split in SPLITS:
        img_out = OUT / split / "images"
        lbl_out = OUT / split / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        n_split = 0
        for src in SOURCES:
            off = offsets[src]
            img_dir = DATASETS / src / split / "images"
            lbl_dir = DATASETS / src / split / "labels"
            for img in sorted(img_dir.glob("*")):
                lbl = lbl_dir / (img.stem + ".txt")
                shutil.copy2(img, img_out / f"{src}_{img.name}")
                # remap class ids by offset
                lines_out = []
                if lbl.exists():
                    for line in lbl.read_text().splitlines():
                        parts = line.split()
                        if not parts:
                            continue
                        parts[0] = str(int(parts[0]) + off)
                        lines_out.append(" ".join(parts))
                (lbl_out / f"{src}_{img.stem}.txt").write_text(
                    "\n".join(lines_out) + ("\n" if lines_out else ""))
                n_split += 1
        counts[split] = n_split
        print(f"  {split}: {n_split} images")

    data_yaml = {
        "train": str(OUT / "train" / "images").replace("\\", "/"),
        "val":   str(OUT / "valid" / "images").replace("\\", "/"),
        "test":  str(OUT / "test" / "images").replace("\\", "/"),
        "nc":    len(combined_names),
        "names": combined_names,
    }
    with open(OUT / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"\nWrote {OUT / 'data.yaml'}")
    print(f"Total: {sum(counts.values())} images, {len(combined_names)} classes")


if __name__ == "__main__":
    main()
