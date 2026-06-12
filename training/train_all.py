"""
train_all.py
------------
Train YOLOv11 Nano models for block, d6, and d16 dice locally.

Each model trains independently against its own labeled dataset.
After training, each model is exported to ONNX for Pi deployment.

Outputs:
    runs/block/<run_name>/weights/best.pt           PyTorch checkpoint
    runs/block/<run_name>/weights/best.onnx         ONNX export
    models/block.onnx                                Final ONNX (copy of best.onnx)
    (same for d6 and d16)

Default settings chosen to mirror the successful Roboflow runs:
    Model:        yolo11n (Nano)
    Image size:   640
    Batch size:   16 (RTX 4080 can handle this easily)
    Epochs:       100 (Ultralytics has built-in early stopping)
    Optimizer:    auto (Ultralytics default)
    Patience:     20 (stop if no improvement for 20 epochs)

Usage:
    python train_all.py                # train all three
    python train_all.py block          # train only block
    python train_all.py d6 d16         # train d6 and d16
"""

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# --- Paths ---
ROOT       = Path(__file__).parent
DATASETS   = ROOT / "datasets"
RUNS       = ROOT / "runs"
MODELS_OUT = ROOT / "models"
MODELS_OUT.mkdir(exist_ok=True)

# Pretrained checkpoint — YOLOv11 nano with COCO weights
# Ultralytics auto-downloads on first use
PRETRAINED = "yolo11n.pt"

# Training settings
TRAIN_KWARGS = dict(
    imgsz=640,
    batch=16,
    epochs=100,
    patience=20,        # early stopping
    optimizer="auto",
    device=0,           # GPU 0
    workers=0,          # Windows: avoid DLL init crashes in worker subprocesses
    verbose=True,
    plots=True,
)

# Dataset configurations
DATASET_CONFIGS = {
    "block": dict(
        data=DATASETS / "block" / "data.yaml",
        name="block",
    ),
    "d6": dict(
        data=DATASETS / "d6" / "data.yaml",
        name="d6",
    ),
    "d16": dict(
        data=DATASETS / "d16" / "data.yaml",
        name="d16",
    ),
    # 27-class merged dataset (run merge_datasets.py first) — one model
    # for all dice types, removes manual type switching on the rig.
    "combined": dict(
        data=DATASETS / "combined" / "data.yaml",
        name="combined",
    ),
}


def train_one(key: str):
    cfg = DATASET_CONFIGS[key]
    data_yaml = cfg["data"]
    run_name  = cfg["name"]

    print()
    print("=" * 70)
    print(f"  Training: {key}")
    print(f"  Dataset:  {data_yaml}")
    print("=" * 70)
    print()

    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        return

    # Fresh model from pretrained checkpoint
    model = YOLO(PRETRAINED)

    # Train
    results = model.train(
        data=str(data_yaml),
        project=str(RUNS),
        name=run_name,
        exist_ok=False,    # creates run_name, run_name2, etc. if collision
        **TRAIN_KWARGS,
    )

    # Best weights path
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"WARN: best.pt not found at {best_pt}")
        return

    print()
    print(f"Training done. best.pt = {best_pt}")
    print()

    # Export to ONNX
    print(f"Exporting {key} to ONNX...")
    model = YOLO(str(best_pt))
    onnx_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
    )
    print(f"ONNX export: {onnx_path}")

    # Copy ONNX to models/ for easy SCP later
    out = MODELS_OUT / f"{key}.onnx"
    shutil.copy2(onnx_path, out)
    print(f"Copied to {out}")
    print()


def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(DATASET_CONFIGS.keys())

    for k in keys:
        if k not in DATASET_CONFIGS:
            print(f"Unknown dataset: {k}. Choices: {list(DATASET_CONFIGS.keys())}")
            continue
        train_one(k)

    print()
    print("=" * 70)
    print("  All training complete.")
    print("=" * 70)
    print()
    print("Models in:", MODELS_OUT)
    for p in sorted(MODELS_OUT.glob("*.onnx")):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:20s} {size_mb:5.1f} MB")


if __name__ == "__main__":
    main()
