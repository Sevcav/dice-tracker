r"""
export_onnx.py
--------------
Exports block_dice_model.pt to block_dice_model.onnx for use with
ONNX Runtime on the Raspberry Pi (no PyTorch required on Pi).

Run on Windows PC:
    cd "C:/Users/chapm/Dice Code"
    python classifier/export_onnx.py
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE, "classifier", "block_dice_model.pt")
ONNX_PATH   = os.path.join(BASE, "classifier", "block_dice_model.onnx")
LABELS_PATH = os.path.join(BASE, "classifier", "labels.json")

def export():
    print(f"Loading checkpoint from: {MODEL_PATH}")
    checkpoint  = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    print(f"Classes ({num_classes}): {class_names}")

    # Rebuild model architecture
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Save labels.json so Pi knows class index → name mapping
    labels = {str(i): name for i, name in enumerate(class_names)}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Labels saved to: {LABELS_PATH}")

    # Dummy input — batch=1, RGB, 128x128
    dummy = torch.zeros(1, 3, 128, 128)

    print(f"Exporting ONNX to: {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )

    print("Verifying export...")
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out  = sess.run(None, {"input": dummy.numpy()})
    print(f"ONNX output shape: {out[0].shape}  ✓")
    print(f"\nExport complete!")
    print(f"  Model : {ONNX_PATH}")
    print(f"  Labels: {LABELS_PATH}")
    print(f"\nCopy both files to the Pi:")
    print("  scp classifier/block_dice_model.onnx classifier/labels.json sevcav@192.168.68.88:/home/sevcav/dice-tracker/classifier/")

if __name__ == "__main__":
    export()
