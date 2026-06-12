"""
train_block_dice.py
-------------------
Trains a lightweight CNN to classify Blood Bowl block dice faces.

Classes (6 total):
    0  both_down
    1  pow
    2  push
    3  player_down
    4  stumble
    5  d6_bb_logo   (the BB-logo face on the black d6)

Usage:
    python classifier/train_block_dice.py

Output:
    classifier/block_dice_model.pt   (saved model weights)
    classifier/labels.json           (class index -> name)
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image


def clahe_loader(path: str) -> Image.Image:
    """
    Load image and apply CLAHE normalisation before training.
    Matches the preprocessing applied at inference time in block_dice_classifier.py.
    """
    bgr  = cv2.imread(path)
    if bgr is None:
        return Image.open(path).convert("RGB")
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    rgb_eq = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_eq)

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR  = os.path.join(BASE, "training_data")
MODEL_OUT  = os.path.join(BASE, "classifier", "block_dice_model.pt")
LABELS_OUT = os.path.join(BASE, "classifier", "labels.json")

IMG_SIZE   = 128
BATCH_SIZE = 8          # small batch — we have few images
EPOCHS     = 60         # more epochs now we have more augmented data
LR         = 0.0005
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data transforms ────────────────────────────────────────────────────────────

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),           # dice can land at any angle
    # Aggressive lighting augmentation — simulates store windows, sunlight
    # shifts, overhead fluorescent, shadows from hands/objects
    transforms.ColorJitter(
        brightness=0.7,   # very bright (sunlight) to quite dark (shadow)
        contrast=0.7,     # washed-out to high contrast
        saturation=0.8,   # colour cast from warm/cool light sources
        hue=0.15,         # slight hue shift (different dice colours)
    ),
    transforms.RandomGrayscale(p=0.15),      # occasionally train on grey
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """
    MobileNetV3-Small pretrained on ImageNet, with the classifier head
    replaced for our dice classes.  Fast, small, accurate on few samples.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    print(f"Using device: {DEVICE}")
    print(f"Loading training data from: {TRAIN_DIR}\n")

    # Load full dataset with CLAHE loader + train transforms
    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf,
                                        loader=clahe_loader)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    print(f"Classes found ({num_classes}): {class_names}")
    for cls, idx in full_dataset.class_to_idx.items():
        count = sum(1 for _, label in full_dataset.samples if label == idx)
        print(f"  [{idx}] {cls}: {count} images")

    # Save label map
    label_map = {str(idx): name for name, idx in full_dataset.class_to_idx.items()}
    with open(LABELS_OUT, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nLabels saved to {LABELS_OUT}")

    # 80/20 train/val split
    val_size   = max(1, int(len(full_dataset) * 0.20))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Apply val transforms to validation subset (same CLAHE loader)
    val_ds.dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_tf,
                                          loader=clahe_loader)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds   = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={train_loss/len(train_loader):.4f}  "
              f"val_acc={val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
            }, MODEL_OUT)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    train()
