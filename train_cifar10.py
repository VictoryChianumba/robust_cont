"""
Explanation Blindspot Replication: CIFAR-10 Training Script
============================================================
Trains 4 architectures on CIFAR-10 to later test whether standard
adversarial attacks produce explanation blindspots (high attribution
stability despite successful misclassification).

Models: ResNet-18, VGG-16, DenseNet-121, ViT-B/16
All pretrained on ImageNet, fine-tuned on CIFAR-10.

Usage:
    python train_cifar10.py --model resnet18 --seed 0
    python train_cifar10.py --model all --seed 0
"""

import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
CIFAR10_CLASSES = 10
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

MODEL_NAMES = ["resnet18", "vgg16", "densenet121", "vit_b_16"]

# CIFAR-10 mean/std (standard values)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Data
# ============================================================
def get_dataloaders():
    """
    CIFAR-10 with standard augmentation for training.
    Images resized to 224x224 for pretrained model compatibility.
    """
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, test_loader

# ============================================================
# Models
# ============================================================
def get_model(model_name: str) -> nn.Module:
    """
    Load pretrained ImageNet model and replace final classifier
    for CIFAR-10 (10 classes).
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, CIFAR10_CLASSES)

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(4096, CIFAR10_CLASSES)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, CIFAR10_CLASSES)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, CIFAR10_CLASSES)

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_NAMES}")

    return model

# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Eval", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def train_model(model_name: str, seed: int):
    """Full training pipeline for one model + seed combination."""

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training {model_name} | Seed {seed} | Device: {device}")

    print(f"{'='*60}")

    # Data
    train_loader, test_loader = get_dataloaders()

    # Model
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    best_acc = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            save_path = os.path.join(
                CHECKPOINT_DIR, f"{model_name}_seed{seed}_best.pth"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "model_name": model_name,
                "seed": seed,
            }, save_path)
            print(f"  -> Saved best model (acc: {best_acc:.4f})")

    # Save training history
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = os.path.join(
        RESULTS_DIR, f"{model_name}_seed{seed}_history.json"
    )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFinished {model_name} seed {seed} | Best Test Acc: {best_acc:.4f}")
    return best_acc

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models")
    parser.add_argument(
        "--model", type=str, default="resnet18",
        choices=MODEL_NAMES + ["all"],
        help="Model to train (or 'all' for all models)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.model == "all":
        results = {}
        for name in MODEL_NAMES:
            acc = train_model(name, args.seed)
            results[name] = acc
        print(f"\n{'='*60}")
        print("Summary:")
        for name, acc in results.items():
            print(f"  {name}: {acc:.4f}")
    else:
        train_model(args.model, args.seed)