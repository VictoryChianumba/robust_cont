"""
CIFAR-10 Adversarial Attack Sweep
==================================
Runs FGSM, PGD, FGSM_LP, PGD_LP, and DeepFool across all trained
checkpoints using the evasion library.

Usage:
    python run_attacks.py
    python run_attacks.py --attack FGSM
    python run_attacks.py --model resnet18
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from evasion import ChannelConfig, RunConfig, EvasionRunner
from evasion.explainers import IGExplainer

# ---- constants from training script ----
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = 10
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

SEEDS = [0, 42, 94]
MODEL_NAMES = ["resnet18", "vgg16", "densenet121", "vit_b_16"]
ALL_ATTACKS = ["FGSM", "PGD", "FGSM_LP", "PGD_LP", "DeepFool"]

# ---- clamp bounds derived from CIFAR-10 normalisation ----
# pixel values in [0,1] -> normalised: (0 - mean) / std and (1 - mean) / std
_mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
_std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
TRAIN_MIN_T = (torch.zeros(3, 1, 1) - _mean) / _std   # (3, 1, 1)
TRAIN_MAX_T = (torch.ones(3, 1, 1)  - _mean) / _std   # (3, 1, 1)

# prenorm_std is per-channel std (same as CIFAR10_STD here)
PRENORM_STD_NP = np.array(CIFAR10_STD)


def get_test_loader(resize_224: bool = False) -> DataLoader:
    transform = transforms.Compose([
        *([transforms.Resize(224)] if resize_224 else []),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)


def load_all_test_data(resize_224: bool = False):
    """Load entire test set into a single tensor pair."""
    loader = get_test_loader(resize_224)
    X, y = next(iter(loader))
    return X, y


def get_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, CIFAR10_CLASSES)

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, CIFAR10_CLASSES),
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        model.classifier = nn.Linear(model.classifier.in_features, CIFAR10_CLASSES)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, CIFAR10_CLASSES)

    return model


def load_checkpoint(model: nn.Module, model_name: str, seed: int, device: torch.device) -> nn.Module:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_seed{seed}_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run(attacks_to_run, models_to_run):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # evasion config — no ROI for CIFAR-10
    channel_cfg = ChannelConfig()

    run_cfg = RunConfig(
        budget_grid=[0.25, 0.5, 1.0, 2.0],
        median_std_pre=float(np.median(PRENORM_STD_NP)),
        prenorm_std_np=PRENORM_STD_NP,
        attr_max_n=128,
        n_steps_ig=16,
        seeds=SEEDS,
    )

    train_min_t = TRAIN_MIN_T.to(device)
    train_max_t = TRAIN_MAX_T.to(device)

    all_rows = []

    for model_name in models_to_run:
        print(f"\n{'='*50}\nModel: {model_name}")
        resize = (model_name == "vit_b_16")

        # load test data once per model (ViT needs 224x224)
        X, y = load_all_test_data(resize_224=resize)
        X = X.to(device)
        y = y.to(device)

        for seed in SEEDS:
            print(f"  Seed {seed}")
            model = get_model(model_name).to(device)
            model = load_checkpoint(model, model_name, seed, device)

            ig = IGExplainer(model, n_steps=run_cfg.n_steps_ig, max_n=run_cfg.attr_max_n)

            runner = EvasionRunner(
                model=model,
                X=X, y=y,
                train_min_t=train_min_t,
                train_max_t=train_max_t,
                run_cfg=run_cfg,
                channel_cfg=channel_cfg,
                explainers={"IG": ig.attribute},
            )

            rows = []
            if "FGSM" in attacks_to_run:
                rows += runner.run_linf_sweep("FGSM", seed=seed, lp_sigma_t=None)
            if "PGD" in attacks_to_run:
                rows += runner.run_linf_sweep("PGD", seed=seed, lp_sigma_t=None)
            if "FGSM_LP" in attacks_to_run:
                rows += runner.run_linf_sweep("FGSM", seed=seed, lp_sigma_t=3.0)
            if "PGD_LP" in attacks_to_run:
                rows += runner.run_linf_sweep("PGD", seed=seed, lp_sigma_t=3.0)
            if "DeepFool" in attacks_to_run:
                rows += runner.run_deepfool_sweep(seed=seed)

            for r in rows:
                r["model_name"] = model_name
                r["seed"] = seed

            all_rows.extend(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "adversarial_results_cifar10.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} with {len(all_rows)} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="all", choices=ALL_ATTACKS + ["all"])
    parser.add_argument("--model", type=str, default="all", choices=MODEL_NAMES + ["all"])
    args = parser.parse_args()

    attacks = ALL_ATTACKS if args.attack == "all" else [args.attack]
    model_names = MODEL_NAMES if args.model == "all" else [args.model]

    run(attacks, model_names)