#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: train.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Training script for the diabetic retinopathy classifier. Handles:
- Data loading and preprocessing
- Stratified train/validation split
- Model creation and optimization
- Training loop with validation and checkpointing

Usage:
    python -m diabetic_retinopathy.train \
        --data-dir data \
        --batch-size 32 \
        --epochs 20

Notes:
- Assumes that `trainLabels.csv` is located at `<data_dir>/trainLabels.csv`.
- Images are expected under `<data_dir>/train_images/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .datasets import RetinopathyDataset, build_transforms
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diabetic retinopathy classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Base data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_datasets(cfg: TrainingConfig) -> Tuple[RetinopathyDataset, RetinopathyDataset]:
    """Create train and validation datasets with stratified splitting."""

    # Temporary dataset just to read labels for splitting
    temp_ds = RetinopathyDataset(
        images_dir=cfg.train_images_dir,
        labels_csv=cfg.train_labels_csv,
        transform=None,
    )

    labels = temp_ds.labels
    indices = np.arange(len(temp_ds))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.val_split,
        random_state=cfg.random_seed,
    )

    train_idx, val_idx = next(splitter.split(indices, labels))

    train_transforms = build_transforms(cfg.image_size, train=True)
    val_transforms = build_transforms(cfg.image_size, train=False)

    train_ds = RetinopathyDataset(
        images_dir=cfg.train_images_dir,
        labels_csv=cfg.train_labels_csv,
        transform=train_transforms,
        indices=train_idx,
    )

    val_ds = RetinopathyDataset(
        images_dir=cfg.train_images_dir,
        labels_csv=cfg.train_labels_csv,
        transform=val_transforms,
        indices=val_idx,
    )

    return train_ds, val_ds


def create_dataloaders(
    train_ds: RetinopathyDataset,
    val_ds: RetinopathyDataset,
    cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig()
    cfg.data_dir = args.data_dir
    cfg.train_images_dir = args.data_dir / "train_images"
    cfg.train_labels_csv = args.data_dir / "trainLabels.csv"
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.learning_rate = args.learning_rate
    cfg.image_size = args.image_size
    cfg.val_split = args.val_split
    cfg.random_seed = args.seed

    cfg.ensure_output_dirs()

    set_seed(cfg.random_seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_ds, val_ds = create_datasets(cfg)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, cfg)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    best_val_acc = 0.0
    best_model_path = cfg.output_dir / "models" / "best_model.pt"
    metrics_path = cfg.output_dir / "metrics" / "training_metrics.json"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"Epoch {epoch}/{cfg.num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to: {best_model_path}")

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
