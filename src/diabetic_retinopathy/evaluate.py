#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: evaluate.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Evaluation script for the diabetic retinopathy classifier. Loads a trained
checkpoint and computes metrics on a validation or test split.

Usage:
    python -m diabetic_retinopathy.evaluate \
        --data-dir data \
        --checkpoint outputs/models/best_model.pt

Notes:
- By default, this script reuses the validation split created in `train.py`
  by randomly splitting again with the same seed and fraction.
- For a dedicated test set, create a separate CSV and change the paths here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .datasets import RetinopathyDataset, build_transforms
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diabetic retinopathy classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_eval_loader(cfg: TrainingConfig) -> DataLoader:
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
    _, val_idx = next(splitter.split(indices, labels))

    transforms = build_transforms(cfg.image_size, train=False)
    val_ds = RetinopathyDataset(
        images_dir=cfg.train_images_dir,
        labels_csv=cfg.train_labels_csv,
        transform=transforms,
        indices=val_idx,
    )

    loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return loader


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray, str]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["no_dr", "dr"], digits=4)

    return acc, cm, report


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig()
    cfg.data_dir = args.data_dir
    cfg.train_images_dir = args.data_dir / "train_images"
    cfg.train_labels_csv = args.data_dir / "trainLabels.csv"
    cfg.image_size = args.image_size
    cfg.val_split = args.val_split
    cfg.random_seed = args.seed

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    loader = build_eval_loader(cfg)

    model = build_model(cfg).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    acc, cm, report = evaluate_model(model, loader, device)

    print("\nEvaluation Results")
    print("-------------------")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":  # pragma: no cover
    main()
