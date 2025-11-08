#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: datasets.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Dataset and transform definitions for diabetic retinopathy classification.
This module provides a PyTorch Dataset that reads Kaggle retina images and
converts the 5-class severity label into a binary label.

Usage:
    from diabetic_retinopathy.config import TrainingConfig
    from diabetic_retinopathy.datasets import RetinopathyDataset, build_transforms

    cfg = TrainingConfig()
    transforms = build_transforms(cfg.image_size, train=True)
    dataset = RetinopathyDataset(
        images_dir=cfg.train_images_dir,
        labels_csv=cfg.train_labels_csv,
        transform=transforms,
    )

Notes:
- The original labels are integers in [0, 4]. We map them to {0, 1} via
  `binary_label = int(severity > 0)`.
- The dataset expects files named `<image_id>.jpeg` by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def severity_to_binary_label(severity: int) -> int:
    """Map a 5-level severity label (0-4) to a binary label.

    Parameters
    ----------
    severity:
        Original Kaggle DR severity label.

    Returns
    -------
    int
        0 if no DR, 1 otherwise.
    """

    return int(severity > 0)


def build_transforms(image_size: int, train: bool = True) -> Callable:
    """Build torchvision transforms for training or evaluation.

    Parameters
    ----------
    image_size:
        Target spatial size (height = width = image_size).
    train:
        If True, include data augmentation.

    Returns
    -------
    Callable
        A transform callable that takes a PIL.Image and returns a tensor.
    """

    if train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.1),
                T.RandomRotation(degrees=10),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class RetinopathyDataset(Dataset):
    """PyTorch Dataset for diabetic retinopathy detection.

    Parameters
    ----------
    images_dir:
        Directory containing retina images.
    labels_csv:
        Path to the labels CSV file (Kaggle trainLabels.csv format).
    transform:
        Optional transform to apply to each image.
    indices:
        Optional array of indices to subset the dataset (for train/val splits).
    image_ext:
        File extension of the images (default: ".jpeg").
    """

    def __init__(
        self,
        images_dir: Path,
        labels_csv: Path,
        transform: Optional[Callable] = None,
        indices: Optional[np.ndarray] = None,
        image_ext: str = ".jpeg",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_csv = Path(labels_csv)
        self.transform = transform
        self.image_ext = image_ext

        df = pd.read_csv(self.labels_csv)
        # Expect columns: image, level
        self.image_ids = df["image"].values
        self.severity = df["level"].values.astype(int)
        self.labels = np.array([severity_to_binary_label(s) for s in self.severity], dtype=np.int64)

        if indices is not None:
            self.image_ids = self.image_ids[indices]
            self.labels = self.labels[indices]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.image_ids)

    def _resolve_image_path(self, image_id: str) -> Path:
        return self.images_dir / f"{image_id}{self.image_ext}"

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", int]:
        from torch import Tensor  # Local import to avoid unconditional torch dependency at import

        image_id = self.image_ids[idx]
        label = int(self.labels[idx])

        img_path = self._resolve_image_path(image_id)
        if not img_path.is_file():
            # Fallback: try common alternative extensions
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = img_path.with_suffix(ext)
                if candidate.is_file():
                    img_path = candidate
                    break

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        assert isinstance(image, Tensor)
        return image, label


__all__ = ["RetinopathyDataset", "build_transforms", "severity_to_binary_label"]
