#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Central configuration definitions and helper utilities for the project.
All tunable hyperparameters and important paths live here.

Usage:
    from diabetic_retinopathy.config import TrainingConfig
    cfg = TrainingConfig()

Notes:
- Adjust defaults to match your local environment (e.g. GPU, data_dir).
- Command-line interfaces may override some fields at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import torch


@dataclass
class TrainingConfig:
    """Configuration container for training and evaluation.

    Attributes
    ----------
    project_name:
        Human-readable project name.
    data_dir:
        Base directory for dataset files.
    train_images_dir:
        Directory containing training images.
    train_labels_csv:
        Path to CSV file with labels (Kaggle trainLabels.csv).
    output_dir:
        Base directory for all outputs (models, logs, metrics).
    image_size:
        Square size to which input images are resized.
    batch_size:
        Batch size for training and evaluation.
    num_workers:
        Number of DataLoader workers.
    learning_rate:
        Learning rate for the optimizer.
    weight_decay:
        L2 regularization factor.
    num_epochs:
        Number of training epochs.
    val_split:
        Fraction of the dataset used for validation.
    random_seed:
        Random seed for reproducibility.
    num_classes:
        Number of output classes (2 for binary DR / no-DR).
    device:
        Device identifier ("cuda" or "cpu").
    use_pretrained:
        Whether to use a pretrained backbone (if available).
    """

    project_name: str = "Diabetic Retinopathy Detection"

    data_dir: Path = Path("data")
    train_images_dir: Path = data_dir / "train_images"
    train_labels_csv: Path = data_dir / "trainLabels.csv"

    output_dir: Path = Path("outputs")

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20

    val_split: float = 0.2
    random_seed: int = 42

    num_classes: int = 2
    use_pretrained: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of the config."""

        return asdict(self)

    def ensure_output_dirs(self) -> None:
        """Create all necessary output directories if they do not exist."""

        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)


__all__ = ["TrainingConfig"]
