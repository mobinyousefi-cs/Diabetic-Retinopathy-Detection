#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: model.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Model definitions for diabetic retinopathy classification.

Usage:
    from diabetic_retinopathy.config import TrainingConfig
    from diabetic_retinopathy.model import build_model

    cfg = TrainingConfig()
    model = build_model(cfg)

Notes:
- Uses EfficientNet-B0 as the default backbone when available.
- Falls back to a small custom CNN when EfficientNet cannot be imported.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .config import TrainingConfig


class SimpleCNN(nn.Module):
    """A small CNN fallback model for binary classification.

    This is intentionally lightweight so that it can run even on modest hardware
    or in environments without access to large pretrained backbones.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return x


def _build_efficientnet_backbone(num_classes: int, use_pretrained: bool) -> nn.Module:
    """Build an EfficientNet-B0 backbone if available.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    use_pretrained:
        Whether to load ImageNet pretrained weights.

    Returns
    -------
    nn.Module
        A classification model.
    """

    try:
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("torchvision is required for EfficientNet backbone") from exc

    # Newer torchvision versions expose weights enums; we handle both styles.
    try:  # pragma: no cover - depends on torchvision version
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    except Exception:  # pragma: no cover - fallback for older APIs
        model = models.efficientnet_b0(pretrained=use_pretrained)  # type: ignore[arg-type]
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def build_model(cfg: TrainingConfig) -> nn.Module:
    """Factory function to build the classification model from config.

    Parameters
    ----------
    cfg:
        TrainingConfig instance with model-related parameters.

    Returns
    -------
    nn.Module
        Instantiated model ready for training or inference.
    """

    try:
        model = _build_efficientnet_backbone(num_classes=cfg.num_classes, use_pretrained=cfg.use_pretrained)
    except Exception:
        # Fallback to SimpleCNN if anything goes wrong (e.g. missing torchvision).
        model = SimpleCNN(num_classes=cfg.num_classes)

    return model


__all__ = ["build_model", "SimpleCNN"]
