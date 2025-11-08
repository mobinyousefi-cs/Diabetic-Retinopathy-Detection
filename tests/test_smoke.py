#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: test_smoke.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Basic smoke tests for the diabetic retinopathy project. These tests
ensure that:
- The package imports correctly.
- The configuration object can be instantiated.
- A small batch passes through the model without errors.

Usage:
    pytest tests/test_smoke.py

Notes:
- These are not exhaustive tests, but they are useful to catch obvious
  wiring issues after refactoring.
"""

from __future__ import annotations

import torch

from diabetic_retinopathy.config import TrainingConfig
from diabetic_retinopathy.model import build_model


def test_config_instantiation() -> None:
    cfg = TrainingConfig()
    assert cfg.image_size > 0
    assert cfg.num_classes == 2


def test_model_forward_pass() -> None:
    cfg = TrainingConfig(image_size=224, num_classes=2)
    model = build_model(cfg)

    dummy_input = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    outputs = model(dummy_input)

    assert outputs.shape == (2, cfg.num_classes)
