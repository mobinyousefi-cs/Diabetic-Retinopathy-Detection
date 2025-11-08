#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: predict.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Inference script for the diabetic retinopathy classifier. Loads a trained
model checkpoint and predicts DR presence for a single image or a folder
of images.

Usage:
    # Single image
    python -m diabetic_retinopathy.predict \
        --image-path path/to/image.jpeg \
        --checkpoint outputs/models/best_model.pt

    # All images in a folder
    python -m diabetic_retinopathy.predict \
        --image-dir path/to/folder \
        --checkpoint outputs/models/best_model.pt

Notes:
- Outputs both the predicted class and softmax probabilities for each image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from .config import TrainingConfig
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict diabetic retinopathy for images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-path", type=Path, help="Path to a single image")
    group.add_argument("--image-dir", type=Path, help="Directory with images to predict")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    return parser.parse_args()


def build_inference_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(path: Path, transform: T.Compose, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
    return tensor.to(device)


def list_images_in_dir(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()]


def predict_batch(
    model: torch.nn.Module,
    image_paths: Iterable[Path],
    transform: T.Compose,
    device: torch.device,
) -> List[Tuple[Path, int, float, float]]:
    """Run prediction on a batch of image paths.

    Returns
    -------
    List of tuples: (image_path, predicted_class, prob_no_dr, prob_dr).
    """

    model.eval()
    results: List[Tuple[Path, int, float, float]] = []

    with torch.no_grad():
        for img_path in image_paths:
            tensor = load_image(img_path, transform, device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(probs.argmax())
            prob_no_dr, prob_dr = float(probs[0]), float(probs[1])
            results.append((img_path, pred_class, prob_no_dr, prob_dr))

    return results


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig()
    cfg.image_size = args.image_size

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    model = build_model(cfg).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    transform = build_inference_transform(cfg.image_size)

    if args.image_path is not None:
        image_paths = [args.image_path]
    else:
        image_paths = list_images_in_dir(args.image_dir)

    if not image_paths:
        raise SystemExit("No images found to predict.")

    results = predict_batch(model, image_paths, transform, device)

    for img_path, pred_class, prob_no_dr, prob_dr in results:
        print("========================================")
        print(f"Image: {img_path}")
        label_str = "DR present" if pred_class == 1 else "No DR"
        print(f"Predicted class: {pred_class} ({label_str})")
        print(f"Probability (no DR): {prob_no_dr:.4f}")
        print(f"Probability (DR):    {prob_dr:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
