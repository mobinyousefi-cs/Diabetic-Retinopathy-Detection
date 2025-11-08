# Diabetic Retinopathy Detection

Automatic screening of diabetic retinopathy using deep learning.

This repository implements a clean, reproducible PyTorch pipeline to train and evaluate a convolutional neural network on retinal fundus images. The goal is to classify whether a patient shows signs of diabetic retinopathy (DR) or not.

The project is structured as a modern Python package and is suitable both for learning and for building upon in research or production.

---

## 1. Project Overview

Diabetic retinopathy is a leading cause of vision loss worldwide. Early detection through regular screening is critical.

In this project we:

- Load and preprocess retina images from the **Kaggle Diabetic Retinopathy Detection** dataset.
- Map the original 5 severity levels (0–4) to a **binary label**: `0 = no DR`, `1 = any DR`.
- Train a CNN-based classifier (EfficientNet-B0 backbone by default).
- Evaluate the model on a validation/test set.
- Provide a simple **CLI** to run inference on new images.

Dataset: Kaggle competition – *Diabetic Retinopathy Detection*.

> You must manually download the dataset from Kaggle and place it on your machine. The code assumes a local folder structure described below.

Kaggle page (sign-in required): `https://www.kaggle.com/c/diabetic-retinopathy-detection/data`

---

## 2. Repository Structure

```text
.
├─ LICENSE
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ train_images/          # All training images (JPEGs from Kaggle)
│  ├─ test_images/           # Optional test images (if you create a split)
│  └─ trainLabels.csv        # Kaggle training labels CSV
├─ outputs/
│  ├─ models/                # Saved model checkpoints
│  ├─ logs/                  # Training logs
│  └─ metrics/               # Metrics, plots, reports
├─ src/
│  └─ diabetic_retinopathy/
│     ├─ __init__.py
│     ├─ config.py           # All hyperparameters and paths
│     ├─ datasets.py         # Dataset & transforms
│     ├─ model.py            # Model architectures
│     ├─ train.py            # Training loop & validation
│     ├─ evaluate.py         # Evaluation on a held-out set
│     └─ predict.py          # Inference on single images or folders
└─ tests/
   └─ test_smoke.py          # Basic smoke tests
```

You can adapt the structure as needed, but this layout is a good starting point for maintainable DS/ML projects.

---

## 3. Installation

### 3.1. Create and activate a virtual environment

It is strongly recommended to use a virtual environment (e.g. `venv` or Conda).

```bash
# Using venv (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate        # Windows (PowerShell or cmd)
```

### 3.2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you install PyTorch manually from the official website (e.g. with GPU support), you can remove or adjust the `torch` and `torchvision` lines in `requirements.txt`.

---

## 4. Preparing the Data

1. Go to the Kaggle competition page for **Diabetic Retinopathy Detection**.
2. Download the **training images** and **trainLabels.csv** file.
3. Extract the training images into:

   ```text
   data/train_images/
   ```

4. Place `trainLabels.csv` into:

   ```text
   data/trainLabels.csv
   ```

5. (Optional) If you want a fixed test set, you can later export a subset of the dataset into `data/test_images/` with a corresponding CSV.

The default configuration assumes that each image file in `train_images` is named `<image_id>.jpeg` as in the Kaggle dataset.

---

## 5. Usage

All entry points live inside the `src/diabetic_retinopathy` package and can be called either as modules or directly via `python -m` (recommended in development).

For clarity, commands below assume you are in the project root and the virtual environment is activated.

### 5.1. Training

Train a model using the default configuration:

```bash
python -m diabetic_retinopathy.train
```

Useful options:

```bash
python -m diabetic_retinopathy.train \
    --data-dir data \
    --batch-size 32 \
    --epochs 20 \
    --learning-rate 1e-4 \
    --image-size 224
```

During training, the script will:

- Create a stratified train/validation split based on the binary label.
- Train the model and evaluate after each epoch.
- Save the best-performing model checkpoint to `outputs/models/best_model.pt`.
- Log metrics to the console; basic metrics are saved as JSON in `outputs/metrics/`.

### 5.2. Evaluation

Run evaluation on a dataset and a saved model checkpoint:

```bash
python -m diabetic_retinopathy.evaluate \
    --data-dir data \
    --checkpoint outputs/models/best_model.pt
```

This will compute accuracy, confusion matrix, and a few additional metrics for the specified split (by default the validation set used during training).

### 5.3. Prediction / Inference

Predict DR presence for a single image:

```bash
python -m diabetic_retinopathy.predict \
    --image-path path/to/retina_image.jpeg \
    --checkpoint outputs/models/best_model.pt
```

Example output:

```text
Image: path/to/retina_image.jpeg
Predicted class: 1 (diabetic retinopathy present)
Probability (DR present): 0.87
Probability (no DR):      0.13
```

You can also run predictions on all images inside a folder.

---

## 6. Configuration

Most hyperparameters and paths are defined in `src/diabetic_retinopathy/config.py`.

Key options include:

- `data_dir`: base directory containing `train_images` and `trainLabels.csv`.
- `image_size`: input image resolution (default `224`).
- `batch_size`: mini-batch size.
- `learning_rate`: learning rate for the optimizer.
- `weight_decay`: L2 regularization.
- `num_epochs`: number of training epochs.
- `val_split`: fraction of data used for validation.
- `random_seed`: random seed for reproducibility.

These can be overridden from the command line for quick experiments.

---

## 7. Model Details

By default, the model is based on **EfficientNet-B0** (from `torchvision.models`) with:

- Pretrained ImageNet weights (if available).
- Last classification layer replaced with a 2-unit head for binary classification.
- Standard data augmentations (random flips, small rotations, normalization) for the training set.

If EfficientNet is not available on your system, the code falls back to a small custom CNN so the project remains runnable in restricted environments.

You can easily extend `model.py` to:

- Use other backbones (ResNet, ViT, etc.).
- Add multi-head outputs, attention modules, or uncertainty estimation.

---

## 8. Tests

Basic tests live in the `tests` directory.

Run tests with:

```bash
pytest
```

The smoke tests check that:

- The project imports successfully.
- The configuration object can be instantiated.
- A small synthetic batch passes through the model.

You can add more unit tests and integration tests as the project grows.

---

## 9. Coding Style & License

The codebase follows a clear, modular style appropriate for a MSc-level data science project:

- Type hints where helpful.
- Explicit configuration and reproducibility.
- Separation of concerns (data, model, training, inference).

The project is released under the **MIT License** (see `LICENSE`).

Author: **Mobin Yousefi** · GitHub: [mobinyousefi-cs](https://github.com/mobinyousefi-cs)

