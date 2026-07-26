# Realitat o Ficció? — Deepfake Detection with CNN

TdR (Treball de Recerca) — Batxillerat 2024, revived and improved in 2026
**Luna Ospina Loaiza**

"Sometimes it is the people no one imagines anything of who do the things that no one can imagine." — Alan Turing

## What is this?

A convolutional neural network (CNN) that detects deepfake images — distinguishing
AI-manipulated faces from real ones. Originally built in 2024 with zero prior deep
learning experience, using only online resources and Python. Revived and retrained
in 2026 with a much larger dataset, GPU acceleration, and overfitting fixes.

## Results

| Version | Dataset size | Accuracy (validation) |
|---|---|---|
| 2024 (original TdR) | 2,041 images | 79% |
| 2026 (this repo) | 140,000 images | **91.65%** |

The 2024 model overfit badly (train/val gap of ~4.7 points). The current model
uses Dropout and EarlyStopping, closing that gap to under 1 point — meaning it
generalizes to new images instead of memorizing the training set.

## Try it yourself

```bash
pip install -r requirements.txt
python predict.py
```

It'll ask for an image path, then print a score between 0 (fake) and 1 (real).

## Files

- `train.py` — trains the model from scratch on the 140k dataset (requires downloading it separately from Kaggle, see below)
- `predict.py` — loads the trained model and classifies a single image, no training needed
- `my_first_daughter.keras` — the trained model (3.27M parameters, ~38MB)
- `origin_story/` — the original 2024 scripts and early experiments, kept for history

## Tech Stack

- Python, TensorFlow / Keras, NumPy, OpenCV
- Trained on an NVIDIA RTX 5060 via WSL2

## Dataset

[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
from Kaggle. To retrain, download it and place it in a `dataset_140k/` folder at
the project root (already excluded from git via `.gitignore` due to size).

## Architecture

3 convolutional blocks (Conv2D + MaxPooling) for feature extraction, followed by
a Dense layer with Dropout(0.5) to prevent overfitting, and a sigmoid output for
binary classification (real vs fake).
