# Realitat o Ficció? Deepfake Detection with CNN

TdR (Treball de Recerca), Batxillerat 2024. Revived and improved in 2026.
**Luna Ospina Loaiza**

> "Sometimes it is the people no one imagines anything of who do the things that no one can imagine."
> Alan Turing

## What is this?

A convolutional neural network that detects deepfake images, telling AI-manipulated
faces apart from real ones. Originally built in 2024 with zero prior deep learning
experience, using only online resources and Python. Rebuilt in 2026 with a much
larger dataset, GPU acceleration, and proper regularization.

## Results

| Version | Dataset size | Validation accuracy |
|---|---|---|
| 2024 (original TdR) | 2,041 images | 79% |
| 2026 (this repo) | 140,000 images | **91.65%** |

The 2024 model overfit badly, with a gap of about 4.7 points between training and
validation accuracy. The current model uses Dropout and EarlyStopping, closing that
gap to under 1 point, which means it generalizes to new images instead of memorizing
the training set.

## Try it yourself

```bash
pip install -r requirements.txt
python predict.py
```

It asks for an image path and prints a score between 0 (fake) and 1 (real), plus the
final verdict. The trained model is included in this repo, so no training is needed.

## Files

- `predict.py` loads the trained model and classifies a single image
- `train.py` trains the model from scratch on the 140k dataset (dataset downloaded separately, see below)
- `my_first_daughter.keras` the trained model, 3.27M parameters, about 38MB
- `origin_story/` the original 2024 scripts and early experiments, kept for history

## Architecture

Three convolutional blocks (Conv2D plus MaxPooling) for feature extraction, then a
Dense layer with Dropout(0.5) to prevent overfitting, and a single sigmoid output
for binary classification.

Input images are resized to 128x128. Note that `predict.py` converts from BGR to RGB
before inference, since OpenCV loads images in BGR while the model was trained on RGB.
Getting that wrong makes the model predict everything as real.

## Tech Stack

Python, TensorFlow / Keras, NumPy, OpenCV. Trained on an NVIDIA RTX 5060 through WSL2.

## Dataset

[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
from Kaggle. To retrain, download it and place it in a `dataset_140k/` folder at the
project root. It is excluded from git via `.gitignore` because of its size.

## What's next

- Training on datasets from other sources, since 91% on Kaggle is not 91% in the real world
- A web interface so anyone can check an image without installing anything
- Eventually, video support
