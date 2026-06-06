# Realitat o Ficció? — Deepfake Detection with CNN

TdR (Treball de Recerca) — Batxillerat 2024  
**Luna Ospina Loaiza**

"Sometimes it is the people no one imagines anything of who do the things that no one can imagine." — Alan Turing

## What is this?
A convolutional neural network (CNN) built from scratch to detect deepfake images — distinguishing AI-manipulated faces from real ones. 
Built with zero prior deep learning experience, using only online resources and Python.

## Results
- 79% accuracy on test data
- Overfitting detected (val_loss diverges after epoch 4) — 
  addressable with more training data and regularization techniques

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- scikit-learn

## Dataset

[Real vs Fake Faces](https://www.kaggle.com/datasets/uditsharma72/real-vs-fake-faces) 
from Kaggle — images split into `training_real` and `training_fake`.

## Architecture

CNN with 3 convolutional layers + MaxPooling, Flatten, 
Dense layer, and sigmoid output for binary classification.
