import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'my_first_daughter.keras')
IMAGE_SIZE = (128, 128)

# Load the trained model from disk (the 3.27 million weights we trained).
model = load_model(MODEL_PATH)

# Ask the user for the image path.
image_path = input("Enter the path to an image: ")

if not os.path.isfile(image_path):
    print(f"Error: no file found at '{image_path}'. Check the path and try again.")
    exit(1)

# Load the image and prepare it the same way training images were prepared.
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, IMAGE_SIZE)
image = image.astype('float32')
image = np.expand_dims(image, axis=0)

# Run the prediction.
prediction = model.predict(image, verbose=0)[0][0]
label = "REAL" if prediction >= 0.5 else "FAKE"

print(f"Score: {prediction:.4f} -> Prediction: {label}")