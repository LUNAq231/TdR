import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths are built relative to this file's location, so the script works
# no matter which folder you run it from.
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_PATH, 'dataset_140k', 'real_vs_fake', 'real-vs-fake', 'train')
VALID_PATH = os.path.join(BASE_PATH, 'dataset_140k', 'real_vs_fake', 'real-vs-fake', 'valid')
MODEL_PATH = os.path.join(BASE_PATH, 'my_first_daughter.keras')

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load images directly from disk in batches, instead of loading
# everything into RAM at once. Labels are inferred from folder names
# (train/real, train/fake).
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Model architecture: 3 convolutional blocks to extract visual features,
# followed by a dense classifier head.
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Dropout randomly disables 50% of neurons during training to prevent
# overfitting (memorizing the training images instead of learning
# general patterns).
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping monitors validation loss and stops training once it stops
# improving for 3 consecutive epochs, restoring the best weights found.
# ModelCheckpoint saves the best model to disk automatically.
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stop, checkpoint]
)

val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Final validation accuracy: {val_accuracy:.4f}")
print(f"Model saved to: {MODEL_PATH}")


def load_image(image_path):
    # Load an image from disk and prepare it the exact same way the training images were prepared 
    # (BGR to RGB + resize + add the batch dimension the model expects).
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(image_path):
    image = load_image(image_path)
    prediction = model.predict(image, verbose=0)[0][0]
    label = "REAL" if prediction >= 0.5 else "FAKE"
    return prediction, label


# Quick check: grab one real and one fake image from the validation set 
# (never used to adjust weights) 
# and confirm the model tells them apart correctly right after training.
print("\n--- Quick check ---")

real_folder = os.path.join(VALID_PATH, 'real')
fake_folder = os.path.join(VALID_PATH, 'fake')

sample_real = os.path.join(real_folder, os.listdir(real_folder)[0])
sample_fake = os.path.join(fake_folder, os.listdir(fake_folder)[0])

real_score, real_label = predict_image(sample_real)
fake_score, fake_label = predict_image(sample_fake)

print(f"Real image  -> score: {real_score:.4f} -> predicted: {real_label}")
print(f"Fake image  -> score: {fake_score:.4f} -> predicted: {fake_label}")