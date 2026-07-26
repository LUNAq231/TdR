import os
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_140k', 'real_vs_fake', 'real-vs-fake', 'train')
valid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_140k', 'real_vs_fake', 'real-vs-fake', 'valid')

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

# Construir el modelo
model = models.Sequential()

# Primera capa: Convolucional
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))

# Capa de agrupamiento (MaxPooling)
model.add(layers.MaxPooling2D((2, 2)))

# Segunda capa: Convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Capa de agrupamiento (MaxPooling)
model.add(layers.MaxPooling2D((2, 2)))

# Tercera capa: Convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanar los datos para conectarlos a la red densa
model.add(layers.Flatten())

# Capa densa (completamente conectada)
model.add(layers.Dense(64, activation='relu'))

# Capa para evitar el overfitting
model.add(layers.Dropout(0.5))

# Capa de salida: Un solo neurón con activación sigmoid
model.add(layers.Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('mi_primera_hija.keras', monitor='val_loss', save_best_only=True)

history = model.fit(
    train_dataset,
    epochs = 20,
    validation_data = val_dataset,
    callbacks = [early_stop, checkpoint]
)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(val_dataset)
print(f"Precisión en el conjunto de prueba: {test_acc}")

# Función para cargar y preprocesar una imagen
def cargar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("Error al cargar la imagen.")
        return None

    # Redimensionar y normalizar
    imagen = cv2.resize(imagen, (128, 128))
    imagen = imagen.astype('float32') / 255.0  # Normaliza entre 0 y 1
    imagen = np.expand_dims(imagen, axis=0)  # Forma: (1, 128, 128, 3)
    return imagen

# Función para hacer una predicción
def predecir_imagen(model, ruta_imagen):
    imagen = cargar_imagen(ruta_imagen)
    if imagen is not None:
        prediccion = model.predict(imagen)
        return prediccion[0][0]  # Obtener el valor escalar de la predicción

# Función para interpretar la predicción
def interpretar_prediccion(prediccion):
    if prediccion >= 0.5:
        return "La imagen es real (1)"
    else:
        return "La imagen es falsa (0)"

# Ruta de la imagen que deseas probar
ruta_imagen_prueba = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_140k', 'real_vs_fake', 'real-vs-fake', 'test', 'real', '00001.jpg')

# Hacer la predicción
prediccion = predecir_imagen(model, ruta_imagen_prueba)

# Interpretar y mostrar el resultado
resultado = interpretar_prediccion(prediccion)
print(f"Predicción: {prediccion}, Resultado: {resultado}")
