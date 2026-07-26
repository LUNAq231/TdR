import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np
from sklearn.model_selection import train_test_split
import cv2  #OpenCV
import os

imatges = []
etiquetes = []

def carregar_imatges(ruta_carpeta, etiqueta):
    for arxiu in os.listdir(ruta_carpeta):
        ruta_arxiu = os.path.join(ruta_carpeta, arxiu)
        imatge = cv2.imread(ruta_arxiu)
        if imatge is not None:
            imatge = cv2.resize(imatge, (128, 128))  
            imatges.append(imatge)
            etiquetes.append(etiqueta)  
    return np.array(imatges), np.array(etiquetes)


#Ruta a las carpetas donde están las imágenes
ruta_imatges_reals = 'C:/Users/Luna/Documents/GitHub/TdR/imagenes/real_and_fake_face_detection/real_and_fake_face/training_real'
ruta_imatges_falses = 'C:/Users/Luna/Documents/GitHub/TdR/imagenes/real_and_fake_face_detection/real_and_fake_face/training_fake'


# Cargar imágenes reales y falsas
imatges_reals, etiquetes_reals = carregar_imatges(ruta_imatges_reals, 1)
imatges_falses, etiquetes_falses = carregar_imatges(ruta_imatges_falses, 0)



# Unir las imágenes y etiquetas en un solo conjunto de datos
imagenes = np.concatenate((imatges_reals, imatges_falses), axis=0)
etiquetas = np.concatenate((etiquetes_reals, etiquetes_falses), axis=0)


X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)



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

# Capa de salida: Un solo neurón con activación sigmoid
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Precisión en el conjunto de prueba: {test_acc}")
