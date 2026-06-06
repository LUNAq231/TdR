import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Función para cargar imágenes y etiquetas
def carregar_imatges(ruta_carpeta, etiqueta):
    imatges = []
    etiquetes = []
    for arxiu in os.listdir(ruta_carpeta):
        ruta_arxiu = os.path.join(ruta_carpeta, arxiu)
        imatge = cv2.imread(ruta_arxiu)
        if imatge is not None:
            imatge = cv2.resize(imatge, (128, 128))  # Redimensionamos la imagen a 128x128 píxeles
            imatges.append(imatge)
            etiquetes.append(etiqueta)  # 1 si es real, 0 si es falsa
        else:
            print("Error al cargar la imagen.")
    return np.array(imatges), np.array(etiquetes)

# Ruta a las carpetas donde están las imágenes
ruta_imatges_reals = 'C:/Users/Luna/Documents/GitHub/TdR/imagenes/real_and_fake_face_detection/real_and_fake_face/training_real'
ruta_imatges_falses = 'C:/Users/Luna/Documents/GitHub/TdR/imagenes/real_and_fake_face_detection/real_and_fake_face/training_fake'

# Cargar imágenes reales y falsas
imatges_reals, etiquetes_reals = carregar_imatges(ruta_imatges_reals, 1)
imatges_falses, etiquetes_falses = carregar_imatges(ruta_imatges_falses, 0)

# Unir las imágenes y etiquetas en un solo conjunto de datos
imagenes = np.concatenate((imatges_reals, imatges_falses), axis=0)
etiquetas = np.concatenate((etiquetes_reals, etiquetes_falses), axis=0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

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

# Capa de salida: Un solo neurón con activación sigmoid
model.add(layers.Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
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
ruta_imagen_prueba = 'ruta/a/la/imagen/de/deepfake.jpg'  # Cambia esto a la ruta de tu imagen

# Hacer la predicción
prediccion = predecir_imagen(model, ruta_imagen_prueba)

# Interpretar y mostrar el resultado
resultado = interpretar_prediccion(prediccion)
print(f"Predicción: {prediccion}, Resultado: {resultado}")
