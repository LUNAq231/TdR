import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np
from sklearn.model_selection import train_test_split
import cv2  #OpenCV
import os

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
    return np.array(imatges), np.array(etiquetes)

# Ruta a las carpetas donde están las imágenes
ruta_imatges_reals = "C:\\Users\\Luna\\Documents\\Datasets\\Kagel\\FvsR_Images Extraidas\\real_and_fake_face_detection\\real"
ruta_imatges_falses = "C:\\Users\\Luna\\Documents\\Datasets\\Kagel\\FvsR_Images Extraidas\\real_and_fake_face_detection\\fake"

# Cargar imágenes reales y falsas
imatges_reals, etiquetes_reals = carregar_imatges(ruta_imatges_reals, 1)
imatges_falses, etiquetes_falses = carregar_imatges(ruta_imatges_falses, 0)

print(imatges_falses)