import os  # Asegúrate de importar el módulo os
import cv2  #OpenCV
import numpy as np
from sklearn.model_selection import train_test_split

ruta_imatges_falses = 'C:/Users/Luna/Documents/GitHub/TdR/imagenes/real_and_fake_face_detection/real_and_fake_face/training_fake'
ruta_imatges_reals = 'C:/Users/Luna/Documents/GitHub/TdR/imagenes/real_and_fake_face_detection/real_and_fake_face/training_real'

imatges = []
etiquetes = []

def carregar_imatges(ruta_carpeta, etiqueta):
    for arxiu in os.listdir(ruta_carpeta):
        ruta_arxiu = os.path.join(ruta_carpeta, arxiu)
        imatge = cv2.imread(ruta_arxiu)
        if imatge is not None:
            imatge = cv2.resize(imatge, (128, 128))  # Redimensionamos la imagen a 128x128 píxeles
            imatges.append(imatge)
            etiquetes.append(etiqueta)  # 1 si es real, 0 si es falsa
        else: 
            print("Error en carregar la imatge.")
    return np.array(imatges), np.array(etiquetes)
print(imatges)

imagenes_reales, etiquetas_reales = carregar_imatges(ruta_imatges_falses,0)
imagenes_falsas, etiquetas_falsas = carregar_imatges(ruta_imatges_reals,1)

imagenes = np.concatenate((imagenes_reales, imagenes_falsas), axis=0)
etiquetas = np.concatenate((etiquetas_reales, etiquetas_falsas), axis=0)

X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42) 

print(imagenes)