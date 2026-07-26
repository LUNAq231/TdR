import os  # Asegúrate de importar el módulo os

# Verificar si tienes permisos de lectura en la carpeta
ruta_imatges_falses = 'C:/Users/Luna/Documents/GitHub/TdR/imagenes/real_and_fake_face_detection/real_and_fake_face/training_fake'


# Verifica si la carpeta existe
if os.path.exists(ruta_imatges_falses):
    print(f"La carpeta {ruta_imatges_falses} existe.")
    
    # Verifica los permisos
    if os.access(ruta_imatges_falses, os.R_OK):
        print("Tienes permisos de lectura.")
        
        # Lista los archivos en la carpeta
        archivos = os.listdir(ruta_imatges_falses)
        print(f"Archivos en la carpeta: {archivos}")
    else:
        print("No tienes permisos de lectura en esta carpeta.")
else:
    print(f"La carpeta {ruta_imatges_falses} no existe.")