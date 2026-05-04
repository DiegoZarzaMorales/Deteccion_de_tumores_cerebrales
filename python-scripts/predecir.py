"""Wrapper para usar las funciones de predicción de ml.predecir.

Incluye un ejemplo de uso en el bloque __main__ que asume que existe
una imagen de prueba en "brats_data/T1wCE/Image-15.dcm" dentro del
directorio raíz del proyecto.
"""

import os

from ml.predecir import *  # re-exportar helpers para compatibilidad


if __name__ == "__main__":
    import torch

    MODEL_PATH = "models/best_model.pth"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(base_dir, "brats_data", "T1wCE", "Image-15.dcm")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = cargar_modelo(MODEL_PATH, device=DEVICE)
    img, mascara, mascara_prob = predecir_imagen(
        modelo, IMAGE_PATH, device=DEVICE, threshold=0.5
    )
    visualizar_prediccion(img, mascara, mascara_prob)
