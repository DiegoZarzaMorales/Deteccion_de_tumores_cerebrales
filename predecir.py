"""Wrapper para usar las funciones de predicción de ml.predecir."""

from ml.predecir import *  # re-exportar helpers para compatibilidad


if __name__ == "__main__":
    import torch

    MODEL_PATH = "models/best_model.pth"
    IMAGE_PATH = (
        r"c:\\Users\\josez\\Documents\\DisenoDeInterfaz\\brats_data\\T1wCE\\Image-15.dcm"
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = cargar_modelo(MODEL_PATH, device=DEVICE)
    img, mascara, mascara_prob = predecir_imagen(
        modelo, IMAGE_PATH, device=DEVICE, threshold=0.5
    )
    visualizar_prediccion(img, mascara, mascara_prob)
