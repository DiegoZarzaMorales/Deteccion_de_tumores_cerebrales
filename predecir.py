# ============================================================
# PREDICCIÓN CON MODELO ENTRENADO
# ============================================================

import torch
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from modelo_unet import UNet


def cargar_modelo(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Carga un modelo entrenado
    
    Args:
        model_path: Ruta al archivo .pth
        device: Dispositivo
        
    Returns:
        modelo: Modelo cargado y listo para inferencia
    """
    modelo = UNet(in_channels=1, out_channels=1)
    
    checkpoint = torch.load(model_path, map_location=device)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo = modelo.to(device)
    modelo.eval()
    
    print(f"✓ Modelo cargado desde: {model_path}")
    if 'dice' in checkpoint:
        print(f"  Dice Score: {checkpoint['dice']:.4f}")
    
    return modelo


def predecir_imagen(modelo, img_path, size=256, device='cpu', threshold=0.5):
    """
    Realiza predicción en una imagen
    
    Args:
        modelo: Modelo entrenado
        img_path: Ruta a la imagen DICOM
        size: Tamaño de redimensionamiento
        device: Dispositivo
        threshold: Umbral para binarizar
        
    Returns:
        img_original: Imagen original
        mascara_predicha: Máscara predicha
    """
    # Cargar imagen
    dcm = pydicom.dcmread(img_path)
    img = dcm.pixel_array.astype(np.float32)
    img_original = img.copy()
    
    # Normalizar y redimensionar
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_resized = cv2.resize(img, (size, size))
    
    # Convertir a tensor
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_tensor = img_tensor.to(device)
    
    # Predicción
    with torch.no_grad():
        prediccion = modelo(img_tensor)
    
    # Convertir a numpy
    mascara = prediccion.squeeze().cpu().numpy()
    
    # Redimensionar a tamaño original
    mascara_full = cv2.resize(mascara, (img_original.shape[1], img_original.shape[0]))
    
    # Binarizar
    mascara_binaria = (mascara_full > threshold).astype(np.uint8)
    
    return img_original, mascara_binaria, mascara_full


def visualizar_prediccion(img, mascara, mascara_prob=None, save_path=None):
    """Visualiza la predicción"""
    fig, axes = plt.subplots(1, 3 if mascara_prob is not None else 2, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Máscara predicha
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(mascara, cmap='Reds', alpha=0.5)
    axes[1].set_title('Predicción (Tumor Detectado)')
    axes[1].axis('off')
    
    # Probabilidades
    if mascara_prob is not None:
        axes[2].imshow(mascara_prob, cmap='hot')
        axes[2].set_title('Mapa de Probabilidad')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualización guardada en: {save_path}")
    
    plt.show()


def predecir_carpeta(modelo, carpeta_path, output_dir='predicciones', device='cpu'):
    """
    Predice todas las imágenes en una carpeta
    
    Args:
        modelo: Modelo entrenado
        carpeta_path: Carpeta con imágenes DICOM
        output_dir: Directorio de salida
        device: Dispositivo
    """
    carpeta_path = Path(carpeta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Obtener archivos
    archivos = list(carpeta_path.glob("*.dcm"))
    print(f"\nEncontradas {len(archivos)} imágenes")
    
    for idx, archivo in enumerate(archivos):
        print(f"\nProcesando [{idx+1}/{len(archivos)}]: {archivo.name}")
        
        # Predecir
        img, mascara, mascara_prob = predecir_imagen(
            modelo, archivo, device=device
        )
        
        # Guardar visualización
        save_path = output_dir / f"pred_{archivo.stem}.png"
        visualizar_prediccion(img, mascara, mascara_prob, save_path=save_path)
    
    print(f"\n✓ Todas las predicciones guardadas en: {output_dir}")


if __name__ == "__main__":
    # Configuración
    MODEL_PATH = "models/best_model.pth"
    IMAGE_PATH = r"c:\Users\josez\Documents\DisenoDeInterfaz\brats_data\T1wCE\Image-15.dcm"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar modelo
    modelo = cargar_modelo(MODEL_PATH, device=DEVICE)
    
    # Predecir
    img, mascara, mascara_prob = predecir_imagen(
        modelo, IMAGE_PATH, device=DEVICE, threshold=0.5
    )
    
    # Visualizar
    visualizar_prediccion(img, mascara, mascara_prob)
    
    # O predecir carpeta completa
    # CARPETA = r"c:\Users\josez\Documents\DisenoDeInterfaz\brats_data\T1wCE"
    # predecir_carpeta(modelo, CARPETA, device=DEVICE)
