# ============================================================
# PREDICCIÓN CON MODELO ENTRENADO
# ============================================================

import torch
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from .modelo_unet import UNet


def cargar_modelo(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Carga un modelo entrenado"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    in_channels = int(checkpoint.get('in_channels', 1))
    out_channels = int(checkpoint.get('out_channels', 1))
    modelo = UNet(in_channels=in_channels, out_channels=out_channels)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo = modelo.to(device)
    modelo.eval()
    
    print(f"✓ Modelo cargado desde: {model_path}")
    print(f"  Canales de entrada: {in_channels}")
    print(f"  Clases de salida: {out_channels}")
    if 'dice' in checkpoint:
        print(f"  Dice Score: {checkpoint['dice']:.4f}")
    
    return modelo


def _tensor_entrada_desde_2d(img_resized, in_channels):
    """Empaqueta una imagen 2D al número de canales que espera el modelo."""
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # (1, H, W)
    if in_channels > 1:
        # Fallback DICOM mono-canal: repetir canales (mejor usar HDF5 multimodal).
        img_tensor = img_tensor.repeat(in_channels, 1, 1)
    return img_tensor.unsqueeze(0)  # (1, C, H, W)


def _postproceso_componente_principal(img_original, mascara_prob):
    """Conserva el componente conexo más coherente con la zona brillante."""
    img_norm_full = (img_original - img_original.min()) / (
        img_original.max() - img_original.min() + 1e-8
    )
    combinado = mascara_prob * img_norm_full
    max_val = combinado.max()
    if max_val <= 0:
        return np.zeros_like(combinado, dtype=np.uint8)

    max_idx_flat = np.argmax(combinado)
    max_y, max_x = np.unravel_index(max_idx_flat, combinado.shape)
    thr = max_val * 0.7
    mascara_candidata = (combinado >= thr).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        mascara_candidata, connectivity=8
    )
    label_seed = labels[max_y, max_x]
    if label_seed == 0:
        return np.zeros_like(combinado, dtype=np.uint8)
    return (labels == label_seed).astype(np.uint8)


def predecir_imagen(modelo, img_path, size=256, device='cpu', threshold=0.5):
    """Realiza predicción en una imagen DICOM.

    Returns:
        img_original, mascara_binaria (WT), mapa_probabilidad_o_clases
    """
    dcm = pydicom.dcmread(img_path)
    img = dcm.pixel_array.astype(np.float32)
    img_original = img.copy()
    
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_resized = cv2.resize(img_norm, (size, size))

    in_channels = getattr(modelo, 'in_channels', 1)
    out_channels = getattr(modelo, 'out_channels', 1)
    img_tensor = _tensor_entrada_desde_2d(img_resized.astype(np.float32), in_channels)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        prediccion = modelo(img_tensor)

    if out_channels > 1:
        clases = torch.argmax(prediccion, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        clases_full = cv2.resize(
            clases,
            (img_original.shape[1], img_original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mascara_binaria = (clases_full > 0).astype(np.uint8)
        return img_original, mascara_binaria, clases_full.astype(np.float32)

    prediccion = torch.sigmoid(prediccion)
    mascara = prediccion.squeeze().cpu().numpy()
    mascara_full = cv2.resize(mascara, (img_original.shape[1], img_original.shape[0]))
    mascara_binaria = _postproceso_componente_principal(img_original, mascara_full)
    return img_original, mascara_binaria, mascara_full


def visualizar_prediccion(img, mascara, mascara_prob=None, save_path=None):
    """Visualiza la predicción"""
    fig, axes = plt.subplots(1, 3 if mascara_prob is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(mascara, cmap='Reds', alpha=0.5)
    axes[1].set_title('Predicción (Tumor Detectado)')
    axes[1].axis('off')
    
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
    """Predice todas las imágenes en una carpeta"""
    carpeta_path = Path(carpeta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    archivos = list(carpeta_path.glob('*.dcm'))
    print(f"\nEncontradas {len(archivos)} imágenes")
    
    for idx, archivo in enumerate(archivos):
        print(f"\nProcesando [{idx+1}/{len(archivos)}]: {archivo.name}")
        img, mascara, mascara_prob = predecir_imagen(
            modelo, archivo, device=device
        )
        save_path = output_dir / f"pred_{archivo.stem}.png"
        visualizar_prediccion(img, mascara, mascara_prob, save_path=save_path)
    
    print(f"\n✓ Todas las predicciones guardadas en: {output_dir}")


if __name__ == "__main__":
    import os

    MODEL_PATH = "models/best_model.pth"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_PATH = os.path.join(base_dir, "brats_data", "T1wCE", "Image-15.dcm")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    modelo = cargar_modelo(MODEL_PATH, device=DEVICE)
    img, mascara, mascara_prob = predecir_imagen(
        modelo, IMAGE_PATH, device=DEVICE, threshold=0.5
    )
    visualizar_prediccion(img, mascara, mascara_prob)
