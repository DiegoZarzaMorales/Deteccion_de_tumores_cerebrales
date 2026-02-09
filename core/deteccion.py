# ============================================================
# FUNCIONES DE DETECCIÓN DE TUMORES
# ============================================================

import numpy as np
import cv2
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, label, binary_dilation, disk
from skimage.measure import regionprops
from core.config import PARAMETROS_DETECCION


def detectar_tumor(img, return_mask=False):
    """
    Detecta regiones tumorales en una imagen cerebral

    Args:
        img: Imagen de entrada
        return_mask: Si True, retorna también la máscara binaria

    Returns:
        tumor_mask: Máscara binaria de región tumoral (si return_mask=True)
        contours: Contornos de las regiones detectadas
        stats: Estadísticas del tumor detectado
    """
    # Normalizar imagen
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Suavizado para reducir ruido
    img_smooth = gaussian(img_norm, sigma=PARAMETROS_DETECCION['sigma_suavizado'])

    # Umbral automático con Otsu
    try:
        threshold = threshold_otsu(img_smooth)
    except Exception:
        threshold = img_smooth.mean() + 0.75 * img_smooth.std()

    # Crear máscara binaria
    tumor_mask = img_smooth > threshold

    # Morfología: limpiar pequeños objetos
    tumor_mask = remove_small_objects(tumor_mask, min_size=PARAMETROS_DETECCION['min_size_objeto'])

    # Dilatar ligeramente para conectar regiones cercanas
    footprint = disk(PARAMETROS_DETECCION['radio_dilatacion'])
    for _ in range(PARAMETROS_DETECCION['num_dilataciones']):
        tumor_mask = binary_dilation(tumor_mask, footprint=footprint)

    # Etiquetar regiones conectadas
    labeled_mask = label(tumor_mask)
    regions = regionprops(labeled_mask, intensity_image=img_norm)

    # Estadísticas
    stats = {
        'num_regions': len(regions),
        'total_area': np.sum(tumor_mask),
        'mean_intensity': img_norm[tumor_mask].mean() if np.sum(tumor_mask) > 0 else 0,
        'max_intensity': img_norm[tumor_mask].max() if np.sum(tumor_mask) > 0 else 0,
        'regions_info': []
    }

    # Información de cada región
    for region in regions:
        stats['regions_info'].append({
            'area': region.area,
            'centroid': region.centroid,
            'bbox': region.bbox,
            'mean_intensity': region.mean_intensity
        })

    # Encontrar contornos para visualización
    tumor_mask_uint8 = (tumor_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(tumor_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if return_mask:
        return tumor_mask, contours, stats
    else:
        return contours, stats


def extraer_features(img, stats):
    """Extrae características de la imagen y del tumor detectado"""
    # Normalización
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Features básicos
    mean_intensity = float(img_norm.mean())
    std_intensity = float(img_norm.std())

    # Features de detección tumoral
    num_regiones = stats['num_regions']
    area_tumor = stats['total_area']
    intensidad_tumor = stats['mean_intensity']

    # Localización del tumor (centroide promedio)
    if stats['regions_info']:
        centroides = [r['centroid'] for r in stats['regions_info']]
        centro_y = np.mean([c[0] for c in centroides])
        centro_x = np.mean([c[1] for c in centroides])
    else:
        centro_y, centro_x = 0, 0

    return {
        "MeanIntensity": mean_intensity,
        "StdIntensity": std_intensity,
        "NumRegiones": num_regiones,
        "AreaTumor": area_tumor,
        "IntensidadTumor": intensidad_tumor,
        "CentroY": centro_y,
        "CentroX": centro_x,
        "TotalPixels": img.size,
    }
