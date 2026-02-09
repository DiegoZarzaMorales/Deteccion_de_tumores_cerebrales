# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap

from core.deteccion import detectar_tumor


def visualizar_deteccion(img_original, titulo="Detección de Tumor"):
    """Crea visualización completa de la detección tumoral"""
    # Normalizar imagen original
    img_norm = (img_original - img_original.min()) / (
        img_original.max() - img_original.min() + 1e-8
    )

    # Detectar tumor
    tumor_mask, contours, stats = detectar_tumor(img_original, return_mask=True)

    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(titulo, fontsize=16, fontweight="bold", y=0.995)

    # 1. Imagen original
    axes[0, 0].imshow(img_norm, cmap="gray")
    axes[0, 0].set_title("Imagen Original T1wCE", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Imagen con contornos de tumor superpuestos
    img_rgb = np.stack([img_norm] * 3, axis=-1)
    for contour in contours:
        cv2.drawContours(img_rgb, [contour], -1, (1, 0, 0), 2)  # Rojo
    axes[0, 1].imshow(img_rgb)
    axes[0, 1].set_title(
        f"Detección de Tumor ({stats['num_regions']} regiones)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].axis("off")

    # 3. Mapa de calor de intensidad
    cmap_heat = LinearSegmentedColormap.from_list(
        "tumor", ["black", "blue", "cyan", "yellow", "red"]
    )
    im = axes[1, 0].imshow(img_norm, cmap=cmap_heat)
    axes[1, 0].set_title("Mapa de Calor - Intensidad", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Máscara de segmentación
    overlay = np.zeros_like(img_rgb)
    overlay[:, :, 0] = tumor_mask  # Canal rojo para tumor
    blended = img_rgb * 0.6 + overlay * 0.4
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title("Segmentación Tumoral (overlay)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # Agregar estadísticas como texto
    stats_text = f"""
    ESTADISTICAS:
    • Regiones detectadas: {stats['num_regions']}
    • Area total: {stats['total_area']:.0f} pixeles
    • Intensidad media: {stats['mean_intensity']:.3f}
    • Intensidad maxima: {stats['max_intensity']:.3f}
    """
    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()

    return stats
