# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap
from deteccion import detectar_tumor


def visualizar_deteccion(img_original, titulo="Detección de Tumor"):
    """
    Crea visualización completa de la detección tumoral

    Args:
        img_original: Imagen original
        titulo: Título de la visualización
    """
    # Normalizar imagen original
    img_norm = (img_original - img_original.min()) / (img_original.max() - img_original.min() + 1e-8)

    # Detectar tumor
    tumor_mask, contours, stats = detectar_tumor(img_original, return_mask=True)

    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=0.995)

    # 1. Imagen original
    axes[0, 0].imshow(img_norm, cmap='gray')
    axes[0, 0].set_title('Imagen Original T1wCE', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Imagen con contornos de tumor superpuestos
    img_rgb = np.stack([img_norm]*3, axis=-1)
    for contour in contours:
        cv2.drawContours(img_rgb, [contour], -1, (1, 0, 0), 2)  # Rojo
    axes[0, 1].imshow(img_rgb)
    axes[0, 1].set_title(f'Detección de Tumor ({stats["num_regions"]} regiones)',
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Mapa de calor de intensidad
    cmap_heat = LinearSegmentedColormap.from_list('tumor',
                                                   ['black', 'blue', 'cyan', 'yellow', 'red'])
    im = axes[1, 0].imshow(img_norm, cmap=cmap_heat)
    axes[1, 0].set_title('Mapa de Calor - Intensidad', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Máscara de segmentación
    overlay = np.zeros_like(img_rgb)
    overlay[:, :, 0] = tumor_mask  # Canal rojo para tumor
    blended = img_rgb * 0.6 + overlay * 0.4
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('Segmentación Tumoral (overlay)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Agregar estadísticas como texto
    stats_text = f"""
    ESTADISTICAS:
    • Regiones detectadas: {stats['num_regions']}
    • Area total: {stats['total_area']:.0f} pixeles
    • Intensidad media: {stats['mean_intensity']:.3f}
    • Intensidad maxima: {stats['max_intensity']:.3f}
    """
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    return stats


def visualizar_analisis_espacial(df):
    """
    Visualiza el análisis espacial de tumores
    
    Args:
        df: DataFrame con los datos procesados
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot de centroides
    scatter = axes[0].scatter(df['CentroX'], df['CentroY'],
                              c=df['TumorGrande'], cmap='coolwarm',
                              s=df['AreaTumor']/10, alpha=0.6, edgecolors='black')
    axes[0].set_xlabel('Posición X (pixeles)', fontsize=12)
    axes[0].set_ylabel('Posición Y (pixeles)', fontsize=12)
    axes[0].set_title('Localización Espacial de Tumores', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Tipo (0=Pequeño, 1=Grande)', fontsize=10)

    # Histograma 2D (densidad de tumores)
    h = axes[1].hist2d(df['CentroX'], df['CentroY'], bins=20, cmap='hot')
    axes[1].set_xlabel('Posición X (pixeles)', fontsize=12)
    axes[1].set_ylabel('Posición Y (pixeles)', fontsize=12)
    axes[1].set_title('Mapa de Densidad de Tumores', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    plt.colorbar(h[3], ax=axes[1], label='Frecuencia')

    plt.tight_layout()
    plt.show()


def visualizar_distribucion_caracteristicas(df, mediana_area):
    """
    Visualiza la distribución de características tumorales
    
    Args:
        df: DataFrame con los datos
        mediana_area: Valor de la mediana del área tumoral
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Distribución de área tumoral
    axes[0, 0].hist(df['AreaTumor'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(mediana_area, color='red', linestyle='--', linewidth=2,
                       label=f'Mediana = {mediana_area:.0f}')
    axes[0, 0].set_xlabel('Area del Tumor (pixeles)', fontsize=11)
    axes[0, 0].set_ylabel('Frecuencia', fontsize=11)
    axes[0, 0].set_title('Distribución del Area Tumoral', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Número de regiones detectadas
    region_counts = df['NumRegiones'].value_counts().sort_index()
    axes[0, 1].bar(region_counts.index, region_counts.values, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Numero de Regiones', fontsize=11)
    axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
    axes[0, 1].set_title('Distribución de Regiones Tumorales', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Relación área vs intensidad
    scatter = axes[1, 0].scatter(df['AreaTumor'], df['IntensidadTumor'],
                                 c=df['TumorGrande'], cmap='viridis',
                                 s=100, alpha=0.6, edgecolors='black')
    axes[1, 0].set_xlabel('Area del Tumor (pixeles)', fontsize=11)
    axes[1, 0].set_ylabel('Intensidad Media del Tumor', fontsize=11)
    axes[1, 0].set_title('Relación Area vs Intensidad', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Tipo de Tumor')

    # 4. Boxplot comparativo
    data_small = df[df['TumorGrande']==0]['AreaTumor']
    data_large = df[df['TumorGrande']==1]['AreaTumor']
    bp = axes[1, 1].boxplot([data_small, data_large], labels=['Tumor Pequeño', 'Tumor Grande'],
                            patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1, 1].set_ylabel('Area del Tumor (pixeles)', fontsize=11)
    axes[1, 1].set_title('Comparación de Area por Tipo', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def visualizar_importancia_features(importancias_df):
    """
    Visualiza la importancia de las características
    
    Args:
        importancias_df: DataFrame con las importancias de las características
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if 'Area' in x else '#3498db' for x in importancias_df['Característica']]
    ax.barh(importancias_df['Característica'], importancias_df['Importancia'], 
            color=colors, edgecolor='black')
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_title('Importancia de Características para Detección', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
