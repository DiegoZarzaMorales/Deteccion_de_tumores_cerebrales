# ============================================================
# DETECCIÓN Y VISUALIZACIÓN DE TUMORES CEREBRALES
# Enfoque: Localización visual de regiones tumorales
# Autores: Narvaez, Ochoa, Zarza
# ============================================================

# ------------------------------------------------------------
# 0) INSTALACIÓN DE DEPENDENCIAS
# ------------------------------------------------------------
# Ejecutar en terminal: pip install pydicom scikit-image scikit-learn matplotlib seaborn opencv-python

# ------------------------------------------------------------
# 1) IMPORTACIÓN DE LIBRERÍAS
# ------------------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox
import zipfile
import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, label, binary_erosion, binary_dilation
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
import warnings
warnings.filterwarnings('ignore')

# Configuración visual
plt.rcParams['figure.dpi'] = 100
sns.set_style("dark")

print("="*60)
print("SISTEMA DE DETECCIÓN Y VISUALIZACIÓN DE TUMORES CEREBRALES")
print("="*60)

# ------------------------------------------------------------
# 2) CARGA DE DATOS
# ------------------------------------------------------------
print("\n📂 PASO 1: Carga de datos")
print("Se abrirá una ventana para seleccionar el archivo ZIP...")

# Crear ventana oculta de tkinter
root = tk.Tk()
root.withdraw()  # Ocultar ventana principal
root.attributes('-topmost', True)  # Poner diálogo al frente

# Abrir diálogo visual para seleccionar archivo ZIP
zip_name = filedialog.askopenfilename(
    title="Selecciona el archivo ZIP con 'Base de datos Brats'",
    filetypes=[("Archivos ZIP", "*.zip"), ("Todos los archivos", "*.*")]
)

if not zip_name:
    messagebox.showerror("Error", "No se seleccionó ningún archivo")
    print("❌ No se seleccionó ningún archivo. Programa cancelado.")
    exit()

print(f"✅ Archivo seleccionado: {zip_name}")

# Directorio de extracción en la misma carpeta del script
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(script_dir, "brats_data")
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall(extract_dir)

print(f"✅ Datos extraídos en: {extract_dir}")

# Buscar carpetas T1wCE
base_dir = os.path.join(extract_dir, "Base de datos Brats")
t1_dirs = []

for root, dirs, files in os.walk(base_dir):
    if "T1wCE" in dirs:
        t1_dirs.append(os.path.join(root, "T1wCE"))

print(f"✅ Se encontraron {len(t1_dirs)} pacientes con imágenes T1wCE")

# ------------------------------------------------------------
# 3) FUNCIÓN DE DETECCIÓN DE TUMOR
# ------------------------------------------------------------
def detectar_tumor(img, return_mask=False):
    """
    Detecta regiones tumorales en una imagen cerebral

    Returns:
        tumor_mask: Máscara binaria de región tumoral
        contours: Contornos de las regiones detectadas
        stats: Estadísticas del tumor detectado
    """
    # Normalizar imagen
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Suavizado para reducir ruido
    img_smooth = gaussian(img_norm, sigma=1.5)

    # Umbral automático con Otsu
    try:
        threshold = threshold_otsu(img_smooth)
    except:
        threshold = img_smooth.mean() + 0.75 * img_smooth.std()

    # Crear máscara binaria
    tumor_mask = img_smooth > threshold

    # Morfología: limpiar pequeños objetos
    tumor_mask = remove_small_objects(tumor_mask, min_size=50)

    # Dilatar ligeramente para conectar regiones cercanas (aplicar 2 veces)
    from skimage.morphology import disk
    footprint = disk(1)
    tumor_mask = binary_dilation(tumor_mask, footprint=footprint)
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

# ------------------------------------------------------------
# 4) FUNCIÓN DE VISUALIZACIÓN MEJORADA
# ------------------------------------------------------------
def visualizar_deteccion(img_original, titulo="Detección de Tumor"):
    """
    Crea visualización completa de la detección tumoral
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
    📊 ESTADÍSTICAS:
    • Regiones detectadas: {stats['num_regions']}
    • Área total: {stats['total_area']:.0f} píxeles
    • Intensidad media: {stats['mean_intensity']:.3f}
    • Intensidad máxima: {stats['max_intensity']:.3f}
    """
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    return stats

# ------------------------------------------------------------
# 5) PROCESAMIENTO Y EXTRACCIÓN DE FEATURES
# ------------------------------------------------------------
print("\n🔍 PASO 2: Procesamiento de imágenes y detección de tumores")

rows = []
imagenes_procesadas = 0
imagenes_guardadas = []  # Para visualizaciones posteriores

for carpeta in t1_dirs:
    paciente = os.path.basename(os.path.dirname(carpeta))
    print(f"\n📁 Procesando paciente: {paciente}")

    archivos_dcm = [f for f in os.listdir(carpeta) if f.lower().endswith(".dcm")]

    for idx, archivo in enumerate(archivos_dcm[:10]):  # Limitar para demo
        ruta = os.path.join(carpeta, archivo)

        try:
            ds = pydicom.dcmread(ruta, force=True)
            img = ds.pixel_array.astype("float32")
        except:
            continue

        if img.max() == 0:
            continue

        # Detectar tumor
        contours, stats = detectar_tumor(img)

        # Normalización
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Features básicos
        meanI = float(img_norm.mean())
        stdI = float(img_norm.std())

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

        rows.append({
            "Paciente": paciente,
            "Archivo": archivo,
            "MeanIntensity": meanI,
            "StdIntensity": stdI,
            "NumRegiones": num_regiones,
            "AreaTumor": area_tumor,
            "IntensidadTumor": intensidad_tumor,
            "CentroY": centro_y,
            "CentroX": centro_x,
            "TotalPixels": img.size
        })

        # Guardar algunas imágenes para visualización
        if len(imagenes_guardadas) < 6:
            imagenes_guardadas.append((img, paciente, archivo, stats))

        imagenes_procesadas += 1

print(f"\n✅ Total de imágenes procesadas: {imagenes_procesadas}")

# Crear DataFrame
df = pd.DataFrame(rows)

# Etiquetado basado en área del tumor
mediana_area = df["AreaTumor"].median()
df["TumorGrande"] = (df["AreaTumor"] > mediana_area).astype(int)

print(f"\n📊 Mediana de área tumoral: {mediana_area:.0f} píxeles")
print(f"Distribución: {df['TumorGrande'].value_counts().to_dict()}")

# Guardar base de datos
df.to_csv("basededatos_deteccion.csv", index=False)
print("\n✅ Base de datos guardada: basededatos_deteccion.csv")

# ------------------------------------------------------------
# 6) VISUALIZACIONES DE DETECCIÓN INDIVIDUAL
# ------------------------------------------------------------
print("\n" + "="*60)
print("📸 PASO 3: Visualizaciones de detección tumoral")
print("="*60)

for idx, (img, paciente, archivo, stats) in enumerate(imagenes_guardadas[:3]):
    print(f"\n🔬 Visualización {idx+1}/3: {paciente} - {archivo}")
    visualizar_deteccion(img, titulo=f"{paciente} - {archivo}")

# ------------------------------------------------------------
# 7) ANÁLISIS ESPACIAL DE TUMORES
# ------------------------------------------------------------
print("\n" + "="*60)
print("🗺️ PASO 4: Análisis espacial de tumores")
print("="*60)

# Mapa de calor de ubicaciones tumorales
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot de centroides
scatter = axes[0].scatter(df['CentroX'], df['CentroY'],
                          c=df['TumorGrande'], cmap='coolwarm',
                          s=df['AreaTumor']/10, alpha=0.6, edgecolors='black')
axes[0].set_xlabel('Posición X (píxeles)', fontsize=12)
axes[0].set_ylabel('Posición Y (píxeles)', fontsize=12)
axes[0].set_title('Localización Espacial de Tumores', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()  # Invertir Y para coincidir con coordenadas de imagen
axes[0].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Tipo (0=Pequeño, 1=Grande)', fontsize=10)

# Histograma 2D (densidad de tumores)
h = axes[1].hist2d(df['CentroX'], df['CentroY'], bins=20, cmap='hot')
axes[1].set_xlabel('Posición X (píxeles)', fontsize=12)
axes[1].set_ylabel('Posición Y (píxeles)', fontsize=12)
axes[1].set_title('Mapa de Densidad de Tumores', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
plt.colorbar(h[3], ax=axes[1], label='Frecuencia')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8) DISTRIBUCIÓN DE CARACTERÍSTICAS TUMORALES
# ------------------------------------------------------------
print("\n📊 PASO 5: Análisis de características tumorales")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribución de área tumoral
axes[0, 0].hist(df['AreaTumor'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(mediana_area, color='red', linestyle='--', linewidth=2,
                   label=f'Mediana = {mediana_area:.0f}')
axes[0, 0].set_xlabel('Área del Tumor (píxeles)', fontsize=11)
axes[0, 0].set_ylabel('Frecuencia', fontsize=11)
axes[0, 0].set_title('Distribución del Área Tumoral', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Número de regiones detectadas
region_counts = df['NumRegiones'].value_counts().sort_index()
axes[0, 1].bar(region_counts.index, region_counts.values, color='coral', edgecolor='black')
axes[0, 1].set_xlabel('Número de Regiones', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].set_title('Distribución de Regiones Tumorales', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Relación área vs intensidad
scatter = axes[1, 0].scatter(df['AreaTumor'], df['IntensidadTumor'],
                             c=df['TumorGrande'], cmap='viridis',
                             s=100, alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('Área del Tumor (píxeles)', fontsize=11)
axes[1, 0].set_ylabel('Intensidad Media del Tumor', fontsize=11)
axes[1, 0].set_title('Relación Área vs Intensidad', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Tipo de Tumor')

# 4. Boxplot comparativo
data_small = df[df['TumorGrande']==0]['AreaTumor']
data_large = df[df['TumorGrande']==1]['AreaTumor']
bp = axes[1, 1].boxplot([data_small, data_large], labels=['Tumor Pequeño', 'Tumor Grande'],
                        patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
axes[1, 1].set_ylabel('Área del Tumor (píxeles)', fontsize=11)
axes[1, 1].set_title('Comparación de Área por Tipo', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9) CLASIFICACIÓN CON MACHINE LEARNING
# ------------------------------------------------------------
print("\n" + "="*60)
print("🤖 PASO 6: Clasificación con Machine Learning")
print("="*60)

features = ["MeanIntensity", "StdIntensity", "NumRegiones",
            "AreaTumor", "IntensidadTumor"]

X = df[features]
y = df["TumorGrande"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)

print(f"\n✅ Random Forest Accuracy: {acc_rf:.2%}")
print("\nReporte de clasificación:")
print(classification_report(y_test, pred_rf, target_names=['Tumor Pequeño', 'Tumor Grande']))

# Importancia de características
importancias = pd.DataFrame({
    'Característica': features,
    'Importancia': rf.feature_importances_
}).sort_values('Importancia', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if 'Area' in x else '#3498db' for x in importancias['Característica']]
ax.barh(importancias['Característica'], importancias['Importancia'], color=colors, edgecolor='black')
ax.set_xlabel('Importancia', fontsize=12)
ax.set_title('Importancia de Características para Detección', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10) VISUALIZACIÓN COMPARATIVA: 3 CASOS
# ------------------------------------------------------------
print("\n" + "="*60)
print("🔬 PASO 7: Casos representativos")
print("="*60)

# Seleccionar 3 casos: tumor pequeño, mediano, grande
df_sorted = df.sort_values('AreaTumor')
idx_small = df_sorted.index[len(df_sorted)//4]
idx_medium = df_sorted.index[len(df_sorted)//2]
idx_large = df_sorted.index[3*len(df_sorted)//4]

casos_interes = [idx_small, idx_medium, idx_large]
titulos = ['TUMOR PEQUEÑO', 'TUMOR MEDIANO', 'TUMOR GRANDE']

for caso_idx, titulo in zip(casos_interes, titulos):
    if caso_idx < len(imagenes_guardadas):
        img, pac, arch, stats = imagenes_guardadas[caso_idx]
        print(f"\n{titulo}: {pac} - Área: {df.loc[caso_idx, 'AreaTumor']:.0f} píxeles")
        visualizar_deteccion(img, titulo=f"{titulo} - {pac}")

# ------------------------------------------------------------
# 11) RESUMEN FINAL
# ------------------------------------------------------------
print("\n" + "="*60)
print("✅ PROCESO COMPLETADO")
print("="*60)

print(f"""
📊 RESUMEN DE RESULTADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Imágenes procesadas:        {imagenes_procesadas}
🎯 Accuracy clasificación:     {acc_rf:.2%}
📏 Área tumoral promedio:      {df['AreaTumor'].mean():.0f} píxeles
🔢 Regiones promedio/imagen:   {df['NumRegiones'].mean():.1f}
💡 Feature más importante:     {importancias.iloc[0]['Característica']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Archivos generados:
   • basededatos_deteccion.csv

🎯 Características detectadas:
   • Localización espacial del tumor
   • Área y extensión tumoral
   • Intensidad de regiones tumorales
   • Número de focos tumorales

💡 Próximos pasos sugeridos:
   1. Validación con segmentaciones manuales
   2. Análisis volumétrico 3D
   3. Integración con datos clínicos
   4. Desarrollo de interfaz interactiva
""")

print("\n🎓 Listo para tu presentación!")