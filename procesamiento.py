# ============================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================

import os
import pydicom
import pandas as pd
from deteccion import detectar_tumor, extraer_features
from config import LIMITE_IMAGENES_POR_PACIENTE, NUM_IMAGENES_VISUALIZACION


def procesar_imagenes(t1_dirs):
    """
    Procesa todas las imágenes DICOM y extrae características
    
    Args:
        t1_dirs: Lista de directorios con imágenes T1wCE
        
    Returns:
        tuple: (DataFrame con datos, lista de imágenes guardadas)
    """
    print("\n" + "="*60)
    print("PASO 2: Procesamiento de imagenes y deteccion de tumores")
    print("="*60)
    
    rows = []
    imagenes_procesadas = 0
    imagenes_guardadas = []  # Para visualizaciones posteriores
    
    for carpeta in t1_dirs:
        paciente = os.path.basename(os.path.dirname(carpeta))
        print(f"\nProcesando paciente: {paciente}")
        
        archivos_dcm = [f for f in os.listdir(carpeta) if f.lower().endswith(".dcm")]
        
        for idx, archivo in enumerate(archivos_dcm[:LIMITE_IMAGENES_POR_PACIENTE]):
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
            
            # Extraer características
            features = extraer_features(img, stats)
            
            # Agregar información del paciente
            features["Paciente"] = paciente
            features["Archivo"] = archivo
            
            rows.append(features)
            
            # Guardar algunas imágenes para visualización
            if len(imagenes_guardadas) < NUM_IMAGENES_VISUALIZACION:
                imagenes_guardadas.append((img, paciente, archivo, stats))
            
            imagenes_procesadas += 1
    
    print(f"\nTotal de imagenes procesadas: {imagenes_procesadas}")
    
    # Crear DataFrame
    df = pd.DataFrame(rows)
    
    return df, imagenes_guardadas


def etiquetar_tumores(df):
    """
    Etiqueta los tumores como grandes o pequeños según la mediana
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        tuple: (DataFrame actualizado, valor de mediana)
    """
    mediana_area = df["AreaTumor"].median()
    df["TumorGrande"] = (df["AreaTumor"] > mediana_area).astype(int)
    
    print(f"\nMediana de area tumoral: {mediana_area:.0f} pixeles")
    print(f"Distribucion: {df['TumorGrande'].value_counts().to_dict()}")
    
    return df, mediana_area


def guardar_base_datos(df, script_dir):
    """
    Guarda el DataFrame en un archivo CSV
    
    Args:
        df: DataFrame a guardar
        script_dir: Directorio donde guardar el archivo
    """
    ruta_csv = os.path.join(script_dir, "basededatos_deteccion.csv")
    df.to_csv(ruta_csv, index=False)
    print(f"\nBase de datos guardada: {ruta_csv}")
