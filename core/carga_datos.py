# ============================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================

import os
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox


def seleccionar_archivo_zip():
    """Abre un diálogo visual para seleccionar el archivo ZIP"""
    print("\n" + "="*60)
    print("PASO 1: Selección de datos")
    print("="*60)
    print("\nAbriendo ventana de selección de archivo...")

    # Crear ventana oculta de tkinter
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal
    root.attributes("-topmost", True)  # Poner diálogo al frente

    # Abrir diálogo visual para seleccionar archivo ZIP
    zip_name = filedialog.askopenfilename(
        title="Selecciona el archivo ZIP con 'Base de datos Brats'",
        filetypes=[("Archivos ZIP", "*.zip"), ("Todos los archivos", "*.*")],
    )

    root.destroy()  # Destruir ventana temporal

    if not zip_name:
        messagebox.showerror("Error", "No se seleccionó ningún archivo")
        print("\nNo se seleccionó ningún archivo. Programa cancelado.")
        return None

    print(f"\nArchivo seleccionado: {zip_name}")
    return zip_name


def extraer_datos(zip_name, base_dir):
    """Extrae el archivo ZIP en el directorio 'brats_data' en la raíz"""
    extract_dir = os.path.join(base_dir, "brats_data")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"\nExtrayendo datos en: {extract_dir}")

    with zipfile.ZipFile(zip_name, "r") as z:
        z.extractall(extract_dir)

    print("Datos extraídos exitosamente")

    return extract_dir


def buscar_carpetas_t1(extract_dir):
    """Busca las carpetas T1wCE en el directorio extraído"""
    base_dir = os.path.join(extract_dir, "Base de datos Brats")
    t1_dirs = []

    print("\nBuscando imágenes T1wCE...")

    for root, dirs, files in os.walk(base_dir):
        if "T1wCE" in dirs:
            t1_dirs.append(os.path.join(root, "T1wCE"))

    print(f"Se encontraron {len(t1_dirs)} pacientes con imágenes T1wCE")

    return t1_dirs
