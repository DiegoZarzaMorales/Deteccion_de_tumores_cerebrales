# ============================================================
# INTERFAZ VISUAL PROFESIONAL
# Sistema de Deteccion de Tumores Cerebrales
# ============================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pydicom
import torch

from core.procesamiento import procesar_imagenes, etiquetar_tumores, guardar_base_datos
from core.clasificacion import entrenar_clasificador
from core.visualizacion import visualizar_deteccion
from core.carga_datos import extraer_datos, buscar_carpetas_t1
from ml.predecir import (
    cargar_modelo as cargar_modelo_ia,
    predecir_imagen as predecir_imagen_ia,
    predecir_carpeta as predecir_carpeta_ia,
    visualizar_prediccion as visualizar_prediccion_ia,
)



class SistemaTumoresGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Deteccion y Visualizacion de Tumores Cerebrales")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2c3e50")

        self.zip_path = None
        self.df = None
        self.imagenes_guardadas = []
        self.imagen_actual = 0
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # script_dir debe apuntar a la raíz del proyecto para rutas de modelos/predicciones
        self.proyecto_dir = os.path.dirname(self.script_dir)
        self.modelo_ia = None
        self.modelo_yolo = None
        self.device_ia = "cuda" if torch.cuda.is_available() else "cpu"

        self.crear_interfaz()

    def crear_interfaz(self):
        """Crea la interfaz grafica completa"""

        frame_titulo = tk.Frame(self.root, bg="#34495e", height=80)
        frame_titulo.pack(fill=tk.X, padx=10, pady=10)

        titulo = tk.Label(
            frame_titulo,
            text="SISTEMA DE DETECCION DE TUMORES CEREBRALES",
            font=("Arial", 24, "bold"),
            bg="#34495e",
            fg="white",
        )
        titulo.pack(pady=20)

        frame_central = tk.Frame(self.root, bg="#2c3e50")
        frame_central.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.crear_panel_controles(frame_central)
        self.crear_panel_visualizacion(frame_central)
        self.crear_barra_estado()

    def crear_panel_controles(self, parent):
        frame_controles = tk.Frame(parent, bg="#34495e", width=350)
        frame_controles.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        frame_controles.pack_propagate(False)

        titulo_panel = tk.Label(
            frame_controles,
            text="PANEL DE CONTROL",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white",
        )
        titulo_panel.pack(pady=20)

        ttk.Separator(frame_controles, orient="horizontal").pack(
            fill=tk.X, padx=20, pady=10
        )

        self.btn_cargar = tk.Button(
            frame_controles,
            text="CARGAR BASE DE DATOS",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            cursor="hand2",
            command=self.cargar_datos,
            height=2,
        )
        self.btn_cargar.pack(pady=10, padx=20, fill=tk.X)

        self.btn_procesar = tk.Button(
            frame_controles,
            text="PROCESAR IMAGENES",
            font=("Arial", 12, "bold"),
            bg="#2ecc71",
            fg="white",
            activebackground="#27ae60",
            activeforeground="white",
            cursor="hand2",
            command=self.procesar_datos,
            height=2,
            state=tk.DISABLED,
        )
        self.btn_procesar.pack(pady=10, padx=20, fill=tk.X)

        self.btn_clasificar = tk.Button(
            frame_controles,
            text="CLASIFICAR TUMORES",
            font=("Arial", 12, "bold"),
            bg="#9b59b6",
            fg="white",
            activebackground="#8e44ad",
            activeforeground="white",
            cursor="hand2",
            command=self.clasificar_datos,
            height=2,
            state=tk.DISABLED,
        )
        self.btn_clasificar.pack(pady=10, padx=20, fill=tk.X)

        self.btn_ia_imagen = tk.Button(
            frame_controles,
            text="IA: ANALIZAR IMAGEN T1wCE",
            font=("Arial", 12, "bold"),
            bg="#f1c40f",
            fg="black",
            activebackground="#f39c12",
            activeforeground="black",
            cursor="hand2",
            command=self.ia_analizar_imagen,
            height=2,
        )
        self.btn_ia_imagen.pack(pady=10, padx=20, fill=tk.X)

        self.btn_ia_carpeta = tk.Button(
            frame_controles,
            text="IA: ANALIZAR CARPETA T1wCE",
            font=("Arial", 12, "bold"),
            bg="#f1c40f",
            fg="black",
            activebackground="#f39c12",
            activeforeground="black",
            cursor="hand2",
            command=self.ia_analizar_carpeta,
            height=2,
        )
        self.btn_ia_carpeta.pack(pady=5, padx=20, fill=tk.X)

        ttk.Separator(frame_controles, orient="horizontal").pack(
            fill=tk.X, padx=20, pady=20
        )

        frame_nav = tk.Frame(frame_controles, bg="#34495e")
        frame_nav.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(
            frame_nav,
            text="NAVEGACION DE IMAGENES",
            font=("Arial", 11, "bold"),
            bg="#34495e",
            fg="white",
        ).pack(pady=(0, 10))

        frame_botones_nav = tk.Frame(frame_nav, bg="#34495e")
        frame_botones_nav.pack()

        self.btn_anterior = tk.Button(
            frame_botones_nav,
            text="◄ ANTERIOR",
            font=("Arial", 10, "bold"),
            bg="#e74c3c",
            fg="white",
            cursor="hand2",
            command=self.imagen_anterior,
            state=tk.DISABLED,
            width=12,
        )
        self.btn_anterior.pack(side=tk.LEFT, padx=5)

        self.btn_siguiente = tk.Button(
            frame_botones_nav,
            text="SIGUIENTE ►",
            font=("Arial", 10, "bold"),
            bg="#e74c3c",
            fg="white",
            cursor="hand2",
            command=self.imagen_siguiente,
            state=tk.DISABLED,
            width=12,
        )
        self.btn_siguiente.pack(side=tk.LEFT, padx=5)

        self.label_contador = tk.Label(
            frame_nav,
            text="Imagen 0 de 0",
            font=("Arial", 10),
            bg="#34495e",
            fg="white",
        )
        self.label_contador.pack(pady=10)

        ttk.Separator(frame_controles, orient="horizontal").pack(
            fill=tk.X, padx=20, pady=20
        )

        self.crear_panel_estadisticas(frame_controles)

        self.progress = ttk.Progressbar(
            frame_controles, mode="indeterminate", length=300
        )
        self.progress.pack(pady=20, padx=20)

    def crear_panel_estadisticas(self, parent):
        frame_stats = tk.Frame(parent, bg="#34495e")
        frame_stats.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(
            frame_stats,
            text="ESTADISTICAS",
            font=("Arial", 11, "bold"),
            bg="#34495e",
            fg="white",
        ).pack(pady=(0, 10))

        self.frame_stats_content = tk.Frame(
            frame_stats, bg="#2c3e50", relief=tk.SUNKEN, bd=2
        )
        self.frame_stats_content.pack(fill=tk.BOTH, expand=True)

        self.label_stats = tk.Label(
            self.frame_stats_content,
            text="No hay datos procesados",
            font=("Consolas", 9),
            bg="#2c3e50",
            fg="#ecf0f1",
            justify=tk.LEFT,
            anchor="nw",
            padx=10,
            pady=10,
        )
        self.label_stats.pack(fill=tk.BOTH, expand=True)

    def crear_panel_visualizacion(self, parent):
        frame_viz = tk.Frame(parent, bg="#34495e")
        frame_viz.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        titulo_viz = tk.Label(
            frame_viz,
            text="VISUALIZACION DE TUMORES",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="white",
        )
        titulo_viz.pack(pady=10)

        self.frame_canvas = tk.Frame(frame_viz, bg="white")
        self.frame_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.label_inicial = tk.Label(
            self.frame_canvas,
            text="Cargue y procese datos para visualizar tumores",
            font=("Arial", 14),
            bg="white",
            fg="#7f8c8d",
        )
        self.label_inicial.pack(expand=True)

    def crear_barra_estado(self):
        self.frame_estado = tk.Frame(self.root, bg="#34495e", height=40)
        self.frame_estado.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))

        self.label_estado = tk.Label(
            self.frame_estado,
            text="Estado: Listo para comenzar",
            font=("Arial", 10),
            bg="#34495e",
            fg="white",
            anchor="w",
        )
        self.label_estado.pack(side=tk.LEFT, padx=10, pady=5)

    def actualizar_estado(self, mensaje):
        self.label_estado.config(text=f"Estado: {mensaje}")
        self.root.update()

    # =============================
    # MODO IA SIMPLE (U-NET / YOLO)
    # =============================

    def ia_cargar_modelo(self):
        """Carga el modelo de IA entrenado si aún no está en memoria"""
        if self.modelo_ia is not None:
            return True

        model_path = os.path.join(self.proyecto_dir, "models", "best_model.pth")
        if not os.path.exists(model_path):
            messagebox.showerror(
                "Modelo no encontrado",
                "No se encontró el modelo entrenado en 'models/best_model.pth'.\n"
                "Primero ejecuta el entrenamiento (entrenar.py).",
            )
            return False

        try:
            self.actualizar_estado("Cargando modelo IA...")
            self.modelo_ia = cargar_modelo_ia(model_path, device=self.device_ia)
            self.actualizar_estado("Modelo IA listo")
            return True
        except Exception as e:
            messagebox.showerror("Error IA", f"No se pudo cargar el modelo IA:\n{e}")
            self.actualizar_estado("Error al cargar modelo IA")
            return False

    def ia_analizar_imagen(self):
        # Usar siempre U-Net (YOLO removido)
        if not self.ia_cargar_modelo():
            return

        img_path = filedialog.askopenfilename(
            title="Selecciona imagen DICOM T1wCE",
            filetypes=[("DICOM", "*.dcm"), ("Todos los archivos", "*.*")],
        )

        if not img_path:
            return

        try:
            modelo_txt = "U-Net"
            self.actualizar_estado(f"Analizando imagen con {modelo_txt}...")

            img, mascara, mascara_prob = predecir_imagen_ia(
                self.modelo_ia,
                img_path,
                device=self.device_ia,
                threshold=0.5,
            )

            visualizar_prediccion_ia(img, mascara, mascara_prob)
            self.actualizar_estado(f"Análisis con {modelo_txt} completado")
        except Exception as e:
            messagebox.showerror("Error IA", f"Error al analizar imagen:\n{e}")
            self.actualizar_estado("Error en análisis IA")

    def ia_analizar_carpeta(self):
        # Usar siempre U-Net (YOLO removido)
        if not self.ia_cargar_modelo():
            return

        carpeta = filedialog.askdirectory(
            title="Selecciona carpeta con imágenes T1wCE (.dcm)",
        )

        if not carpeta:
            return

        try:
            modelo_txt = "U-Net"
            self.actualizar_estado(f"Analizando carpeta con {modelo_txt}...")

            predecir_carpeta_ia(
                self.modelo_ia,
                carpeta,
                output_dir=os.path.join(self.proyecto_dir, "predicciones"),
                device=self.device_ia,
            )

            self.actualizar_estado(f"Análisis de carpeta con {modelo_txt} completado")
        except Exception as e:
            messagebox.showerror("Error IA", f"Error al analizar carpeta:\n{e}")
            self.actualizar_estado("Error en análisis IA de carpeta")

    def ia_cargar_modelo_yolo(self):
        """Carga el modelo YOLO26 si aún no está en memoria."""
        if self.modelo_yolo is not None:
            return True
        return False

    # =============================
    # PIPELINE CLÁSICO
    # =============================

    def cargar_datos(self):
        self.zip_path = filedialog.askopenfilename(
            title="Selecciona el archivo ZIP con 'Base de datos Brats'",
            filetypes=[("Archivos ZIP", "*.zip"), ("Todos los archivos", "*.*")],
        )

        if self.zip_path:
            self.actualizar_estado(
                f"Archivo cargado: {os.path.basename(self.zip_path)}"
            )
            self.btn_procesar.config(state=tk.NORMAL)
            messagebox.showinfo(
                "Exito",
                "Archivo ZIP cargado correctamente.\n"
                "Haga clic en 'PROCESAR IMAGENES' para continuar.",
            )
        else:
            self.actualizar_estado("Carga cancelada")

    def procesar_datos(self):
        thread = threading.Thread(target=self._procesar_datos_thread)
        thread.daemon = True
        thread.start()

    def _procesar_datos_thread(self):
        try:
            self.progress.start()
            self.actualizar_estado("Extrayendo datos...")

            extract_dir = extraer_datos(self.zip_path, self.proyecto_dir)
            t1_dirs = buscar_carpetas_t1(extract_dir)

            if len(t1_dirs) == 0:
                messagebox.showerror("Error", "No se encontraron carpetas T1wCE")
                return

            self.actualizar_estado(f"Procesando {len(t1_dirs)} pacientes...")

            self.df, self.imagenes_guardadas = procesar_imagenes(t1_dirs)
            self.df, self.mediana_area = etiquetar_tumores(self.df)
            guardar_base_datos(self.df, self.proyecto_dir)

            self.progress.stop()
            self.actualizar_estado("Procesamiento completado")

            self.actualizar_estadisticas()

            self.btn_clasificar.config(state=tk.NORMAL)
            self.btn_anterior.config(state=tk.NORMAL)
            self.btn_siguiente.config(state=tk.NORMAL)

            self.imagen_actual = 0
            self.mostrar_imagen_actual()

            messagebox.showinfo(
                "Exito",
                f"Se procesaron {len(self.df)} imagenes correctamente!",
            )

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Error al procesar datos: {str(e)}")
            self.actualizar_estado("Error en el procesamiento")

    def clasificar_datos(self):
        try:
            self.progress.start()
            self.actualizar_estado("Clasificando tumores...")

            modelo, accuracy, importancias = entrenar_clasificador(self.df)

            self.progress.stop()
            self.actualizar_estado(
                f"Clasificacion completada - Accuracy: {accuracy:.2%}"
            )

            self.actualizar_estadisticas(accuracy)

            messagebox.showinfo(
                "Exito",
                "Clasificacion completada!\n\n"
                f"Accuracy: {accuracy:.2%}\n"
                f"Feature mas importante: {importancias.iloc[0]['Característica']}",
            )

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Error en clasificacion: {str(e)}")

    def actualizar_estadisticas(self, accuracy=None):
        if self.df is not None:
            stats_text = f"""
IMAGENES PROCESADAS: {len(self.df)}

AREA TUMORAL:
  Promedio: {self.df['AreaTumor'].mean():.0f} px
  Mediana: {self.mediana_area:.0f} px
  Max: {self.df['AreaTumor'].max():.0f} px
  Min: {self.df['AreaTumor'].min():.0f} px

REGIONES:
  Promedio: {self.df['NumRegiones'].mean():.1f}

DISTRIBUCION:
  Tumores pequeños: {(self.df['TumorGrande']==0).sum()}
  Tumores grandes: {(self.df['TumorGrande']==1).sum()}
"""
            if accuracy:
                stats_text += f"\nACCURACY ML: {accuracy:.2%}"

            self.label_stats.config(text=stats_text)

    def mostrar_imagen_actual(self):
        if not self.imagenes_guardadas:
            return

        for widget in self.frame_canvas.winfo_children():
            widget.destroy()

        img, paciente, archivo, stats = self.imagenes_guardadas[self.imagen_actual]

        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Paciente: {paciente} | Archivo: {archivo}")
        ax.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        self.label_contador.config(
            text=f"Imagen {self.imagen_actual + 1} de {len(self.imagenes_guardadas)}"
        )

    def imagen_anterior(self):
        if self.imagen_actual > 0:
            self.imagen_actual -= 1
            self.mostrar_imagen_actual()

    def imagen_siguiente(self):
        if self.imagen_actual < len(self.imagenes_guardadas) - 1:
            self.imagen_actual += 1
            self.mostrar_imagen_actual()


def main():
    root = tk.Tk()
    app = SistemaTumoresGUI(root)
    root.mainloop()
