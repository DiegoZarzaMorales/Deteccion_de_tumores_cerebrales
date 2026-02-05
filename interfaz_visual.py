# ============================================================
# INTERFAZ VISUAL PROFESIONAL
# Sistema de Deteccion de Tumores Cerebrales
# ============================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import zipfile
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pydicom

from procesamiento import procesar_imagenes, etiquetar_tumores, guardar_base_datos
from clasificacion import entrenar_clasificador
from visualizacion import visualizar_deteccion
from carga_datos import extraer_datos, buscar_carpetas_t1


class SistemaTumoresGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Deteccion y Visualizacion de Tumores Cerebrales")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.zip_path = None
        self.df = None
        self.imagenes_guardadas = []
        self.imagen_actual = 0
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz grafica completa"""
        
        # Frame superior - Titulo
        frame_titulo = tk.Frame(self.root, bg='#34495e', height=80)
        frame_titulo.pack(fill=tk.X, padx=10, pady=10)
        
        titulo = tk.Label(
            frame_titulo,
            text="SISTEMA DE DETECCION DE TUMORES CEREBRALES",
            font=('Arial', 24, 'bold'),
            bg='#34495e',
            fg='white'
        )
        titulo.pack(pady=20)
        
        # Frame central - Contenido principal
        frame_central = tk.Frame(self.root, bg='#2c3e50')
        frame_central.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Controles
        self.crear_panel_controles(frame_central)
        
        # Panel derecho - Visualizacion
        self.crear_panel_visualizacion(frame_central)
        
        # Frame inferior - Estado
        self.crear_barra_estado()
    
    def crear_panel_controles(self, parent):
        """Crea el panel de controles izquierdo"""
        frame_controles = tk.Frame(parent, bg='#34495e', width=350)
        frame_controles.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        frame_controles.pack_propagate(False)
        
        # Titulo del panel
        titulo_panel = tk.Label(
            frame_controles,
            text="PANEL DE CONTROL",
            font=('Arial', 16, 'bold'),
            bg='#34495e',
            fg='white'
        )
        titulo_panel.pack(pady=20)
        
        # Separador
        ttk.Separator(frame_controles, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
        
        # Boton cargar datos
        self.btn_cargar = tk.Button(
            frame_controles,
            text="CARGAR BASE DE DATOS",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            cursor='hand2',
            command=self.cargar_datos,
            height=2
        )
        self.btn_cargar.pack(pady=10, padx=20, fill=tk.X)
        
        # Boton procesar
        self.btn_procesar = tk.Button(
            frame_controles,
            text="PROCESAR IMAGENES",
            font=('Arial', 12, 'bold'),
            bg='#2ecc71',
            fg='white',
            activebackground='#27ae60',
            activeforeground='white',
            cursor='hand2',
            command=self.procesar_datos,
            height=2,
            state=tk.DISABLED
        )
        self.btn_procesar.pack(pady=10, padx=20, fill=tk.X)
        
        # Boton clasificar
        self.btn_clasificar = tk.Button(
            frame_controles,
            text="CLASIFICAR TUMORES",
            font=('Arial', 12, 'bold'),
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            activeforeground='white',
            cursor='hand2',
            command=self.clasificar_datos,
            height=2,
            state=tk.DISABLED
        )
        self.btn_clasificar.pack(pady=10, padx=20, fill=tk.X)
        
        # Separador
        ttk.Separator(frame_controles, orient='horizontal').pack(fill=tk.X, padx=20, pady=20)
        
        # Frame de navegacion de imagenes
        frame_nav = tk.Frame(frame_controles, bg='#34495e')
        frame_nav.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            frame_nav,
            text="NAVEGACION DE IMAGENES",
            font=('Arial', 11, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=(0, 10))
        
        # Botones de navegacion
        frame_botones_nav = tk.Frame(frame_nav, bg='#34495e')
        frame_botones_nav.pack()
        
        self.btn_anterior = tk.Button(
            frame_botones_nav,
            text="◄ ANTERIOR",
            font=('Arial', 10, 'bold'),
            bg='#e74c3c',
            fg='white',
            cursor='hand2',
            command=self.imagen_anterior,
            state=tk.DISABLED,
            width=12
        )
        self.btn_anterior.pack(side=tk.LEFT, padx=5)
        
        self.btn_siguiente = tk.Button(
            frame_botones_nav,
            text="SIGUIENTE ►",
            font=('Arial', 10, 'bold'),
            bg='#e74c3c',
            fg='white',
            cursor='hand2',
            command=self.imagen_siguiente,
            state=tk.DISABLED,
            width=12
        )
        self.btn_siguiente.pack(side=tk.LEFT, padx=5)
        
        # Label contador de imagenes
        self.label_contador = tk.Label(
            frame_nav,
            text="Imagen 0 de 0",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        )
        self.label_contador.pack(pady=10)
        
        # Separador
        ttk.Separator(frame_controles, orient='horizontal').pack(fill=tk.X, padx=20, pady=20)
        
        # Panel de estadisticas
        self.crear_panel_estadisticas(frame_controles)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            frame_controles,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=20, padx=20)
    
    def crear_panel_estadisticas(self, parent):
        """Crea el panel de estadisticas"""
        frame_stats = tk.Frame(parent, bg='#34495e')
        frame_stats.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(
            frame_stats,
            text="ESTADISTICAS",
            font=('Arial', 11, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=(0, 10))
        
        # Frame para stats
        self.frame_stats_content = tk.Frame(frame_stats, bg='#2c3e50', relief=tk.SUNKEN, bd=2)
        self.frame_stats_content.pack(fill=tk.BOTH, expand=True)
        
        self.label_stats = tk.Label(
            self.frame_stats_content,
            text="No hay datos procesados",
            font=('Consolas', 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            justify=tk.LEFT,
            anchor='nw',
            padx=10,
            pady=10
        )
        self.label_stats.pack(fill=tk.BOTH, expand=True)
    
    def crear_panel_visualizacion(self, parent):
        """Crea el panel de visualizacion derecho"""
        frame_viz = tk.Frame(parent, bg='#34495e')
        frame_viz.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Titulo
        titulo_viz = tk.Label(
            frame_viz,
            text="VISUALIZACION DE TUMORES",
            font=('Arial', 16, 'bold'),
            bg='#34495e',
            fg='white'
        )
        titulo_viz.pack(pady=10)
        
        # Canvas para matplotlib
        self.frame_canvas = tk.Frame(frame_viz, bg='white')
        self.frame_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Mensaje inicial
        self.label_inicial = tk.Label(
            self.frame_canvas,
            text="Cargue y procese datos para visualizar tumores",
            font=('Arial', 14),
            bg='white',
            fg='#7f8c8d'
        )
        self.label_inicial.pack(expand=True)
    
    def crear_barra_estado(self):
        """Crea la barra de estado inferior"""
        self.frame_estado = tk.Frame(self.root, bg='#34495e', height=40)
        self.frame_estado.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))
        
        self.label_estado = tk.Label(
            self.frame_estado,
            text="Estado: Listo para comenzar",
            font=('Arial', 10),
            bg='#34495e',
            fg='white',
            anchor='w'
        )
        self.label_estado.pack(side=tk.LEFT, padx=10, pady=5)
    
    def actualizar_estado(self, mensaje):
        """Actualiza el mensaje de estado"""
        self.label_estado.config(text=f"Estado: {mensaje}")
        self.root.update()
    
    def cargar_datos(self):
        """Carga el archivo ZIP con los datos"""
        self.zip_path = filedialog.askopenfilename(
            title="Selecciona el archivo ZIP con 'Base de datos Brats'",
            filetypes=[("Archivos ZIP", "*.zip"), ("Todos los archivos", "*.*")]
        )
        
        if self.zip_path:
            self.actualizar_estado(f"Archivo cargado: {os.path.basename(self.zip_path)}")
            self.btn_procesar.config(state=tk.NORMAL)
            messagebox.showinfo("Exito", "Archivo ZIP cargado correctamente.\nHaga clic en 'PROCESAR IMAGENES' para continuar.")
        else:
            self.actualizar_estado("Carga cancelada")
    
    def procesar_datos(self):
        """Procesa los datos en un hilo separado"""
        thread = threading.Thread(target=self._procesar_datos_thread)
        thread.daemon = True
        thread.start()
    
    def _procesar_datos_thread(self):
        """Procesa los datos (ejecutado en hilo separado)"""
        try:
            self.progress.start()
            self.actualizar_estado("Extrayendo datos...")
            
            # Extraer datos
            extract_dir = extraer_datos(self.zip_path, self.script_dir)
            t1_dirs = buscar_carpetas_t1(extract_dir)
            
            if len(t1_dirs) == 0:
                messagebox.showerror("Error", "No se encontraron carpetas T1wCE")
                return
            
            self.actualizar_estado(f"Procesando {len(t1_dirs)} pacientes...")
            
            # Procesar imagenes
            self.df, self.imagenes_guardadas = procesar_imagenes(t1_dirs)
            self.df, self.mediana_area = etiquetar_tumores(self.df)
            guardar_base_datos(self.df, self.script_dir)
            
            self.progress.stop()
            self.actualizar_estado("Procesamiento completado")
            
            # Actualizar estadisticas
            self.actualizar_estadisticas()
            
            # Habilitar botones
            self.btn_clasificar.config(state=tk.NORMAL)
            self.btn_anterior.config(state=tk.NORMAL)
            self.btn_siguiente.config(state=tk.NORMAL)
            
            # Mostrar primera imagen
            self.imagen_actual = 0
            self.mostrar_imagen_actual()
            
            messagebox.showinfo("Exito", f"Se procesaron {len(self.df)} imagenes correctamente!")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Error al procesar datos: {str(e)}")
            self.actualizar_estado("Error en el procesamiento")
    
    def clasificar_datos(self):
        """Clasifica los tumores"""
        try:
            self.progress.start()
            self.actualizar_estado("Clasificando tumores...")
            
            modelo, accuracy, importancias = entrenar_clasificador(self.df)
            
            self.progress.stop()
            self.actualizar_estado(f"Clasificacion completada - Accuracy: {accuracy:.2%}")
            
            # Actualizar estadisticas con accuracy
            self.actualizar_estadisticas(accuracy)
            
            messagebox.showinfo("Exito", f"Clasificacion completada!\n\nAccuracy: {accuracy:.2%}\nFeature mas importante: {importancias.iloc[0]['Característica']}")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Error en clasificacion: {str(e)}")
    
    def actualizar_estadisticas(self, accuracy=None):
        """Actualiza el panel de estadisticas"""
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
        """Muestra la imagen actual en el canvas"""
        if not self.imagenes_guardadas:
            return
        
        # Limpiar canvas anterior
        for widget in self.frame_canvas.winfo_children():
            widget.destroy()
        
        # Obtener imagen actual
        img, paciente, archivo, stats = self.imagenes_guardadas[self.imagen_actual]
        
        # Crear figura de matplotlib
        from deteccion import detectar_tumor
        from matplotlib.colors import LinearSegmentedColormap
        import cv2
        
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        tumor_mask, contours, stats = detectar_tumor(img, return_mask=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{paciente} - {archivo}", fontsize=14, fontweight='bold')
        
        # 1. Imagen original
        axes[0, 0].imshow(img_norm, cmap='gray')
        axes[0, 0].set_title('Imagen Original T1wCE', fontsize=10)
        axes[0, 0].axis('off')
        
        # 2. Imagen con contornos
        img_rgb = np.stack([img_norm]*3, axis=-1)
        for contour in contours:
            cv2.drawContours(img_rgb, [contour], -1, (1, 0, 0), 2)
        axes[0, 1].imshow(img_rgb)
        axes[0, 1].set_title(f'Deteccion ({stats["num_regions"]} regiones)', fontsize=10)
        axes[0, 1].axis('off')
        
        # 3. Mapa de calor
        cmap_heat = LinearSegmentedColormap.from_list('tumor', ['black', 'blue', 'cyan', 'yellow', 'red'])
        axes[1, 0].imshow(img_norm, cmap=cmap_heat)
        axes[1, 0].set_title('Mapa de Calor', fontsize=10)
        axes[1, 0].axis('off')
        
        # 4. Segmentacion
        overlay = np.zeros_like(img_rgb)
        overlay[:, :, 0] = tumor_mask
        blended = img_rgb * 0.6 + overlay * 0.4
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title('Segmentacion', fontsize=10)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Integrar con tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Actualizar contador
        self.label_contador.config(text=f"Imagen {self.imagen_actual + 1} de {len(self.imagenes_guardadas)}")
    
    def imagen_anterior(self):
        """Muestra la imagen anterior"""
        if self.imagen_actual > 0:
            self.imagen_actual -= 1
            self.mostrar_imagen_actual()
    
    def imagen_siguiente(self):
        """Muestra la imagen siguiente"""
        if self.imagen_actual < len(self.imagenes_guardadas) - 1:
            self.imagen_actual += 1
            self.mostrar_imagen_actual()


def main():
    """Funcion principal"""
    root = tk.Tk()
    app = SistemaTumoresGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
