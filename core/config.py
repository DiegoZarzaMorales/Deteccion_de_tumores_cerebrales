# ============================================================
# CONFIGURACIÓN DEL SISTEMA
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suprimir advertencias
warnings.filterwarnings('ignore')

# Configuración visual de gráficos
plt.rcParams['figure.dpi'] = 100
sns.set_style("dark")

# Parámetros de detección de tumores
PARAMETROS_DETECCION = {
    'sigma_suavizado': 1.5,
    'min_size_objeto': 50,
    'radio_dilatacion': 1,
    'num_dilataciones': 2
}

# Límite de imágenes a procesar por paciente (para demo)
LIMITE_IMAGENES_POR_PACIENTE = 10

# Número de imágenes a guardar para visualización
NUM_IMAGENES_VISUALIZACION = 6

# Número de casos representativos a mostrar
NUM_CASOS_REPRESENTATIVOS = 3
