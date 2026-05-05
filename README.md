# Sistema de Detección y Visualización de Tumores Cerebrales
# Python 3.14.4 OBLIGATORIO

## Descripción
Sistema para detección y visualización de tumores cerebrales en imágenes T1wCE.
Incluye:
- Pipeline clásico de procesamiento + Machine Learning.
- Modelo de IA basado en U‑Net (PyTorch) para detectar/segmentar la zona tumoral.
- Interfaz de escritorio (Tkinter) y frontend web (Flask) para uso sencillo por el usuario.

## Estructura del proyecto

```
DisenoDeInterfaz/
│
├── python-scripts/       # Scripts que antes estaban en la raíz
│   ├── main.py           # Entrada GUI de escritorio (Tkinter)
│   ├── predecir.py       # Wrapper para usar predicción U‑Net por consola
│   ├── preparar_yolo_dcm.py
│   ├── entrenar_yolo.py
│   ├── dataset_loader.py
│   └── modelo_unet.py
├── entrenar.py           # Wrapper para entrenar la U‑Net
├── convertir_mat_a_imagen.py
├── web_app.py            # Wrapper para lanzar la app web (Flask)
├── core/                 # Lógica clásica
│   ├── config.py         # Configuración y parámetros de detección
│   ├── carga_datos.py    # Carga y extracción de ZIP de BraTS
│   ├── procesamiento.py  # Procesamiento de imágenes y features
│   ├── deteccion.py      # Algoritmos de detección clásica
│   ├── clasificacion.py  # Random Forest y métricas
│   └── visualizacion.py  # Gráficas y análisis espacial
├── ml/                   # Módulos de IA (PyTorch)
│   ├── modelo_unet.py    # Arquitectura U‑Net
│   ├── dataset_loader.py # Dataset/Dataloaders BraTS (pseudo‑máscaras)
│   ├── entrenar.py       # Lazo de entrenamiento U‑Net
│   ├── predecir.py       # Funciones de inferencia y visualización IA (U‑Net)
│   └── predecir.py       # Funciones de inferencia y visualización IA (U‑Net)
├── gui/
│   └── interfaz_visual.py # Interfaz completa de escritorio
├── web/
│   └── app.py            # Aplicación Flask (frontend HTML)
├── models/               # Modelos entrenados (.pth)
├── static/               # Recursos estáticos web (incluye resultados PNG)
├── uploads/              # Archivos subidos en modo web
├── requirements.txt      # Dependencias
└── README.md
```

## Instalación

1. Crear y activar un entorno virtual (opcional pero recomendado).
2. Instalar dependencias:
	```bash
	pip install -r requirements.txt
	```

### Descargar modelo entrenado (carpeta `models`)

El modelo de IA entrenado no se sube a GitHub por tamaño (>100 MB por archivo).

1. Descarga la carpeta `models` desde Google Drive:
	- Enlace: https://drive.google.com/drive/folders/1FGNE-fo40H9GsfXLJ5kDjGT3_25ifG94?usp=sharing
2. Copia la carpeta `models` descargada dentro de la raíz del proyecto, de forma que quede:
	```
	DisenoDeInterfaz/
	├── models/
	│   ├── best_model.pth
	│   ├── checkpoint_epoch_10.pth (opcional)
	│   ├── checkpoint_epoch_20.pth (opcional)
	│   ├── checkpoint_epoch_30.pth (opcional)
	│   └── training_history.png (opcional)
	└── ... resto de archivos del proyecto
	```

Con esto, los scripts `entrenar.py`, `python-scripts/predecir.py`, la GUI (`python-scripts/main.py`) y la app web (`web_app.py`) encontrarán automáticamente `models/best_model.pth`.

## Uso rápido

### 1) Entrenar la IA (U‑Net)

1. Coloca los datos BraTS en la ruta esperada (por defecto):
	`brats_data/Base de datos Brats/.../T1wCE/*.dcm`
2. Ejecuta el entrenamiento:
	```bash
	python entrenar.py
	```
3. El mejor modelo se guardará en `models/best_model.pth`.

> Nota: El soporte para YOLO fue retirado del flujo principal y los recursos (código, modelos y datasets)
> se han archivado en la carpeta `YOLO-ARCHIVE` dentro del repositorio para referencia futura.

### 2) Interfaz de escritorio (modo completo)

```bash
python python-scripts/main.py
```

- Carga un ZIP con la base de datos (BraTS).
- Procesa imágenes, detecta tumores, entrena un clasificador clásico y muestra análisis.
- Botones extra permiten usar IA (U‑Net o YOLO) para analizar imágenes T1wCE individuales o carpetas.

### 3) Frontend web (modo simple para usuario)

```bash
python web_app.py
```

Luego abre en el navegador: `http://localhost:5000`

- Sube un archivo `.dcm` o un `.zip` con varias T1wCE.
-- Elige el modelo (`U-Net`) y la IA resalta la región tumoral, mostrando área aproximada, lado de la imagen y aviso de calidad.

<!-- YOLO support removed; resources archived in `YOLO-ARCHIVE` -->

## Autores

Cuevas Estrada Joel
Zarza Morales Diego

## Venv
Creacion dentro del folder del proyecto (En windows es diferentre)
Windows

``bash
.\.venv\Scripts\Activate.ps1
``

LINUX
``bash
python3 -m venv .venv
source .venv/bin/activate
``
entrar al venv: source .venv/bin/activate
salir del venv: deactivate