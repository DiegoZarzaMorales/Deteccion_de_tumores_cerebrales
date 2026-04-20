# Sistema de Detección y Visualización de Tumores Cerebrales

## Descripción
Sistema para detección y visualización de tumores cerebrales en imágenes T1wCE.
Incluye:
- Pipeline clásico de procesamiento + Machine Learning.
- Modelo de IA con opción de U‑Net (PyTorch) o YOLO26 para detectar/segmentar la zona tumoral.
- Interfaz de escritorio (Tkinter) y frontend web (Flask) para uso sencillo por el usuario.

## Estructura del proyecto

```
DisenoDeInterfaz/
│
├── main.py               # Entrada GUI de escritorio (Tkinter)
├── entrenar.py           # Wrapper para entrenar la U‑Net
├── entrenar_yolo.py      # Wrapper para entrenar YOLO-Seg
├── predecir.py           # Wrapper para usar predicción U‑Net por consola
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
│   ├── yolo_infer.py     # Funciones de inferencia YOLO
│   └── entrenar_yolo.py  # Lazo de entrenamiento YOLO-Seg
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

Con esto, los scripts `entrenar.py`, `predecir.py`, la GUI (`main.py`) y la app web (`web_app.py`) encontrarán automáticamente `models/best_model.pth`.

## Uso rápido

### 1) Entrenar la IA (U‑Net)

1. Coloca los datos BraTS en la ruta esperada (por defecto):
	`brats_data/Base de datos Brats/.../T1wCE/*.dcm`
2. Ejecuta el entrenamiento:
	```bash
	python entrenar.py
	```
3. El mejor modelo se guardará en `models/best_model.pth`.

### 1.1) Entrenar la IA (YOLO)

1. Prepara dataset YOLO en:
	`datasets/brats_yolo/data.yaml`
2. Ejecuta el entrenamiento:
	```bash
	python entrenar_yolo.py
	```
3. Por defecto usa `yolo26n-seg.pt` como base y copia el mejor modelo a `models/yolo26n-seg.pt`.

### 2) Interfaz de escritorio (modo completo)

```bash
python main.py
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
- Elige el modelo (`U-Net` o `YOLO26`) y la IA resalta la región tumoral, mostrando área aproximada, lado de la imagen y aviso de calidad.

### Nota sobre YOLO

- Web (selector):
	- `YOLO26` intenta `models/yolo26n-seg.pt` y si no existe usa `yolo26n-seg.pt`.
- GUI (botón YOLO): usa `models/yolo26n-seg.pt` y, si no existe, usa `yolo26n-seg.pt`.
- Para una guía detallada de entrenamiento YOLO, revisa `GUIA_ENTRENAMIENTO_YOLO.md`.

## Autores

Cuevas Estrada Joel
Zarza Morales Diego

## Venv
Creacion dentro del folder del proyecto

``bash
python3 -m venv .venv
source .venv/bin/activate
``
entrar al venv: source .venv/bin/activate
salir del venv: deactivate