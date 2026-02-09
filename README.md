# Sistema de Detección y Visualización de Tumores Cerebrales

## Descripción
Sistema para detección y visualización de tumores cerebrales en imágenes T1wCE.
Incluye:
- Pipeline clásico de procesamiento + Machine Learning.
- Modelo de IA (U‑Net en PyTorch) para segmentar automáticamente la zona tumoral.
- Interfaz de escritorio (Tkinter) y frontend web (Flask) para uso sencillo por el usuario.

## Estructura del proyecto

```
DisenoDeInterfaz/
│
├── main.py               # Entrada GUI de escritorio (Tkinter)
├── entrenar.py           # Wrapper para entrenar la U‑Net
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
│   └── predecir.py       # Funciones de inferencia y visualización IA
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

## Uso rápido

### 1) Entrenar la IA (U‑Net)

1. Coloca los datos BraTS en la ruta esperada (por defecto):
	`brats_data/Base de datos Brats/.../T1wCE/*.dcm`
2. Ejecuta el entrenamiento:
	```bash
	python entrenar.py
	```
3. El mejor modelo se guardará en `models/best_model.pth`.

### 2) Interfaz de escritorio (modo completo)

```bash
python main.py
```

- Carga un ZIP con la base de datos (BraTS).
- Procesa imágenes, detecta tumores, entrena un clasificador clásico y muestra análisis.
- Botones extra permiten usar la IA (U‑Net) para analizar imágenes T1wCE individuales o carpetas.

### 3) Frontend web (modo simple para usuario)

```bash
python web_app.py
```

Luego abre en el navegador: `http://localhost:5000`

- Sube un archivo `.dcm` o un `.zip` con varias T1wCE.
- La IA resalta la región tumoral más brillante y muestra área aproximada, lado de la imagen y aviso de calidad.

## Autores

Narvaez, Ochoa, Zarza
