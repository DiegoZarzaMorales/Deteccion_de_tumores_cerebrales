# 🧠 Guía de Entrenamiento YOLO (Segmentación)

Esta guía explica cómo entrenar YOLO en este proyecto (enfocado en YOLO26) para detectar/segmentar tumor cerebral y dejar el modelo integrado con la web y la GUI.

## 1) Archivos añadidos para entrenamiento

- `ml/entrenar_yolo.py`: lógica de entrenamiento YOLO-Seg.
- `python-scripts/entrenar_yolo.py`: wrapper para ejecutar desde `python-scripts/`.
- `datasets/brats_yolo/data.yaml`: plantilla de configuración del dataset.

## 2) Requisitos

Instala dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Verifica que `ultralytics` esté disponible (viene en `requirements.txt`).

## 3) Estructura de dataset requerida (YOLO-Seg)

YOLO no entrena directo con DICOM `.dcm`. Debes convertir imágenes a `.png` o `.jpg` y tener etiquetas en formato YOLO de segmentación.

Estructura esperada:

```text
Deteccion_de_tumores_cerebrales/
└── datasets/
    └── brats_yolo/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   │   ├── img_0001.png
        │   │   └── ...
        │   └── val/
        │       ├── img_1001.png
        │       └── ...
        └── labels/
            ├── train/
            │   ├── img_0001.txt
            │   └── ...
            └── val/
                ├── img_1001.txt
                └── ...
```

### Formato de etiqueta YOLO-Seg (`.txt`)

Cada línea representa una instancia:

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- Coordenadas normalizadas entre `0` y `1`.
- Para este caso, una sola clase:
  - `0 = tumor`

Ejemplo:

```text
0 0.421 0.332 0.438 0.351 0.456 0.378 0.470 0.401 ...
```

> Si no tienes máscaras reales anotadas, primero debes generar/obtener anotaciones. Sin etiquetas confiables no hay entrenamiento útil.

## 4) Configurar `data.yaml`

Ya tienes una plantilla en `datasets/brats_yolo/data.yaml`:

```yaml
path: datasets/brats_yolo
train: images/train
val: images/val

names:
  0: tumor
```

Si cambias de ruta, actualiza `path`.

## 5) Entrenar YOLO

### Comando rápido (desde raíz del proyecto)

```bash
python python-scripts/entrenar_yolo.py
```

Usa por defecto:
- `data = datasets/brats_yolo/data.yaml`
- `model = yolo26n-seg.pt`
- `epochs = 100`
- `imgsz = 256`
- `batch = 8`

### Comando personalizado

```bash
python python-scripts/entrenar_yolo.py --data datasets/brats_yolo/data.yaml --model yolo26n-seg.pt --epochs 120 --batch 8 --imgsz 256 --device 0 --name tumor_brats_v1
```

Parámetros útiles:
- `--device 0`: GPU 0 (NVIDIA).
- `--device cpu`: forzar CPU.
- `--no-copy`: evita copiar `best.pt` a `models/`.

## 6) Dónde queda el modelo entrenado

Ultralytics guarda salidas en:

```text
runs/segment/<nombre_experimento>/
```

El mejor peso suele quedar en:

```text
runs/segment/<nombre_experimento>/weights/best.pt
```

Además, por defecto el script lo copia automáticamente a (según modelo base):

```text
models/yolo26n-seg.pt
o models/yolo11n-seg.pt
```

Esto lo deja listo para:
- Web: `python web_app.py`
- GUI: `python python-scripts/main.py`

## 7) Validación rápida post-entrenamiento

1. Levanta web:

```bash
python web_app.py
```

2. Abre `http://127.0.0.1:5000`.
3. Elige modelo **YOLO26**.
4. Sube un `.dcm` de prueba y verifica que dibuja tumor/heatmap.

## 8) Errores comunes

- **`No se encontró data.yaml`**
  - Revisa ruta en `--data`.
- **No detecta GPU**
  - Verifica instalación de CUDA/PyTorch compatible.
- **Resultados pobres**
  - Revisa calidad de etiquetas, balance de datos, y aumenta épocas.
- **OOM (memoria GPU)**
  - Baja `--batch` o `--imgsz`.

## 9) Recomendaciones prácticas

- Si tienes checkpoint YOLO26, usa `yolo26n-seg.pt`.
- Si buscas más capacidad, evalúa variantes YOLO26 más grandes.
- Mantén separado train/val por paciente para evitar fuga de datos.
- Guarda versión del experimento con `--name` claro (ej. `tumor_brats_2026_02`).

---

Si quieres, en el siguiente paso te puedo agregar un script de **conversión DICOM + máscara binaria → YOLO-Seg labels** para automatizar la creación del dataset `images/labels`.
