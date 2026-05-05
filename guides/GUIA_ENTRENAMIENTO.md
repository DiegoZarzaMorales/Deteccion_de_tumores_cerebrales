# 🧠 Guía de Entrenamiento - Deep Learning para Detección de Tumores

> Nota: El soporte para YOLO fue retirado del flujo principal; los recursos (código, modelos y datasets)
> se han archivado en la carpeta `YOLO-ARCHIVE` dentro del repositorio para referencia futura.

## 📋 ¿Qué hace cada archivo?

### 1. **modelo_unet.py**
- Contiene la arquitectura U-Net (red neuronal para segmentación)
- U-Net es el estándar en imágenes médicas
- Aprende a detectar tumores pixel por pixel

### 2. **dataset_loader.py**
- Carga las imágenes DICOM de tu carpeta BraTS
- Prepara los datos para el entrenamiento
- Crea máscaras automáticas (en proyecto real, serían anotadas por médicos)

### 3. **entrenar.py**
- **ESTE ES EL ARCHIVO PRINCIPAL PARA ENTRENAR**
- Entrena la IA con tus imágenes
- Guarda el modelo en la carpeta `models/`

### 4. **predecir.py**
- Usa el modelo entrenado para detectar tumores en nuevas imágenes
- Genera visualizaciones de las predicciones

---

## 🚀 Pasos para Entrenar la IA

### **Paso 1: Instalar Dependencias**

```bash
pip install torch torchvision tqdm tensorboard
```

O instalar todo de una vez:
```bash
pip install -r requirements.txt
```

### **Paso 2: Entrenar el Modelo**

Ejecuta el script de entrenamiento:

```bash
python entrenar.py
```

Esto iniciará el entrenamiento y verás:
- Barra de progreso con el avance
- Métricas: Loss, Dice Score, IoU, Accuracy
- Se guardará el mejor modelo en `models/best_model.pth`

**⚙️ Configuración:**
- **Épocas**: 30 (puedes cambiar a más para mejor precisión)
- **Batch size**: 4 (reduce si tienes poca RAM)
- **Learning rate**: 0.001

### **Paso 3: Usar el Modelo Entrenado**

Una vez entrenado, puedes predecir tumores en nuevas imágenes:

```bash
python python-scripts/predecir.py
```

---

## 📊 ¿Cómo Funciona el Entrenamiento?

1. **Carga de Datos**: Lee todas las imágenes DICOM de BraTS
2. **Preprocesamiento**: Normaliza y redimensiona a 256x256
3. **Entrenamiento**: La red U-Net aprende a identificar patrones de tumores
4. **Validación**: Evalúa el modelo en datos no vistos
5. **Guardado**: Guarda el mejor modelo basado en Dice Score

### Métricas:
- **Dice Score**: Mide qué tan bien la IA detecta el tumor (0-1, más alto = mejor)
- **IoU**: Intersección sobre unión (similar al Dice)
- **Accuracy**: Precisión general de la predicción

---

## 🎯 Personalización

### Cambiar Número de Épocas:
```python
# En entrenar.py, línea final
modelo, history = entrenar(
    root_dir=ROOT_DIR,
    num_epochs=50,  # ← Cambia aquí (30 por defecto)
    batch_size=4,
    learning_rate=0.001
)
```

### Cambiar Modalidad de Imagen:
```python
# En dataset_loader.py, línea 81
BraTSDataset(root_dir, modality='T1wCE')  # ← Cambia: T1w, T2w, FLAIR
```

### Usar GPU (NVIDIA):
- Si tienes GPU NVIDIA, el código la detectará automáticamente
- Entrenamiento será **mucho más rápido**
- Verás: `Dispositivo: cuda`

---

## 📁 Estructura de Salida

Después del entrenamiento:
```
DisenoDeInterfaz/
├── models/
│   ├── best_model.pth           # Mejor modelo entrenado
│   ├── checkpoint_epoch_10.pth  # Checkpoints cada 10 épocas
│   └── training_history.png     # Gráficas de entrenamiento
├── runs/
│   └── unet_tumor/              # Logs de TensorBoard
└── predicciones/                # Predicciones guardadas
```

---

## 🔍 Ver Progreso con TensorBoard

Mientras entrena, puedes visualizar en tiempo real:

```bash
tensorboard --logdir=runs
```

Abre tu navegador en: http://localhost:6006

---

## ⚠️ Nota Importante

**Este código usa máscaras automáticas (pseudo-labels)** porque no tienes anotaciones manuales. Para un proyecto médico real:

1. Necesitas máscaras anotadas por expertos (ground truth)
2. BraTS dataset completo ya incluye estas anotaciones
3. Modifica `_crear_mascara_automatica()` para cargar las anotaciones reales

---

## 💡 Consejos

- **Empieza con pocas épocas (10-20)** para probar
- **Monitorea el Dice Score**: si no mejora, ajusta learning rate
- **Guarda checkpoints**: no perderás progreso si se interrumpe
- **Valida siempre**: asegúrate que no haya overfitting

---

## 🎓 Siguientes Pasos

1. ✅ Entrenar el modelo básico
2. ✅ Evaluar resultados con `python-scripts/predecir.py`
3. 🔄 Ajustar hiperparámetros si es necesario
4. 🎯 Integrar con tu interfaz actual
5. 📈 Usar datos reales con anotaciones médicas

---

**¡Listo! Ahora tienes todo para entrenar tu IA de detección de tumores** 🚀
