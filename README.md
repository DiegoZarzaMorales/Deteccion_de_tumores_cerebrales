# Sistema de Detección y Visualización de Tumores Cerebrales

## Descripción
Sistema automatizado para la detección y visualización de regiones tumorales en imágenes cerebrales T1wCE.

## Estructura del Proyecto

```
DisenoDeInterfaz/
│
├── main.py                    # Archivo principal (ejecutar este)
├── config.py                  # Configuración del sistema
├── carga_datos.py            # Funciones de carga de archivos
├── procesamiento.py          # Procesamiento de imágenes
├── deteccion.py              # Algoritmos de detección
├── visualizacion.py          # Funciones de visualización
├── clasificacion.py          # Machine Learning
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo

```

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Ejecuta el programa principal:
```bash
python main.py
```

2. Se abrirá una ventana para seleccionar el archivo ZIP con los datos

3. El sistema procesará automáticamente las imágenes y mostrará las visualizaciones

## Características

- Detección automática de tumores
- Visualización de regiones tumorales
- Análisis espacial
- Clasificación con Machine Learning
- Generación de reportes

## Autores

Narvaez, Ochoa, Zarza
