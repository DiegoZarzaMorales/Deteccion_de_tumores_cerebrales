"""
Wrapper para lanzar el entrenamiento definido en ml.entrenar.

Por defecto usa la base de datos local en "brats_data/Base de datos Brats",
pero también acepta una ruta externa por parámetro.
"""

import os
import argparse
import warnings
from pathlib import Path

# Suprimir RuntimeWarnings ruidosos de numpy.getlimits en este entorno de Windows
warnings.filterwarnings(
    "ignore",
    message=r"Numpy built with MINGW-W64 on Windows 64 bits is experimental.*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in (exp2|nextafter|log10)",
    category=RuntimeWarning,
    module=r"numpy\.core\.getlimits",
)

from ml.entrenar import entrenar


def parse_args():
    parser = argparse.ArgumentParser(description="Lanzar entrenamiento U-Net")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Guardar un checkpoint cada N épocas",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Número de workers para DataLoader (0=main thread). Prueba 2-6 en Windows.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="prefetch_factor para DataLoader (cantidad por worker)",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Usar persistent_workers=True en DataLoader (mantener workers entre epochs)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Ruta a la carpeta raíz del dataset BraTS (por ejemplo, C:\\Users\\RESURGE\\Desktop\\learn-resources\\BraTS-PEDs-v1\\Training)",
    )
    return parser.parse_args()


def resolver_directorio_datos(data_dir, base_dir):
    """Resuelve la carpeta de entrenamiento real a partir de una ruta externa o local.

    Si se pasa la raíz del dataset BraTS-PEDs-v1, usa la subcarpeta Training.
    Si ya se pasa Training o Validation, respeta esa ruta.
    """
    if data_dir:
        data_path = Path(data_dir)
    else:
        data_path = Path(base_dir) / "brats_data" / "Base de datos Brats"

    if data_path.name.lower() in {"training", "validation"}:
        return str(data_path)

    training_path = data_path / "Training"
    if training_path.exists():
        return str(training_path)

    validation_path = data_path / "Validation"
    if validation_path.exists():
        return str(validation_path)

    return str(data_path)

## CODIGO PARA EJECUTAR CON PARAMETROS PERSONALIZADOS DESDE LA TERMINAL:
## python entrenar.py --device cuda --epochs 1 --batch-size 8 --num-workers 4 --prefetch-factor 2 --persistent-workers --checkpoint-every 0 --data-dir "C:\Users\RESURGE\Desktop\learn-resources\BraTS-PEDs-v1\Training"

if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = resolver_directorio_datos(args.data_dir, base_dir)

    entrenar(
        root_dir=ROOT_DIR,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_every_n_epochs=args.checkpoint_every,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        save_dir="models",
        device=None if args.device == "auto" else args.device,
    )
