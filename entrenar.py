"""
Wrapper para lanzar el entrenamiento definido en ml.entrenar.

Por defecto usa la base de datos local en "brats_data/Base de datos Brats",
pero también acepta una ruta externa por parámetro.
"""

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Lanzar entrenamiento U-Net")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Guardar un checkpoint cada N épocas (0 desactiva checkpoints periódicos)",
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
        help=(
            "Ruta a la carpeta raíz del dataset BraTS "
            '(por ejemplo, "C:\\Users\\ElZuk\\MRI-IA-Training\\data\\raw")'
        ),
    )
    parser.add_argument(
        "--hdf5-empty-keep-ratio",
        type=float,
        default=0.25,
        help=(
            "Fracción de cortes HDF5 sin tumor (no negros) a conservar como "
            "negativos. 0=solo tumores, 1=todas las vacías. Default: 0.25"
        ),
    )
    args = parser.parse_args(argv)
    _validar_args(parser, args)
    return args


def _validar_args(parser, args):
    """Valida rangos básicos sin alterar la lógica de entrenamiento."""
    if args.epochs < 1:
        parser.error("--epochs debe ser >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size debe ser >= 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate debe ser > 0")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every debe ser >= 0")
    if args.num_workers < 0:
        parser.error("--num-workers debe ser >= 0")
    if args.num_workers == 0 and args.prefetch_factor < 1:
        # prefetch solo aplica con workers; no bloqueamos, pero evitamos valores inválidos
        # cuando sí hay workers.
        pass
    if args.num_workers > 0 and args.prefetch_factor < 1:
        parser.error("--prefetch-factor debe ser >= 1 cuando --num-workers > 0")
    if not 0.0 <= args.hdf5_empty_keep_ratio <= 1.0:
        parser.error("--hdf5-empty-keep-ratio debe estar entre 0 y 1")


def resolver_directorio_datos(data_dir, base_dir):
    """Resuelve la carpeta de entrenamiento real a partir de una ruta externa o local.

    Si se pasa la raíz del dataset BraTS-PEDs-v1, usa la subcarpeta Training.
    Si ya se pasa Training o Validation, respeta esa ruta.
    """
    if data_dir:
        data_path = Path(data_dir).expanduser()
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


def main(argv=None):
    args = parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    root_dir = resolver_directorio_datos(args.data_dir, base_dir)

    print(f"Directorio de datos: {root_dir}")

    entrenar(
        root_dir=root_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_every_n_epochs=args.checkpoint_every,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        save_dir="models",
        device=None if args.device == "auto" else args.device,
        hdf5_empty_keep_ratio=args.hdf5_empty_keep_ratio,
    )


# Ejemplo:
# python entrenar.py --data-dir "C:\Users\ElZuk\MRI-IA-Training\data\raw" --device cuda --epochs 100 --batch-size 16 --learning-rate 0.0001 --num-workers 2 --persistent-workers --prefetch-factor 2 --checkpoint-every 10

if __name__ == "__main__":
    main()
