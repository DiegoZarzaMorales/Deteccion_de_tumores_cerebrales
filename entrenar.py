"""
Wrapper para lanzar el entrenamiento definido en ml.entrenar.

Asume que la base de datos está en la carpeta "brats_data/Base de datos Brats"
en la raíz del proyecto.
"""

import os
import argparse

from ml.entrenar import entrenar


def parse_args():
    parser = argparse.ArgumentParser(description="Lanzar entrenamiento U-Net")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(base_dir, "brats_data", "Base de datos Brats")

    entrenar(
        root_dir=ROOT_DIR,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir="models",
        device=None if args.device == "auto" else args.device,
    )
