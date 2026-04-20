"""
Wrapper para lanzar el entrenamiento definido en ml.entrenar.

Asume que la base de datos está en la carpeta "brats_data/Base de datos Brats"
en la raíz del proyecto.
"""

import os

from ml.entrenar import entrenar


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(base_dir, "brats_data", "Base de datos Brats")

    entrenar(
        root_dir=ROOT_DIR,
        num_epochs=3,
        batch_size=4,
        learning_rate=0.001,
        save_dir="models",
    )
