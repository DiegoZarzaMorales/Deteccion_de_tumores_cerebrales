"""Wrapper para lanzar el entrenamiento definido en ml.entrenar."""

from ml.entrenar import entrenar


if __name__ == "__main__":
    ROOT_DIR = r"c:\Users\josez\Documents\DisenoDeInterfaz\brats_data\Base de datos Brats"

    entrenar(
        root_dir=ROOT_DIR,
        num_epochs=30,
        batch_size=4,
        learning_rate=0.001,
        save_dir="models",
    )
