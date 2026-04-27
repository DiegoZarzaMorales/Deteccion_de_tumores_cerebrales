"""Wrapper para convertir .mat a PNG/JPG.

Ejemplo:
    python convertir_mat_a_imagen.py --input /ruta/a/mats --output datasets/mat_export --recursive --all-slices
"""

from ml.convertir_mat_a_imagen import main


if __name__ == "__main__":
    main()
