"""Wrapper para preparar dataset YOLO-Seg desde MRI DICOM 2D.

Uso:
    python preparar_yolo_dcm.py --overwrite
"""

from ml.preparar_yolo_dcm import main


if __name__ == "__main__":
    main()
