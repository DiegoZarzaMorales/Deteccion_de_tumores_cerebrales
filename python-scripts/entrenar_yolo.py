"""Wrapper para entrenar YOLO con la configuración del proyecto.

Uso rápido:
    python entrenar_yolo.py

Uso con parámetros:
    python entrenar_yolo.py --data datasets/brats_yolo/data.yaml --epochs 100 --batch 8
"""

from ml.entrenar_yolo import entrenar_yolo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar YOLO-Seg para detección de tumores")
    parser.add_argument("--data", default="datasets/brats_yolo/data.yaml", help="Ruta a data.yaml")
    parser.add_argument("--model", default="yolo26n-seg.pt", help="Modelo base")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=256, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="0, 1, cpu, etc.")
    parser.add_argument("--name", default="tumor_brats", help="Nombre del experimento")
    parser.add_argument("--no-copy", action="store_true", help="No copiar best.pt a models/")

    args = parser.parse_args()

    entrenar_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        copy_best_to_models=not args.no_copy,
    )
