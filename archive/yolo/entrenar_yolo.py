"""Entrenamiento YOLO para detección/segmentación de tumores.

Este módulo asume que ya existe un dataset en formato YOLO:
- images/train, images/val
- labels/train, labels/val
- data.yaml
"""

from pathlib import Path
import shutil


def entrenar_yolo(
    data_yaml,
    model="yolo26n-seg.pt",
    epochs=100,
    imgsz=256,
    batch=8,
    device=None,
    project="runs/segment",
    name="tumor_brats",
    patience=20,
    workers=0,
    copy_best_to_models=True,
    models_dir="models",
):
    """Entrena YOLO-Seg y opcionalmente copia best.pt al directorio models/."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "No se encontró 'ultralytics'. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"No se encontró data.yaml en: {data_yaml}\n"
            "Prepara primero el dataset YOLO y revisa la ruta."
        )

    if device is None:
        try:
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print("\n" + "=" * 64)
    print("ENTRENAMIENTO YOLO (SEGMENTACIÓN)")
    print("=" * 64)
    print(f"Dataset YAML: {data_yaml}")
    print(f"Modelo base:  {model}")
    print(f"Épocas:       {epochs}")
    print(f"Img size:     {imgsz}")
    print(f"Batch size:   {batch}")
    print(f"Device:       {device}")
    print("=" * 64 + "\n")

    yolo = YOLO(model)
    resultados = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
    )

    if copy_best_to_models:
        best_path = Path(resultados.save_dir) / "weights" / "best.pt"
        if best_path.exists():
            models_path = Path(models_dir)
            models_path.mkdir(exist_ok=True)
            model_name = Path(str(model)).name.lower()
            if "yolo26" in model_name:
                output_name = "yolo26n-seg.pt"
            elif "yolo11" in model_name:
                output_name = "yolo11n-seg.pt"
            else:
                output_name = "yolo26n-seg.pt"
            destino = models_path / output_name
            shutil.copy2(best_path, destino)
            print(f"✓ Modelo copiado a: {destino}")
        else:
            print("⚠ No se encontró best.pt para copiar al directorio models/")

    print(f"\n✓ Entrenamiento finalizado. Resultados en: {resultados.save_dir}")
    return resultados


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar YOLO-Seg para tumores cerebrales")
    parser.add_argument(
        "--data",
        default="datasets/brats_yolo/data.yaml",
        help="Ruta al data.yaml de YOLO",
    )
    parser.add_argument("--model", default="yolo26n-seg.pt", help="Modelo base YOLO-Seg")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=256, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo (ej: 0 para GPU0, 'cpu' para CPU)",
    )
    parser.add_argument("--name", default="tumor_brats", help="Nombre del experimento")
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="No copiar best.pt al directorio models/",
    )

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
