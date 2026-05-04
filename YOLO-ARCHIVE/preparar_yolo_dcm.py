"""Prepara un dataset YOLO-Seg desde MRI DICOM 2D.

Genera:
- images/train, images/val
- labels/train, labels/val
- data.yaml

Notas:
- Usa una mascara heuristica basada en pixeles brillantes (pseudo-label).
- El split train/val se hace por paciente para reducir fuga de datos.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
import yaml


@dataclass
class DatasetStats:
    total_images: int = 0
    train_images: int = 0
    val_images: int = 0
    images_with_label: int = 0
    empty_labels: int = 0


def _normalizar_dicom_a_float(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """Convierte el pixel_array a float32 en [0, 1]."""
    img = ds.pixel_array.astype(np.float32)
    min_v = float(np.min(img))
    max_v = float(np.max(img))
    return (img - min_v) / (max_v - min_v + 1e-8)


def _crear_mascara_automatica(img_norm: np.ndarray) -> np.ndarray:
    """Crea mascara binaria heuristica en base a zonas brillantes."""
    high_percentile = 0.98
    threshold = float(np.quantile(img_norm, high_percentile))
    mask = (img_norm >= threshold).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = 1 + int(np.argmax(areas))
        mask = (labels == max_idx).astype(np.uint8)

    return mask


def _contours_to_yolo_seg_lines(mask: np.ndarray, min_area: float = 30.0) -> list[str]:
    """Convierte mascara binaria a lineas YOLO-Seg (clase 0)."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines: list[str] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Suavizado leve del poligono para evitar segmentos excesivos.
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        points = approx.reshape(-1, 2)
        if points.shape[0] < 3:
            continue

        coords = []
        for x, y in points:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")

        line = "0 " + " ".join(coords)
        lines.append(line)

    return lines


def _save_dataset_yaml(output_root: Path) -> None:
    payload = {
        "path": str(output_root.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "tumor"},
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _collect_patient_dirs(input_root: Path, modality: str) -> list[Path]:
    patient_dirs: list[Path] = []
    for patient in sorted(input_root.iterdir()):
        if not patient.is_dir():
            continue
        modality_dir = patient / modality
        if modality_dir.exists():
            patient_dirs.append(patient)
    return patient_dirs


def _prepare_output_dirs(output_root: Path, overwrite: bool) -> None:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)

    required = [
        output_root / "images" / "train",
        output_root / "images" / "val",
        output_root / "labels" / "train",
        output_root / "labels" / "val",
    ]
    for directory in required:
        directory.mkdir(parents=True, exist_ok=True)


def _process_split(
    patients: list[Path],
    split_name: str,
    modality: str,
    output_root: Path,
    imgsz: int,
    min_area: float,
    stats: DatasetStats,
) -> None:
    images_out = output_root / "images" / split_name
    labels_out = output_root / "labels" / split_name

    for patient_dir in patients:
        modality_dir = patient_dir / modality
        dcm_files = sorted(modality_dir.glob("*.dcm"))

        for dcm_file in dcm_files:
            ds = pydicom.dcmread(dcm_file, force=True)
            img_norm = _normalizar_dicom_a_float(ds)
            img_resized = cv2.resize(img_norm, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

            # Imagen RGB uint8 para YOLO.
            img_u8 = (img_resized * 255.0).clip(0, 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

            mask = _crear_mascara_automatica(img_resized)
            lines = _contours_to_yolo_seg_lines(mask, min_area=min_area)

            base_name = f"{patient_dir.name}_{dcm_file.stem}"
            image_path = images_out / f"{base_name}.png"
            label_path = labels_out / f"{base_name}.txt"

            cv2.imwrite(str(image_path), img_rgb)
            label_path.write_text("\n".join(lines), encoding="utf-8")

            stats.total_images += 1
            if split_name == "train":
                stats.train_images += 1
            else:
                stats.val_images += 1

            if lines:
                stats.images_with_label += 1
            else:
                stats.empty_labels += 1


def preparar_dataset_yolo_desde_dcm(
    input_root: str,
    output_root: str,
    modality: str = "T1wCE",
    imgsz: int = 256,
    val_split: float = 0.2,
    seed: int = 42,
    min_area: float = 30.0,
    overwrite: bool = False,
) -> DatasetStats:
    """Convierte DICOM 2D a dataset YOLO-Seg usando pseudo-labels."""
    input_path = Path(input_root).resolve()
    output_path = Path(output_root).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"No existe input_root: {input_path}")

    if overwrite:
        # Proteccion para evitar borrar accidentalmente los datos fuente.
        same_path = output_path == input_path
        output_is_parent_of_input = False
        try:
            input_path.relative_to(output_path)
            output_is_parent_of_input = True
        except ValueError:
            output_is_parent_of_input = False

        if same_path or output_is_parent_of_input:
            raise ValueError(
                "Configuracion insegura: --output-root no puede ser igual al input ni su carpeta padre cuando usas --overwrite."
            )

    patient_dirs = _collect_patient_dirs(input_path, modality)
    if not patient_dirs:
        raise ValueError(
            f"No se encontraron carpetas de pacientes con modalidad '{modality}' en {input_path}"
        )

    random.seed(seed)
    random.shuffle(patient_dirs)

    val_count = max(1, int(len(patient_dirs) * val_split)) if len(patient_dirs) > 1 else 0
    val_patients = patient_dirs[:val_count]
    train_patients = patient_dirs[val_count:]

    if not train_patients and val_patients:
        train_patients, val_patients = val_patients, []

    _prepare_output_dirs(output_path, overwrite=overwrite)
    _save_dataset_yaml(output_path)

    stats = DatasetStats()
    _process_split(train_patients, "train", modality, output_path, imgsz, min_area, stats)
    _process_split(val_patients, "val", modality, output_path, imgsz, min_area, stats)

    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepara dataset YOLO-Seg desde MRI DICOM 2D")
    parser.add_argument(
        "--input-root",
        default="brats_data/Base de datos Brats",
        help="Raiz con carpetas de pacientes",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/brats_yolo",
        help="Directorio de salida YOLO",
    )
    parser.add_argument("--modality", default="T1wCE", help="Modalidad DICOM (T1wCE/T1w/T2w/FLAIR)")
    parser.add_argument("--imgsz", type=int, default=256, help="Tamano de imagen de salida")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraccion para validacion")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--min-area", type=float, default=30.0, help="Area minima de poligono")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra output-root antes de generar el dataset",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    stats = preparar_dataset_yolo_desde_dcm(
        input_root=args.input_root,
        output_root=args.output_root,
        modality=args.modality,
        imgsz=args.imgsz,
        val_split=args.val_split,
        seed=args.seed,
        min_area=args.min_area,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 64)
    print("DATASET YOLO-Seg GENERADO")
    print("=" * 64)
    print(f"Total imagenes:     {stats.total_images}")
    print(f"Train imagenes:     {stats.train_images}")
    print(f"Val imagenes:       {stats.val_images}")
    print(f"Con etiqueta:       {stats.images_with_label}")
    print(f"Etiquetas vacias:   {stats.empty_labels}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
