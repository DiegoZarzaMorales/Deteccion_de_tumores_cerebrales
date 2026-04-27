"""Convierte archivos .mat a imagenes PNG/JPG.

Soporta:
- .mat clasico (scipy.io.loadmat)
- .mat v7.3 (HDF5) si h5py esta instalado

Casos:
- Array 2D: guarda una imagen
- Array 3D: puede guardar slice central o todas las slices
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scipy.io import loadmat
except Exception as exc:
    raise ImportError(
        "No se pudo importar scipy.io.loadmat. Instala scipy con: pip install scipy"
    ) from exc


def _list_user_keys(payload: dict[str, Any]) -> list[str]:
    return [k for k in payload.keys() if not k.startswith("__")]


def _select_numeric_key(payload: dict[str, Any], preferred_key: str | None = None) -> str:
    keys = _list_user_keys(payload)
    if not keys:
        raise ValueError("El .mat no contiene variables de usuario.")

    if preferred_key:
        if preferred_key not in payload:
            raise KeyError(f"No existe la clave '{preferred_key}' en el .mat")
        arr = np.asarray(payload[preferred_key])
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"La clave '{preferred_key}' no es numerica")
        return preferred_key

    numeric_keys = []
    for key in keys:
        arr = np.asarray(payload[key])
        if np.issubdtype(arr.dtype, np.number):
            numeric_keys.append((key, arr.size))

    if not numeric_keys:
        raise ValueError("No se encontraron arrays numericos en el .mat")

    numeric_keys.sort(key=lambda x: x[1], reverse=True)
    return numeric_keys[0][0]


def _read_mat(path: Path, key: str | None = None) -> tuple[np.ndarray, str]:
    try:
        payload = loadmat(path)
        selected_key = _select_numeric_key(payload, preferred_key=key)
        arr = np.asarray(payload[selected_key])
        return arr, selected_key
    except NotImplementedError:
        # Probable .mat v7.3 (HDF5)
        try:
            import h5py
        except Exception as exc:
            raise ImportError(
                "Este .mat parece v7.3 (HDF5). Instala h5py con: pip install h5py"
            ) from exc

        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys()]
            if not keys:
                raise ValueError("El .mat v7.3 no contiene datasets")

            if key:
                if key not in f:
                    raise KeyError(f"No existe la clave '{key}' en el .mat")
                arr = np.array(f[key])
                selected_key = key
            else:
                best_key = None
                best_size = -1
                for k in keys:
                    candidate = np.array(f[k])
                    if np.issubdtype(candidate.dtype, np.number) and candidate.size > best_size:
                        best_key = k
                        best_size = candidate.size
                if best_key is None:
                    raise ValueError("No se encontraron datasets numericos en el .mat")
                arr = np.array(f[best_key])
                selected_key = best_key

            # En HDF5 suele venir transpuesto respecto a MATLAB.
            arr = np.squeeze(arr)
            return arr, selected_key


def _normalize_to_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_v = float(np.min(img))
    max_v = float(np.max(img))
    if max_v - min_v < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img_n = (img - min_v) / (max_v - min_v)
    return (img_n * 255.0).clip(0, 255).astype(np.uint8)


def _save_2d(arr2d: np.ndarray, out_file: Path) -> None:
    img_u8 = _normalize_to_u8(arr2d)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_file), img_u8)
    if not ok:
        raise IOError(f"No se pudo guardar imagen: {out_file}")


def _iter_slices(arr3d: np.ndarray, axis: int):
    num = arr3d.shape[axis]
    for i in range(num):
        if axis == 0:
            yield i, arr3d[i, :, :]
        elif axis == 1:
            yield i, arr3d[:, i, :]
        else:
            yield i, arr3d[:, :, i]


def _guess_slice_axis(arr3d: np.ndarray) -> int:
    # En MRI 3D suele haber una dimension de slices menor que H y W.
    return int(np.argmin(arr3d.shape))


def convertir_mat(
    input_path: Path,
    output_dir: Path,
    fmt: str = "png",
    key: str | None = None,
    all_slices: bool = False,
    slice_axis: int | None = None,
    center_slice: bool = True,
) -> list[Path]:
    arr, used_key = _read_mat(input_path, key=key)
    arr = np.squeeze(arr)

    if arr.ndim < 2:
        raise ValueError(f"Array invalido ndim={arr.ndim} en {input_path}")

    stem = input_path.stem
    out_paths: list[Path] = []

    if arr.ndim == 2:
        out_file = output_dir / f"{stem}_{used_key}.{fmt}"
        _save_2d(arr, out_file)
        out_paths.append(out_file)
        return out_paths

    if arr.ndim > 3:
        # Si viene con canales extra, tomar primer canal hasta quedar en 3D.
        while arr.ndim > 3:
            arr = arr[..., 0]

    axis = _guess_slice_axis(arr) if slice_axis is None else slice_axis

    if all_slices:
        for i, slc in _iter_slices(arr, axis):
            out_file = output_dir / f"{stem}_{used_key}_slice_{i:04d}.{fmt}"
            _save_2d(slc, out_file)
            out_paths.append(out_file)
        return out_paths

    if center_slice:
        center_i = arr.shape[axis] // 2
        if axis == 0:
            slc = arr[center_i, :, :]
        elif axis == 1:
            slc = arr[:, center_i, :]
        else:
            slc = arr[:, :, center_i]
        out_file = output_dir / f"{stem}_{used_key}_slice_{center_i:04d}.{fmt}"
        _save_2d(slc, out_file)
        out_paths.append(out_file)
        return out_paths

    # Fallback: primera slice.
    i, slc = next(_iter_slices(arr, axis))
    out_file = output_dir / f"{stem}_{used_key}_slice_{i:04d}.{fmt}"
    _save_2d(slc, out_file)
    out_paths.append(out_file)
    return out_paths


def _collect_mat_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.mat" if recursive else "*.mat"
    return sorted(input_path.glob(pattern))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convierte .mat a PNG/JPG")
    parser.add_argument("--input", required=True, help="Archivo .mat o carpeta con .mat")
    parser.add_argument("--output", default="datasets/mat_export", help="Carpeta de salida")
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png")
    parser.add_argument("--key", default=None, help="Clave del array dentro del .mat")
    parser.add_argument("--recursive", action="store_true", help="Buscar .mat recursivamente")
    parser.add_argument("--all-slices", action="store_true", help="Exportar todas las slices de volumen 3D")
    parser.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--no-center-slice", action="store_true", help="No usar slice central (si no all-slices)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"No existe input: {input_path}")

    mat_files = _collect_mat_files(input_path, recursive=args.recursive)
    if not mat_files:
        raise FileNotFoundError(f"No se encontraron .mat en: {input_path}")

    total_out = 0
    print(f"Archivos .mat encontrados: {len(mat_files)}")

    for mat_file in mat_files:
        rel_parent = mat_file.parent.relative_to(input_path.parent) if input_path.is_file() else mat_file.parent.relative_to(input_path)
        out_subdir = output_dir / rel_parent
        out_paths = convertir_mat(
            input_path=mat_file,
            output_dir=out_subdir,
            fmt=args.format,
            key=args.key,
            all_slices=args.all_slices,
            slice_axis=args.slice_axis,
            center_slice=not args.no_center_slice,
        )
        total_out += len(out_paths)
        print(f"OK {mat_file.name} -> {len(out_paths)} imagen(es)")

    print("=" * 64)
    print(f"Conversion finalizada. Imagenes generadas: {total_out}")
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
