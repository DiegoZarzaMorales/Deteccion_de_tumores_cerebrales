# ============================================================
# CARGADOR DE DATOS PARA ENTRENAMIENTO
# ============================================================

import os
import warnings
import pickle

# Suprimir warnings de NumPy en este entorno antes de importar numpy
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

import numpy as np
import pydicom
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import random
import re

# Subir esta versión invalida caches antiguos del índice BraTS.
INDEX_CACHE_VERSION = 3

# Orden de canales en BraTS 2020 HDF5: FLAIR, T1, T1ce, T2.
BRATS_HDF5_CHANNEL_NAMES = ("FLAIR", "T1", "T1ce", "T2")
DEFAULT_HDF5_CHANNELS = (0, 1, 2, 3)

# Máscara HDF5 one-hot: ch0=NCR, ch1=ED, ch2=ET → ids 1, 2, 3 (0 = fondo).
BRATS_CLASS_NAMES = ("background", "NCR", "ED", "ET")
NUM_CLASSES = len(BRATS_CLASS_NAMES)


class BraTSDataset(Dataset):
    """
    Dataset para imágenes de tumores cerebrales BraTS
    
    Args:
        root_dir: Directorio raíz con las carpetas de pacientes
        modality: Modalidad de imagen ('T1w', 'T1wCE', 'T2w', 'FLAIR')
        augment: Si True, aplica aumentos aleatorios a imagen y máscara
        size: Tamaño de redimensionamiento (default: 256x256)
        hdf5_empty_keep_ratio: Fracción de cortes HDF5 sin tumor (no negros)
            que se conservan como negativos. 0 = solo tumores; 1 = todas las vacías.
        hdf5_channels: Índices de modalidades HDF5 a usar como entrada
            multimodal (default: las 4 → FLAIR, T1, T1ce, T2).
        num_classes: Clases de salida (4 = fondo + NCR + ED + ET).
    """
    def __init__(
        self,
        root_dir,
        modality='T1wCE',
        augment=False,
        size=256,
        data_list=None,
        verbose=True,
        hdf5_image_channel=None,
        hdf5_channels=DEFAULT_HDF5_CHANNELS,
        hdf5_empty_keep_ratio=0.25,
        hdf5_filter_seed=42,
        num_classes=NUM_CLASSES,
    ):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.augment = augment
        self.size = size
        self.verbose = verbose
        # Compatibilidad: hdf5_image_channel=2 (antiguo T1ce-only) → un solo canal.
        if hdf5_image_channel is not None:
            self.hdf5_channels = (int(hdf5_image_channel),)
        else:
            self.hdf5_channels = tuple(int(c) for c in hdf5_channels)
        if not self.hdf5_channels:
            raise ValueError("hdf5_channels no puede estar vacío")
        self.in_channels = len(self.hdf5_channels)
        self.num_classes = int(num_classes)
        self.hdf5_empty_keep_ratio = float(np.clip(hdf5_empty_keep_ratio, 0.0, 1.0))
        self.hdf5_filter_seed = hdf5_filter_seed
        self._nifti_cache = {}
        self.data_list = data_list if data_list is not None else self._cargar_lista_archivos()

    def _obtener_nifti(self, path):
        """Caché local de imágenes NIfTI por worker para evitar recargar el volumen completo."""
        path = str(path)
        if path not in self._nifti_cache:
            self._nifti_cache[path] = nib.load(path)
        return self._nifti_cache[path]

    def _cargar_hdf5(self, path):
        """Carga una imagen y su máscara desde un archivo HDF5 BraTS."""
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Para usar datasets HDF5 se necesita h5py. "
                "Instálelo con: pip install h5py"
            ) from exc

        with h5py.File(path, "r") as h5_file:
            datasets = {}

            def collect_datasets(name, value):
                if isinstance(value, h5py.Dataset):
                    datasets[name.lower()] = value[()]

            h5_file.visititems(collect_datasets)

        if not datasets:
            raise ValueError(f"El archivo HDF5 no contiene datasets: {path}")

        image = next(
            (value for name, value in datasets.items() if "image" in name or "img" in name),
            None,
        )
        mask = next(
            (value for name, value in datasets.items() if "mask" in name or "label" in name),
            None,
        )

        if image is None or mask is None:
            arrays = list(datasets.values())
            if len(arrays) >= 2:
                image, mask = arrays[0], arrays[1]

        if image is None or mask is None:
            raise ValueError(
                f"No se pudieron identificar imagen y máscara en el HDF5: {path}. "
                f"Datasets encontrados: {list(datasets)}"
            )

        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask)
        if image.ndim == 3:
            max_needed = max(self.hdf5_channels)
            if image.shape[-1] <= max_needed:
                raise ValueError(
                    f"La imagen HDF5 no tiene los canales {self.hdf5_channels}: {path}"
                )
            image = image[..., list(self.hdf5_channels)]
            if image.ndim == 3 and image.shape[-1] == 1:
                image = image[..., 0]
        mask = self._mascara_hdf5_a_clases(mask)
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Se esperaba imagen 2D o HWC multimodal tras seleccionar canales: {path}"
            )

        return image, mask

    @staticmethod
    def _mascara_hdf5_a_clases(mask):
        """Convierte máscara HDF5 one-hot (H,W,3) a mapa de clases (H,W)."""
        mask = np.asarray(mask)
        if mask.ndim == 2:
            return (mask > 0).astype(np.int64)

        if mask.ndim != 3:
            raise ValueError(f"Máscara HDF5 con forma no soportada: {mask.shape}")

        labels = np.zeros(mask.shape[:2], dtype=np.int64)
        # Canales: 0→NCR(1), 1→ED(2), 2→ET(3). Sin solapes en BraTS HDF5.
        for channel_index in range(min(3, mask.shape[-1])):
            labels[mask[..., channel_index] > 0] = channel_index + 1
        return labels

    @staticmethod
    def _mascara_nifti_a_clases(seg_slice):
        """Mapea etiquetas BraTS NIfTI (0,1,2,4) a (0,1,2,3)."""
        seg = np.asarray(seg_slice)
        labels = np.zeros(seg.shape, dtype=np.int64)
        labels[seg == 1] = 1  # NCR
        labels[seg == 2] = 2  # ED
        labels[seg == 4] = 3  # ET
        # Si ya viniera como 1/2/3, respetarlo.
        labels[seg == 3] = 3
        return labels

    @staticmethod
    def _normalizar_imagen(img):
        """Normaliza a [0, 1]; si es multimodal, por canal."""
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 2:
            return (img - img.min()) / (img.max() - img.min() + 1e-8)

        out = np.empty_like(img, dtype=np.float32)
        for channel in range(img.shape[-1]):
            ch = img[..., channel]
            out[..., channel] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        return out

    def _imagen_a_chw(self, img):
        """Convierte HW o HWC al tensor CHW con self.in_channels."""
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img] * self.in_channels, axis=0)
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
            if img.shape[0] == 1 and self.in_channels > 1:
                img = np.repeat(img, self.in_channels, axis=0)
            elif img.shape[0] != self.in_channels:
                raise ValueError(
                    f"Canales de imagen {img.shape[0]} != in_channels={self.in_channels}"
                )
        else:
            raise ValueError(f"Forma de imagen no soportada: {img.shape}")
        return np.ascontiguousarray(img, dtype=np.float32)

    @staticmethod
    def _import_h5py():
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Para usar datasets HDF5 se necesita h5py. "
                "Instálelo con: pip install h5py"
            ) from exc
        return h5py

    @staticmethod
    def _hdf5_tiene_tumor(mask):
        """True si la máscara HDF5 contiene algún píxel de tumor."""
        mask = np.asarray(mask)
        return bool(np.any(mask > 0))

    @staticmethod
    def _hdf5_imagen_negra(image):
        """True si la imagen es completamente cero (corte sin señal)."""
        image = np.asarray(image)
        return not bool(np.any(image))

    def _clasificar_hdf5(self, path, h5py_module):
        """Devuelve 'tumor', 'empty' o 'black' para un archivo HDF5."""
        with h5py_module.File(path, "r") as h5_file:
            if "image" in h5_file:
                image = h5_file["image"][()]
            else:
                image = next(iter(h5_file.values()))[()]

            if self._hdf5_imagen_negra(image):
                return "black"

            if "mask" in h5_file:
                mask = h5_file["mask"][()]
            else:
                values = list(h5_file.values())
                mask = values[1][()] if len(values) > 1 else None

            if mask is None:
                return "empty"

            return "tumor" if self._hdf5_tiene_tumor(mask) else "empty"

    def _cargar_meta_targets_hdf5(self, hdf5_files):
        """Usa meta_data.csv si existe para saber qué cortes tienen tumor."""
        meta_candidates = [
            self.root_dir / "meta_data.csv",
            *self.root_dir.rglob("meta_data.csv"),
        ]
        meta_path = next((path for path in meta_candidates if path.is_file()), None)
        if meta_path is None:
            return None

        try:
            import csv

            name_to_target = {}
            with open(meta_path, "r", encoding="utf-8", newline="") as meta_file:
                reader = csv.DictReader(meta_file)
                if not reader.fieldnames or "target" not in reader.fieldnames:
                    return None
                path_field = "slice_path" if "slice_path" in reader.fieldnames else None
                for row in reader:
                    raw_path = row.get(path_field) if path_field else None
                    name = Path(raw_path).name if raw_path else None
                    if not name:
                        continue
                    try:
                        name_to_target[name] = int(float(row["target"]))
                    except (TypeError, ValueError):
                        continue

            if not name_to_target:
                return None

            targets = {}
            for path in hdf5_files:
                if path.name in name_to_target:
                    targets[path] = name_to_target[path.name]
            return targets if targets else None
        except Exception:
            return None

    def _filtrar_archivos_hdf5(self, hdf5_files):
        """Quita cortes negros y conserva solo una fracción de máscaras vacías."""
        if not hdf5_files:
            return []

        h5py = self._import_h5py()
        tumor_files = []
        empty_files = []
        black_count = 0
        unread_count = 0

        meta_targets = self._cargar_meta_targets_hdf5(hdf5_files)
        if self.verbose and meta_targets is not None:
            print(
                f"Filtrado HDF5: usando meta_data.csv "
                f"({len(meta_targets)} rutas etiquetadas)"
            )

        total = len(hdf5_files)
        for index, path in enumerate(hdf5_files, start=1):
            if self.verbose and (index == 1 or index % 5000 == 0 or index == total):
                print(f"Filtrado HDF5: revisando {index}/{total}...")

            try:
                if meta_targets is not None and path in meta_targets:
                    if meta_targets[path] > 0:
                        tumor_files.append(path)
                        continue

                    # Vacías según metadata: abrir solo para descartar negros.
                    with h5py.File(path, "r") as h5_file:
                        image = h5_file["image"][()] if "image" in h5_file else next(iter(h5_file.values()))[()]
                    if self._hdf5_imagen_negra(image):
                        black_count += 1
                    else:
                        empty_files.append(path)
                    continue

                label = self._clasificar_hdf5(path, h5py)
                if label == "tumor":
                    tumor_files.append(path)
                elif label == "empty":
                    empty_files.append(path)
                else:
                    black_count += 1
            except Exception:
                unread_count += 1

        keep_n = int(round(len(empty_files) * self.hdf5_empty_keep_ratio))
        if keep_n <= 0 or not empty_files:
            kept_empty = []
        elif keep_n >= len(empty_files):
            kept_empty = list(empty_files)
        else:
            rng = random.Random(self.hdf5_filter_seed)
            kept_empty = rng.sample(empty_files, keep_n)

        selected = sorted(tumor_files + kept_empty, key=lambda path: str(path).lower())

        if self.verbose:
            print(
                "Filtrado HDF5:\n"
                f"  - Candidatos: {total}\n"
                f"  - Con tumor: {len(tumor_files)}\n"
                f"  - Vacías útiles: {len(empty_files)} "
                f"(se conservan {len(kept_empty)}, "
                f"ratio={self.hdf5_empty_keep_ratio:.2f})\n"
                f"  - Negras descartadas: {black_count}\n"
                f"  - No legibles: {unread_count}\n"
                f"  - Total final HDF5: {len(selected)}"
            )

        return selected

    def _cargar_lista_archivos(self):
        """Carga muestras DICOM, NIfTI o HDF5 disponibles."""
        data_list = []

        hdf5_files = sorted(
            set(self.root_dir.rglob("*.h5")) | set(self.root_dir.rglob("*.hdf5"))
        )
        selected_hdf5 = self._filtrar_archivos_hdf5(hdf5_files)
        data_list.extend(("hdf5", path) for path in selected_hdf5)
        
        # Buscar todas las carpetas de pacientes
        for patient_folder in self.root_dir.iterdir():
            if patient_folder.is_dir():
                # Primero buscar DICOMs en la estructura esperada
                modality_path = patient_folder / self.modality
                if modality_path.exists():
                    # Obtener todos los archivos .dcm
                    dcm_files = sorted(list(modality_path.glob("*.dcm")))
                    for dcm_file in dcm_files:
                        data_list.append(dcm_file)
                    continue

                # Si no hay DICOMs, intentar detectar NIfTI (*.nii, *.nii.gz)
                # Buscamos un fichero de imagen correspondiente a la modalidad y su máscara
                nifti_image = None
                nifti_seg = None

                # claves simples para mapear modalidades a fragmentos de nombre comunes
                modality_map = {
                    'T1wCE': ['t1c', 't1ce', 't1_w_ce', 't1wce'],
                    'T1w': ['t1n', 't1', 't1w'],
                    'T2w': ['t2w', 't2'],
                    'FLAIR': ['t2f', 'flair'],
                }

                search_keys = modality_map.get(self.modality, [self.modality.lower()])

                for f in patient_folder.rglob('*.nii*'):
                    name = f.name.lower()
                    # detectar máscara
                    if 'seg' in name or 'mask' in name:
                        nifti_seg = f
                        continue

                    # detectar imagen de la modalidad
                    for key in search_keys:
                        if key in name:
                            nifti_image = f
                            break
                    if nifti_image:
                        # seguir buscando máscara pero no romper el loop
                        continue

                # Si encontramos imagen y máscara, generar entradas por 'slice'
                if nifti_image and nifti_seg:
                    try:
                        img_obj = nib.load(str(nifti_image))
                        seg_obj = nib.load(str(nifti_seg))
                        img_shape = img_obj.shape
                        seg_shape = seg_obj.shape

                        # asumimos que la dimensión de cortes es la última (x,y,z)
                        if len(img_shape) != 3 or len(seg_shape) != 3:
                            continue

                        num_slices = img_shape[2]
                        for z in range(num_slices):
                            # incluir sólo slices con máscaras presentes para mejorar calidad
                            seg_slice = np.asanyarray(seg_obj.dataobj[..., z])
                            if seg_slice.sum() > 0:
                                # guardamos tupla (tipo, image_path, seg_path, slice_index)
                                data_list.append(('nifti', nifti_image, nifti_seg, z))
                    except Exception:
                        # si falla la lectura, ignorar este paciente
                        continue
        
        if self.verbose:
            channel_names = [
                BRATS_HDF5_CHANNEL_NAMES[c]
                if 0 <= c < len(BRATS_HDF5_CHANNEL_NAMES)
                else str(c)
                for c in self.hdf5_channels
            ]
            print(f"Total de imágenes encontradas ({self.modality}): {len(data_list)}")
            print(
                f"Entrada multimodal HDF5: {self.in_channels} canales "
                f"{self.hdf5_channels} → {channel_names}"
            )
            print(
                f"Salida multiclase: {self.num_classes} clases → {BRATS_CLASS_NAMES}"
            )
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        mask = None

        if isinstance(item, tuple) and item[0] == 'hdf5':
            _, hdf5_path = item
            img, mask = self._cargar_hdf5(hdf5_path)
        # Si el item es una tupla creada a partir de NIfTI
        elif isinstance(item, tuple) and item[0] == 'nifti':
            _, img_path, seg_path, z = item
            img_obj = self._obtener_nifti(img_path)
            seg_obj = self._obtener_nifti(seg_path)

            # tomar slice en eje z
            img = np.asanyarray(img_obj.dataobj[..., int(z)], dtype=np.float32)
            mask = self._mascara_nifti_a_clases(
                np.asanyarray(seg_obj.dataobj[..., int(z)])
            )
        else:
            # Cargar imagen DICOM
            dcm_path = self.data_list[idx]
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype(np.float32)
        
        # Normalizar (por canal si es multimodal)
        img = self._normalizar_imagen(img)
        
        # Redimensionar
        img = cv2.resize(img, (self.size, self.size))
        if mask is not None:
            mask = cv2.resize(
                mask.astype(np.float32),
                (self.size, self.size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
        else:
            # DICOM sin anotación: pseudo-máscara → clase NCR genérica (1)
            mask = self._crear_mascara_automatica(img)

        if self.augment:
            img, mask = self._aplicar_augmentacion_sincronizada(img, mask)
        
        # Convertir a tensores: imagen CHW, máscara HW con ids de clase
        img = torch.from_numpy(self._imagen_a_chw(img))  # (C, H, W)
        mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))  # (H, W)
        
        return img, mask

    def _aplicar_augmentacion_sincronizada(self, img, mask):
        """Aplica augmentaciones geométricas iguales a imagen y máscara."""
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        k = random.randint(0, 3)
        if k:
            # HW o HWC: rotar en el plano espacial (ejes 0 y 1)
            img = np.rot90(img, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return img, mask
    
    def _crear_mascara_automatica(self, img):
        """Crea una máscara automática usando brillo (top 2% y componente principal)."""
        # Si es multimodal, usar T1ce (canal 2) o el último disponible como referencia.
        if img.ndim == 3:
            ref_idx = min(2, img.shape[-1] - 1)
            img_ref = img[..., ref_idx]
        else:
            img_ref = img

        # 1) Normalizar por seguridad
        img_norm = (img_ref - img_ref.min()) / (img_ref.max() - img_ref.min() + 1e-8)

        # 2) Tomar solo los píxeles más brillantes (ej. top 2%)
        high_percentile = 0.98  # top 2%
        threshold = np.quantile(img_norm, high_percentile)
        mask = (img_norm >= threshold).astype(np.uint8)

        # 3) Limpieza morfológica suave
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4) Conservar solo la región brillante principal (componente conexo más grande)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # ignorar fondo (índice 0)
            max_idx = 1 + np.argmax(areas)
            mask = (labels == max_idx).astype(np.int64)
        else:
            mask = mask.astype(np.int64)

        # Fallback DICOM: marcar tumor genérico como clase 1 (NCR).
        return mask


def cargar_o_construir_indice_dataset(
    root_dir,
    modality='T1wCE',
    verbose=True,
    usar_cache=True,
    hdf5_empty_keep_ratio=0.25,
    hdf5_filter_seed=42,
    hdf5_channels=DEFAULT_HDF5_CHANNELS,
):
    """Carga el índice de muestras desde cache o lo construye una sola vez.

    Esto evita repetir el escaneo completo del dataset en cada ejecución de entrenar.py.
    """
    root_path = Path(root_dir)
    cache_path = root_path / f".brats_index_cache_{modality.lower()}.pkl"
    empty_keep_ratio = float(np.clip(hdf5_empty_keep_ratio, 0.0, 1.0))
    channels = tuple(int(c) for c in hdf5_channels)

    if usar_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as cache_file:
                cache_data = pickle.load(cache_file)

            if (
                isinstance(cache_data, dict)
                and cache_data.get('version') == INDEX_CACHE_VERSION
                and cache_data.get('root_dir') == str(root_path.resolve())
                and cache_data.get('modality') == modality
                and cache_data.get('hdf5_empty_keep_ratio') == empty_keep_ratio
                and cache_data.get('hdf5_filter_seed') == hdf5_filter_seed
                and isinstance(cache_data.get('data_list'), list)
                and len(cache_data['data_list']) > 0
            ):
                if verbose:
                    print(f"Índice de dataset cargado desde cache: {cache_path.name}")
                return cache_data['data_list']
        except Exception:
            pass

    dataset = BraTSDataset(
        root_path,
        modality=modality,
        augment=False,
        verbose=verbose,
        hdf5_channels=channels,
        hdf5_empty_keep_ratio=empty_keep_ratio,
        hdf5_filter_seed=hdf5_filter_seed,
    )
    data_list = dataset.data_list

    if usar_cache and len(data_list) > 0:
        try:
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(
                    {
                        'version': INDEX_CACHE_VERSION,
                        'root_dir': str(root_path.resolve()),
                        'modality': modality,
                        'hdf5_empty_keep_ratio': empty_keep_ratio,
                        'hdf5_filter_seed': hdf5_filter_seed,
                        'hdf5_channels': channels,
                        'data_list': data_list,
                    },
                    cache_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            if verbose:
                print(f"Índice de dataset guardado en cache: {cache_path.name}")
        except Exception:
            pass

    return data_list


def crear_dataloaders(
    root_dir,
    batch_size=4,
    train_split=0.8,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=False,
    hdf5_empty_keep_ratio=0.25,
    hdf5_filter_seed=42,
    hdf5_channels=DEFAULT_HDF5_CHANNELS,
    num_classes=NUM_CLASSES,
): # por defecto 2
    """Crea dataloaders para entrenamiento y validación.

    Returns:
        train_loader, val_loader, in_channels, num_classes
    """
    channels = tuple(int(c) for c in hdf5_channels)
    # Cargar índice desde cache para evitar escanear todo el dataset en cada ejecución
    full_data_list = cargar_o_construir_indice_dataset(
        root_dir,
        modality='T1wCE',
        verbose=True,
        usar_cache=True,
        hdf5_empty_keep_ratio=hdf5_empty_keep_ratio,
        hdf5_filter_seed=hdf5_filter_seed,
        hdf5_channels=channels,
    )
    full_dataset = BraTSDataset(
        root_dir,
        modality='T1wCE',
        augment=False,
        data_list=full_data_list,
        verbose=False,
        hdf5_channels=channels,
        hdf5_empty_keep_ratio=hdf5_empty_keep_ratio,
        hdf5_filter_seed=hdf5_filter_seed,
        num_classes=num_classes,
    )

    if len(full_dataset) == 0:
        raise ValueError(
            f"No se encontraron imágenes para entrenamiento en {root_dir}. "
            "Revisa que la ruta apunte a BraTS-PEDs-v1/Training o a una carpeta con pacientes BraTS-PED."
        )
    
    # Separar por volumen/paciente para no mezclar cortes relacionados entre splits.
    groups = {}
    for index, item in enumerate(full_data_list):
        if isinstance(item, tuple) and item[0] == "hdf5":
            match = re.search(r"volume_(\d+)_", Path(item[1]).name.lower())
            group_key = f"hdf5-volume-{match.group(1)}" if match else str(item[1])
        elif isinstance(item, tuple) and item[0] == "nifti":
            group_key = str(item[1])
        else:
            group_key = str(Path(item).parent.parent)
        groups.setdefault(group_key, []).append(index)

    group_keys = list(groups)
    random.shuffle(group_keys)
    train_group_count = max(1, int(train_split * len(group_keys)))
    if len(group_keys) > 1:
        train_group_count = min(train_group_count, len(group_keys) - 1)
    train_indices = [
        index for key in group_keys[:train_group_count] for index in groups[key]
    ]
    val_indices = [
        index for key in group_keys[train_group_count:] for index in groups[key]
    ]

    train_dataset = torch.utils.data.Subset(
        BraTSDataset(
            root_dir,
            modality='T1wCE',
            augment=True,
            data_list=full_data_list,
            verbose=False,
            hdf5_channels=channels,
            hdf5_empty_keep_ratio=hdf5_empty_keep_ratio,
            hdf5_filter_seed=hdf5_filter_seed,
            num_classes=num_classes,
        ),
        train_indices,
    )
    val_dataset = torch.utils.data.Subset(
        BraTSDataset(
            root_dir,
            modality='T1wCE',
            augment=False,
            data_list=full_data_list,
            verbose=False,
            hdf5_channels=channels,
            hdf5_empty_keep_ratio=hdf5_empty_keep_ratio,
            hdf5_filter_seed=hdf5_filter_seed,
            num_classes=num_classes,
        ),
        val_indices,
    )
    
    # Dataloaders (optimización: pin_memory, persistent_workers, prefetch_factor)
    pin_memory_flag = True if torch.cuda.is_available() else False

    loader_options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory_flag,
    }
    if num_workers > 0:
        loader_options.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    in_channels = full_dataset.in_channels
    print(f"\nDataloaders creados:")
    print(f"  - Training: {len(train_dataset)} imágenes ({len(train_loader)} batches)")
    print(f"  - Validation: {len(val_dataset)} imágenes ({len(val_loader)} batches)")
    print(f"  - Canales de entrada: {in_channels} {channels}")
    print(f"  - Clases de salida: {num_classes} {BRATS_CLASS_NAMES}")
    
    return train_loader, val_loader, in_channels, num_classes


if __name__ == "__main__":
    # Prueba rápida usando la carpeta brats_data relative al proyecto
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(base_dir, "brats_data", "Base de datos Brats")

    train_loader, val_loader, in_channels, num_classes = crear_dataloaders(root, batch_size=2)
    
    imgs, masks = next(iter(train_loader))
    print(f"\nBatch shape:")
    print(f"  Images: {imgs.shape} (in_channels={in_channels})")
    print(f"  Masks: {masks.shape} (num_classes={num_classes})")
