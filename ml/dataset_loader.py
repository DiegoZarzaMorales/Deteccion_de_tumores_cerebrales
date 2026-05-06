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


class BraTSDataset(Dataset):
    """
    Dataset para imágenes de tumores cerebrales BraTS
    
    Args:
        root_dir: Directorio raíz con las carpetas de pacientes
        modality: Modalidad de imagen ('T1w', 'T1wCE', 'T2w', 'FLAIR')
        augment: Si True, aplica aumentos aleatorios a imagen y máscara
        size: Tamaño de redimensionamiento (default: 256x256)
    """
    def __init__(self, root_dir, modality='T1wCE', augment=False, size=256, data_list=None, verbose=True):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.augment = augment
        self.size = size
        self.verbose = verbose
        self._nifti_cache = {}
        self.data_list = data_list if data_list is not None else self._cargar_lista_archivos()

    def _obtener_nifti(self, path):
        """Caché local de imágenes NIfTI por worker para evitar recargar el volumen completo."""
        path = str(path)
        if path not in self._nifti_cache:
            self._nifti_cache[path] = nib.load(path)
        return self._nifti_cache[path]
        
    def _cargar_lista_archivos(self):
        """Carga la lista de todos los archivos DICOM disponibles"""
        data_list = []
        
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
            print(f"Total de imágenes encontradas ({self.modality}): {len(data_list)}")
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]

        # Si el item es una tupla creada a partir de NIfTI
        if isinstance(item, tuple) and item[0] == 'nifti':
            _, img_path, seg_path, z = item
            img_obj = self._obtener_nifti(img_path)
            seg_obj = self._obtener_nifti(seg_path)

            # tomar slice en eje z
            img = np.asanyarray(img_obj.dataobj[..., int(z)], dtype=np.float32)
            mask = (np.asanyarray(seg_obj.dataobj[..., int(z)]) > 0).astype(np.float32)
        else:
            # Cargar imagen DICOM
            dcm_path = self.data_list[idx]
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype(np.float32)
        
        # Normalizar
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Redimensionar
        img = cv2.resize(img, (self.size, self.size))
        if 'mask' in locals():
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        # Si no venimos de NIfTI, crear máscara automática (pseudo-label)
        if 'mask' not in locals():
            mask = self._crear_mascara_automatica(img)

        if self.augment:
            img, mask = self._aplicar_augmentacion_sincronizada(img, mask)
        
        # Convertir a tensores
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
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
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        return img, mask
    
    def _crear_mascara_automatica(self, img):
        """Crea una máscara automática usando brillo (top 2% y componente principal)."""
        # 1) Normalizar por seguridad
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

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
            mask = (labels == max_idx).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        return mask


def cargar_o_construir_indice_dataset(root_dir, modality='T1wCE', verbose=True, usar_cache=True):
    """Carga el índice de muestras desde cache o lo construye una sola vez.

    Esto evita repetir el escaneo completo del dataset en cada ejecución de entrenar.py.
    """
    root_path = Path(root_dir)
    cache_path = root_path / f".brats_index_cache_{modality.lower()}.pkl"

    if usar_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as cache_file:
                cache_data = pickle.load(cache_file)

            if (
                isinstance(cache_data, dict)
                and cache_data.get('version') == 1
                and cache_data.get('root_dir') == str(root_path.resolve())
                and cache_data.get('modality') == modality
                and isinstance(cache_data.get('data_list'), list)
                and len(cache_data['data_list']) > 0
            ):
                if verbose:
                    print(f"Índice de dataset cargado desde cache: {cache_path.name}")
                return cache_data['data_list']
        except Exception:
            pass

    dataset = BraTSDataset(root_path, modality=modality, augment=False, verbose=verbose)
    data_list = dataset.data_list

    if usar_cache and len(data_list) > 0:
        try:
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(
                    {
                        'version': 1,
                        'root_dir': str(root_path.resolve()),
                        'modality': modality,
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


def crear_dataloaders(root_dir, batch_size=4, train_split=0.8, num_workers=2, prefetch_factor=2, persistent_workers=False): # por defecto 2
    """Crea dataloaders para entrenamiento y validación"""
    # Cargar índice desde cache para evitar escanear todo el dataset en cada ejecución
    full_data_list = cargar_o_construir_indice_dataset(root_dir, modality='T1wCE', verbose=True, usar_cache=True)
    full_dataset = BraTSDataset(root_dir, modality='T1wCE', augment=False, data_list=full_data_list, verbose=False)

    if len(full_dataset) == 0:
        raise ValueError(
            f"No se encontraron imágenes para entrenamiento en {root_dir}. "
            "Revisa que la ruta apunte a BraTS-PEDs-v1/Training o a una carpeta con pacientes BraTS-PED."
        )
    
    # Split train/val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(
        BraTSDataset(
            root_dir,
            modality='T1wCE',
            augment=True,
            data_list=full_data_list,
            verbose=False,
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
        ),
        val_indices,
    )
    
    # Dataloaders (optimización: pin_memory, persistent_workers, prefetch_factor)
    pin_memory_flag = True if torch.cuda.is_available() else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
    )
    
    print(f"\nDataloaders creados:")
    print(f"  - Training: {len(train_dataset)} imágenes ({len(train_loader)} batches)")
    print(f"  - Validation: {len(val_dataset)} imágenes ({len(val_loader)} batches)")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Prueba rápida usando la carpeta brats_data relative al proyecto
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(base_dir, "brats_data", "Base de datos Brats")

    train_loader, val_loader = crear_dataloaders(root, batch_size=2)
    
    imgs, masks = next(iter(train_loader))
    print(f"\nBatch shape:")
    print(f"  Images: {imgs.shape}")
    print(f"  Masks: {masks.shape}")
