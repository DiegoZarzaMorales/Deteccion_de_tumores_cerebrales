# ============================================================
# CARGADOR DE DATOS PARA ENTRENAMIENTO
# ============================================================

import os
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path


class BraTSDataset(Dataset):
    """
    Dataset para imágenes de tumores cerebrales BraTS
    
    Args:
        root_dir: Directorio raíz con las carpetas de pacientes
        modality: Modalidad de imagen ('T1w', 'T1wCE', 'T2w', 'FLAIR')
        transform: Transformaciones a aplicar
        size: Tamaño de redimensionamiento (default: 256x256)
    """
    def __init__(self, root_dir, modality='T1wCE', transform=None, size=256):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.transform = transform
        self.size = size
        self.data_list = self._cargar_lista_archivos()
        
    def _cargar_lista_archivos(self):
        """Carga la lista de todos los archivos DICOM disponibles"""
        data_list = []
        
        # Buscar todas las carpetas de pacientes
        for patient_folder in self.root_dir.iterdir():
            if patient_folder.is_dir():
                modality_path = patient_folder / self.modality
                if modality_path.exists():
                    # Obtener todos los archivos .dcm
                    dcm_files = sorted(list(modality_path.glob("*.dcm")))
                    for dcm_file in dcm_files:
                        data_list.append(dcm_file)
        
        print(f"Total de imágenes encontradas ({self.modality}): {len(data_list)}")
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # Cargar imagen DICOM
        dcm_path = self.data_list[idx]
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
        
        # Normalizar
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Redimensionar
        img = cv2.resize(img, (self.size, self.size))
        
        # Crear máscara (aquí usaremos detección automática como pseudo-label)
        # En un dataset real, tendrías máscaras ground truth
        mask = self._crear_mascara_automatica(img)
        
        # Convertir a tensores
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        # Aplicar transformaciones
        if self.transform:
            img = self.transform(img)
        
        return img, mask
    
    def _crear_mascara_automatica(self, img):
        """
        Crea una máscara automática usando umbralización
        NOTA: En un proyecto real, deberías tener máscaras anotadas por expertos
        """
        # 1) Normalizar por seguridad
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 2) Tomar solo los píxeles más brillantes (ej. top 2%)
        #    Esto refleja mejor que el tumor en T1wCE es la zona más brillante
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


def crear_dataloaders(root_dir, batch_size=4, train_split=0.8, num_workers=0):
    """
    Crea dataloaders para entrenamiento y validación
    
    Args:
        root_dir: Directorio raíz de datos
        batch_size: Tamaño del batch
        train_split: Proporción de datos para entrenamiento
        num_workers: Número de workers para carga
        
    Returns:
        train_loader, val_loader
    """
    # Transformaciones básicas
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
    ])
    
    # Dataset completo
    full_dataset = BraTSDataset(root_dir, modality='T1wCE', transform=transform)
    
    # Split train/val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataloaders creados:")
    print(f"  - Training: {len(train_dataset)} imágenes ({len(train_loader)} batches)")
    print(f"  - Validation: {len(val_dataset)} imágenes ({len(val_loader)} batches)")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Prueba
    root = r"c:\Users\josez\Documents\DisenoDeInterfaz\brats_data\Base de datos Brats"
    train_loader, val_loader = crear_dataloaders(root, batch_size=2)
    
    # Mostrar un batch
    imgs, masks = next(iter(train_loader))
    print(f"\nBatch shape:")
    print(f"  Images: {imgs.shape}")
    print(f"  Masks: {masks.shape}")
