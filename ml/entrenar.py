# ============================================================
# ENTRENAMIENTO DEL MODELO U-NET
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from .modelo_unet import crear_modelo
from .dataset_loader import BRATS_CLASS_NAMES, crear_dataloaders


def resolver_dispositivo(device=None):
    """Resuelve el dispositivo de entrenamiento según la preferencia y disponibilidad."""
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA no está disponible en este entorno. Se usará CPU.")
        return 'cpu'

    return device


class MulticlassDiceLoss(nn.Module):
    """Dice Loss multiclase (promedio sobre clases de tumor, sin fondo)."""

    def __init__(self, smooth=1.0, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits | target: (B, H, W) long
        pred = torch.softmax(pred.float(), dim=1)
        num_classes = pred.shape[1]
        start = 1 if self.ignore_background and num_classes > 1 else 0

        dice_scores = []
        for class_id in range(start, num_classes):
            pred_c = pred[:, class_id]
            target_c = (target == class_id).float()
            intersection = (pred_c * target_c).sum()
            denom = pred_c.sum() + target_c.sum()
            dice_scores.append((2.0 * intersection + self.smooth) / (denom + self.smooth))

        return 1.0 - torch.stack(dice_scores).mean()


class MulticlassCombinedLoss(nn.Module):
    """CrossEntropy + Dice multiclase."""

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MulticlassDiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        target = target.long()
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(
            pred, target
        )


def _dice_binario(pred_mask, target_mask):
    pred_mask = pred_mask.float()
    target_mask = target_mask.float()
    intersection = (pred_mask * target_mask).sum()
    return (2.0 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-8)


def calcular_metricas(pred, target, num_classes=4):
    """Métricas multiclase: Dice medio de tumor, WT y accuracy de píxeles."""
    target = target.long()
    pred_cls = torch.argmax(pred, dim=1)

    correct = (pred_cls == target).sum()
    accuracy = correct.float() / target.numel()

    class_dices = []
    for class_id in range(1, num_classes):
        class_dices.append(
            _dice_binario(pred_cls == class_id, target == class_id).item()
        )

    mean_dice = float(sum(class_dices) / max(len(class_dices), 1))
    dice_wt = _dice_binario(pred_cls > 0, target > 0).item()

    intersection = ((pred_cls > 0) & (target > 0)).sum().float()
    union = ((pred_cls > 0) | (target > 0)).sum().float()
    iou_wt = (intersection / (union + 1e-8)).item()

    metrics = {
        "dice": mean_dice,  # usado para best_model (media NCR/ED/ET)
        "dice_wt": dice_wt,
        "iou": iou_wt,
        "accuracy": accuracy.item(),
    }
    for index, name in enumerate(BRATS_CLASS_NAMES[1:num_classes], start=0):
        metrics[f"dice_{name.lower()}"] = class_dices[index]
    return metrics


def _metricas_vacias(num_classes):
    metrics = {"dice": 0.0, "dice_wt": 0.0, "iou": 0.0, "accuracy": 0.0}
    for name in BRATS_CLASS_NAMES[1:num_classes]:
        metrics[f"dice_{name.lower()}"] = 0.0
    return metrics


def entrenar_epoch(modelo, dataloader, criterion, optimizer, device, num_classes=4):
    """Entrena una época"""
    modelo.train()
    epoch_loss = 0
    epoch_metrics = _metricas_vacias(num_classes)
    valid_batches = 0
    amp_enabled = str(device).startswith('cuda')
    scaler = getattr(entrenar_epoch, '_scaler', None)
    if scaler is None or scaler.is_enabled() != amp_enabled:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        entrenar_epoch._scaler = scaler
    
    pbar = tqdm(dataloader, desc='Training')
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
            outputs = modelo(imgs)
            loss = criterion(outputs, masks)

        if not torch.isfinite(loss):
            print(f"  Advertencia: pérdida no finita en el batch {pbar.n}; se omite.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
        
        with torch.no_grad():
            metrics = calcular_metricas(outputs, masks, num_classes=num_classes)
            epoch_loss += loss.item()
            valid_batches += 1
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
        
        pbar.set_postfix(
            {
                'loss': f'{loss.item():.4f}',
                'dice': f"{metrics['dice']:.4f}",
                'wt': f"{metrics['dice_wt']:.4f}",
            }
        )
    
    if valid_batches == 0:
        raise RuntimeError("No se pudo completar ningún batch de entrenamiento con pérdida finita.")
    epoch_loss /= valid_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= valid_batches
    
    return epoch_loss, epoch_metrics


def validar_epoch(modelo, dataloader, criterion, device, num_classes=4):
    """Valida una época"""
    modelo.eval()
    epoch_loss = 0
    epoch_metrics = _metricas_vacias(num_classes)
    valid_batches = 0
    amp_enabled = str(device).startswith('cuda')
    
    with torch.no_grad():
        for batch_index, (imgs, masks) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
                outputs = modelo(imgs)
                loss = criterion(outputs, masks)

            if not torch.isfinite(loss):
                print(
                    f"  Advertencia: pérdida de validación no finita "
                    f"en el batch {batch_index}; se omite."
                )
                continue

            metrics = calcular_metricas(outputs, masks, num_classes=num_classes)
            epoch_loss += loss.item()
            valid_batches += 1
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
    
    if valid_batches == 0:
        raise RuntimeError("No se pudo completar ningún batch de validación con pérdida finita.")
    epoch_loss /= valid_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= valid_batches
    
    return epoch_loss, epoch_metrics


def entrenar(
    root_dir,
    num_epochs=50,
    batch_size=4,
    learning_rate=0.001,
    checkpoint_every_n_epochs=10,
    save_dir='models',
    device=None,
    num_workers=0,
    prefetch_factor=2,
    persistent_workers=False,
    use_amp=True,
    hdf5_empty_keep_ratio=0.25,
    hdf5_filter_seed=42,
):
    """Función principal de entrenamiento"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = save_dir / 'epoch_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    device = resolver_dispositivo(device)
    amp_enabled = use_amp and str(device).startswith('cuda')
    # Habilitar autotuning de cuDNN para entradas de tamaño fijo (mejora throughput)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO DEL MODELO U-NET")
    print(f"{'='*60}")
    print(f"Dispositivo: {device}")
    print(f"AMP: {'activado' if amp_enabled else 'desactivado'}")
    print(f"Épocas: {num_epochs}")
    print(f"Batch size: {batch_size}")
    lr_text = f"{learning_rate:.10f}".rstrip("0").rstrip(".")
    print(f"Learning rate: {lr_text}")
    print(f"Checkpoint cada: {checkpoint_every_n_epochs} épocas")
    print(f"HDF5 empty keep ratio: {hdf5_empty_keep_ratio}")
    print(f"{'='*60}\n")

    train_loader, val_loader, in_channels, num_classes = crear_dataloaders(
        root_dir,
        batch_size=batch_size,
        train_split=0.8,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        hdf5_empty_keep_ratio=hdf5_empty_keep_ratio,
        hdf5_filter_seed=hdf5_filter_seed,
    )

    modelo, device = crear_modelo(
        device, in_channels=in_channels, out_channels=num_classes
    )

    # Comunicar el estado AMP al helper de entrenamiento por época
    entrenar_epoch._scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    
    criterion = MulticlassCombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    writer = SummaryWriter(log_dir='runs/unet_tumor')
    
    best_dice = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_dice_wt': []}
    
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_metrics = entrenar_epoch(
            modelo, train_loader, criterion, optimizer, device, num_classes=num_classes
        )
        
        val_loss, val_metrics = validar_epoch(
            modelo, val_loader, criterion, device, num_classes=num_classes
        )
        
        scheduler.step(val_loss)
        
        print("\nResultados:")
        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Train Dice(cls): {train_metrics['dice']:.4f} | "
            f"Train Dice(WT): {train_metrics['dice_wt']:.4f}"
        )
        print(
            f"  Val Loss:   {val_loss:.4f} | "
            f"Val Dice(cls):   {val_metrics['dice']:.4f} | "
            f"Val Dice(WT):   {val_metrics['dice_wt']:.4f}"
        )
        print(
            f"  Val IoU(WT): {val_metrics['iou']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
        print(
            "  Val Dice por clase: "
            f"NCR={val_metrics.get('dice_ncr', 0):.4f} | "
            f"ED={val_metrics.get('dice_ed', 0):.4f} | "
            f"ET={val_metrics.get('dice_et', 0):.4f}"
        )
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/dice_mean_tumor', val_metrics['dice'], epoch)
        writer.add_scalar('Metrics/dice_wt', val_metrics['dice_wt'], epoch)
        writer.add_scalar('Metrics/iou_wt', val_metrics['iou'], epoch)
        for name in ("ncr", "ed", "et"):
            key = f"dice_{name}"
            if key in val_metrics:
                writer.add_scalar(f"Metrics/{key}", val_metrics[key], epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_metrics['dice'])
        history['val_dice_wt'].append(val_metrics['dice_wt'])
        
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': modelo.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice,
                    'dice_wt': val_metrics['dice_wt'],
                    'in_channels': in_channels,
                    'out_channels': num_classes,
                    'class_names': BRATS_CLASS_NAMES,
                },
                save_dir / 'best_model.pth',
            )
            print(f"  ✓ Mejor modelo guardado (Dice cls: {best_dice:.4f})")
        
        # Guardar un checkpoint intermedio cada N épocas (solo se conserva el último)
        if checkpoint_every_n_epochs and (epoch + 1) % checkpoint_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': modelo.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'in_channels': in_channels,
                    'out_channels': num_classes,
                    'class_names': BRATS_CLASS_NAMES,
                },
                checkpoint_path,
            )
            for old_checkpoint in checkpoint_dir.glob('checkpoint_epoch_*.pth'):
                if old_checkpoint != checkpoint_path and old_checkpoint.is_file():
                    old_checkpoint.unlink()
            print(f"  ✓ Checkpoint guardado: {checkpoint_path.name} (anteriores eliminados)")
    
    writer.close()
    
    # Graficar con eje de épocas explícito; con 1 época usar marcador para evitar figura "vacía"
    epochs_recorded = len(history['train_loss'])
    if epochs_recorded > 0:
        x_epochs = list(range(1, epochs_recorded + 1))

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(x_epochs, history['train_loss'], label='Train Loss', marker='o')
        plt.plot(x_epochs, history['val_loss'], label='Val Loss', marker='o')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.xticks(x_epochs)
        plt.legend()
        plt.title('Pérdida durante el entrenamiento')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            x_epochs,
            history['val_dice'],
            label='Val Dice (media NCR/ED/ET)',
            color='green',
            marker='o',
        )
        if history.get('val_dice_wt'):
            plt.plot(
                x_epochs,
                history['val_dice_wt'],
                label='Val Dice (WT)',
                color='orange',
                marker='o',
            )
        plt.xlabel('Época')
        plt.ylabel('Dice Score')
        plt.xticks(x_epochs)
        plt.legend()
        plt.title('Dice Score en validación')
        plt.grid(True)

        output_path = save_dir / 'training_history.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Gráficas de rendimiento guardadas en {output_path.resolve()}")
    else:
        print("\n! No se generó training_history.png porque no hubo épocas completadas.")
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Mejor Dice Score: {best_dice:.4f}")
    print(f"Modelo guardado en: {save_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")
    
    return modelo, history


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ROOT_DIR = os.path.join(base_dir, "brats_data", "Base de datos Brats")

    entrenar(
        root_dir=ROOT_DIR,
        num_epochs=30,
        batch_size=4,
        learning_rate=0.001,
        save_dir='models',
    )
