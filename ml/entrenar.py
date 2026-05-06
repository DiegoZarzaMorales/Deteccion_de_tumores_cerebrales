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
from .dataset_loader import crear_dataloaders


def resolver_dispositivo(device=None):
    """Resuelve el dispositivo de entrenamiento según la preferencia y disponibilidad."""
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA no está disponible en este entorno. Se usará CPU.")
        return 'cpu'

    return device


class DiceLoss(nn.Module):
    """Dice Loss para segmentación"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combinación de BCE y Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calcular_metricas(pred, target, threshold=0.5):
    """Calcula métricas de evaluación"""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
    
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    correct = (pred_binary == target_binary).sum()
    total = target_binary.numel()
    accuracy = correct / total
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'accuracy': accuracy.item()
    }


def entrenar_epoch(modelo, dataloader, criterion, optimizer, device):
    """Entrena una época"""
    modelo.train()
    epoch_loss = 0
    epoch_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0}
    amp_enabled = str(device).startswith('cuda')
    scaler = getattr(entrenar_epoch, '_scaler', None)
    if scaler is None or scaler.is_enabled() != amp_enabled:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        entrenar_epoch._scaler = scaler
    
    pbar = tqdm(dataloader, desc='Training')
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = modelo(imgs)
            loss = criterion(outputs, masks)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            metrics = calcular_metricas(outputs, masks)
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f"{metrics['dice']:.4f}"})
    
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics


def validar_epoch(modelo, dataloader, criterion, device):
    """Valida una época"""
    modelo.eval()
    epoch_loss = 0
    epoch_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0}
    amp_enabled = str(device).startswith('cuda')
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = modelo(imgs)
                loss = criterion(outputs, masks)
            
            metrics = calcular_metricas(outputs, masks)
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
    
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
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
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint cada: {checkpoint_every_n_epochs} épocas")
    print(f"{'='*60}\n")
    
    modelo, device = crear_modelo(device)
    
    train_loader, val_loader = crear_dataloaders(
        root_dir,
        batch_size=batch_size,
        train_split=0.8,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    # Comunicar el estado AMP al helper de entrenamiento por época
    entrenar_epoch._scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    writer = SummaryWriter(log_dir='runs/unet_tumor')
    
    best_dice = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_metrics = entrenar_epoch(
            modelo, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_metrics = validar_epoch(
            modelo, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        print("\nResultados:")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_metrics['dice']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_metrics['dice']:.4f}")
        print(f"  Val IoU:    {val_metrics['iou']:.4f} | Val Acc:    {val_metrics['accuracy']:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/dice', val_metrics['dice'], epoch)
        writer.add_scalar('Metrics/iou', val_metrics['iou'], epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_metrics['dice'])
        
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': modelo.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice,
                },
                save_dir / 'best_model.pth',
            )
            print(f"  ✓ Mejor modelo guardado (Dice: {best_dice:.4f})")
        
        # Guardar un checkpoint intermedio cada N épocas
        if checkpoint_every_n_epochs and (epoch + 1) % checkpoint_every_n_epochs == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': modelo.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth',
            )
    
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
        plt.plot(x_epochs, history['val_dice'], label='Val Dice Score', color='green', marker='o')
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
