# ============================================================
# MODELO U-NET PARA SEGMENTACIÓN DE TUMORES
# ============================================================

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Bloque de doble convolución (Conv -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Arquitectura U-Net para segmentación de tumores cerebrales
    
    Args:
        in_channels: Número de canales de entrada (1 para imágenes en escala de grises)
        out_channels: Número de clases de salida (1 para segmentación binaria)
        features: Lista con el número de features en cada nivel [64, 128, 256, 512]
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (parte descendente)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck (fondo de la U)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder (parte ascendente)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Capa final
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Invertir

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsampling
            skip_connection = skip_connections[idx//2]

            # Ajustar tamaño si es necesario
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            # Concatenar con skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # Convolución

        # Return raw logits (no sigmoid). Use BCEWithLogitsLoss for stability with AMP.
        return self.final_conv(x)


def crear_modelo(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Crea y retorna el modelo U-Net
    
    Returns:
        modelo: Modelo U-Net listo para entrenar
        device: Dispositivo (cuda o cpu)
    """
    modelo = UNet(in_channels=1, out_channels=1)
    modelo = modelo.to(device)
    
    print(f"\n{'='*60}")
    print(f"Modelo U-Net creado")
    print(f"Dispositivo: {device}")
    print(f"Parámetros totales: {sum(p.numel() for p in modelo.parameters()):,}")
    print(f"{'='*60}\n")
    
    return modelo, device


def test_modelo():
    """Prueba rápida del modelo"""
    modelo, device = crear_modelo()
    x = torch.randn(1, 1, 256, 256).to(device)
    prediccion = modelo(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {prediccion.shape}")
    print(f"Output range: [{prediccion.min():.3f}, {prediccion.max():.3f}]")


if __name__ == "__main__":
    test_modelo()
