import matplotlib.pyplot as plt
import numpy as np

from ml.dataset_loader import BraTSDataset

root_dir = r"brats_data\Base de datos Brats"

dataset = BraTSDataset(
    root_dir=root_dir,
    modality="T1wCE",
    augment=False,
    verbose=True,
)

indices = np.linspace(0, len(dataset) - 1, min(6, len(dataset)), dtype=int)

fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))

if len(indices) == 1:
    axes = np.expand_dims(axes, axis=0)

for row, index in enumerate(indices):
    image, mask = dataset[index]

    image = image.squeeze().numpy()
    mask = mask.squeeze().numpy()

    axes[row, 0].imshow(image, cmap="gray")
    axes[row, 0].set_title(f"Imagen {index}")

    axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title("Máscara")

    axes[row, 2].imshow(image, cmap="gray")
    axes[row, 2].imshow(mask, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[row, 2].set_title("Superposición")

    for axis in axes[row]:
        axis.axis("off")

plt.tight_layout()
plt.savefig("comprobacion_mascaras.png", dpi=150)
plt.show()