
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("Data")

# Select one defective category
category = "fastening"

# Find an image
image_folder = ROOT / "test" / category
image_path = sorted(image_folder.rglob("*.png"))[0]

# Find its corresponding ground-truth mask
mask_folder = ROOT / "ground_truth" / category / image_path.stem

mask_candidates = (
    sorted(mask_folder.glob("*.png"))
    if mask_folder.exists()
    else []
)

if not mask_candidates:
    alternative = (
        ROOT / "ground_truth" / category /
        f"{image_path.stem}.png"
    )
    if alternative.exists():
        mask_candidates = [alternative]

if not mask_candidates:
    raise FileNotFoundError(
        f"No mask found for {image_path}"
    )

mask_path = mask_candidates[0]

# Load image and mask
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

image_array = np.array(image)
mask_array = np.array(mask)

# Inspect mask information
print("Image:", image_path)
print("Mask:", mask_path)
print("Image shape:", image_array.shape)
print("Mask shape:", mask_array.shape)
print("Unique mask values:", np.unique(mask_array))
print("Defective pixel count:", np.count_nonzero(mask_array))

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_array)
axes[0].set_title("Original Image")

axes[1].imshow(mask_array, cmap="gray")
axes[1].set_title("Ground Truth Mask")

axes[2].imshow(image_array)
axes[2].imshow(mask_array > 0, cmap="Reds", alpha=0.45)
axes[2].set_title("Defect Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
