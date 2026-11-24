
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AutoVIGoodDataset(Dataset):

    def __init__(self, root, image_size=128):

        # 1. Locate the normal training images
        self.image_dir = Path(root) / "train" / "good"

        # 2. Find all PNG images
        self.image_paths = sorted(
            self.image_dir.glob("*.png")
        )

        # 3. Define preprocessing operations
        self.transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor()

        ])

    # 4. Return total number of images
    def __len__(self):

        return len(self.image_paths)

    # 5. Load and preprocess one image
    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)

        return image_tensor


# ----------------------------------
# TEST OUR DATASET
# ----------------------------------

dataset = AutoVIGoodDataset(
    root="Data",
    image_size=128
)

print("Total images:", len(dataset))

sample = dataset[0]

print("Tensor shape:", sample.shape)
print("Tensor dtype:", sample.dtype)
print("Minimum value:", sample.min().item())
print("Maximum value:", sample.max().item())
