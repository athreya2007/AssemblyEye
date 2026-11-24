
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}


class AutoVIGoodDataset(Dataset):

    """
    Loads only normal automotive assembly images
    from AutoVI for autoencoder training.
    """

    def __init__(self, root, image_size=128):

        self.image_dir = Path(root) / "train" / "good"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Training directory not found: {self.image_dir}"
            )

        # Discover all supported image files
        self.image_paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No training images found in {self.image_dir}"
            )

        # Image preprocessing
        self.transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size),
                antialias=True
            ),

            transforms.ToTensor()

        ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image_tensor = self.transform(image)

        return image_tensor
