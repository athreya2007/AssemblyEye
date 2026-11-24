
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}


# ==========================================
# TRAINING DATASET — GOOD IMAGES ONLY
# ==========================================

class AutoVIGoodDataset(Dataset):

    def __init__(self, root, image_size=128):

        self.image_dir = Path(root) / "train" / "good"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Training directory not found: {self.image_dir}"
            )

        self.image_paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError("No training images found.")

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

        return self.transform(image)


# ==========================================
# TEST DATASET — GOOD + DEFECTIVE IMAGES
# ==========================================

class AutoVITestDataset(Dataset):

    def __init__(self, root, image_size=128):

        self.root = Path(root)
        self.test_dir = self.root / "test"
        self.gt_dir = self.root / "ground_truth"

        if not self.test_dir.is_dir():
            raise FileNotFoundError(
                f"Test directory not found: {self.test_dir}"
            )

        self.samples = []

        for category_dir in sorted(self.test_dir.iterdir()):

            if not category_dir.is_dir():
                continue

            category = category_dir.name
            is_good = category.lower() == "good"

            image_paths = sorted(
                path
                for path in category_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:

                mask_path = None

                if not is_good:

                    # Official nested mask structure
                    mask_folder = (
                        self.gt_dir /
                        category /
                        image_path.stem
                    )

                    if mask_folder.is_dir():

                        candidates = sorted(
                            path
                            for path in mask_folder.iterdir()
                            if path.is_file()
                            and path.suffix.lower() in IMAGE_EXTENSIONS
                        )

                        if candidates:
                            mask_path = candidates[0]

                    # Alternative flat mask structure
                    if mask_path is None:

                        for extension in IMAGE_EXTENSIONS:

                            alternative = (
                                self.gt_dir /
                                category /
                                f"{image_path.stem}{extension}"
                            )

                            if alternative.is_file():
                                mask_path = alternative
                                break

                    if mask_path is None:
                        raise FileNotFoundError(
                            f"Mask not found for: {image_path}"
                        )

                self.samples.append((
                    image_path,
                    mask_path,
                    int(not is_good),
                    category
                ))

        if not self.samples:
            raise RuntimeError("No test images found.")

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                antialias=True
            ),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path, label, category = self.samples[index]

        with Image.open(image_path) as img:
            image = self.image_transform(
                img.convert("RGB")
            )

        if label == 0:

            mask = image.new_zeros(
                (1, image.shape[1], image.shape[2])
            )

        else:

            with Image.open(mask_path) as mask_img:
                mask = self.mask_transform(
                    mask_img.convert("L")
                )

            mask = (mask > 0.5).float()

        return image, mask, label, str(image_path), category
