
from pathlib import Path
from collections import Counter
from PIL import Image
import argparse

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}


def find_images(folder):
    return [
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def inspect_dataset(root):
    root = Path(root)

    train_good = root / "train" / "good"
    test_dir = root / "test"
    ground_truth = root / "ground_truth"

    # 1. Check the required directories.
    required = [train_good, test_dir, ground_truth]

    for folder in required:
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing directory: {folder}"
            )

    # 2. Inspect training images.
    train_images = find_images(train_good)

    print("\n===== TRAINING DATA =====")
    print("Normal training images:", len(train_images))

    # 3. Inspect test categories.
    print("\n===== TEST DATA =====")

    defect_images = []

    for category in sorted(test_dir.iterdir()):
        if not category.is_dir():
            continue

        images = find_images(category)
        print(f"{category.name}: {len(images)} images")

        if category.name.lower() != "good":
            defect_images.extend(
                (category.name, image) for image in images
            )

    # 4. Inspect ground-truth masks.
    print("\n===== GROUND TRUTH =====")

    mask_count = 0
    missing_masks = []

    for category, image_path in defect_images:

        mask_folder = (
            ground_truth / category / image_path.stem
        )

        masks = (
            list(mask_folder.glob("*.png"))
            if mask_folder.exists()
            else []
        )

        # Alternative flat mask layout.
        if not masks:
            alternative = (
                ground_truth / category /
                f"{image_path.stem}.png"
            )

            if alternative.exists():
                masks = [alternative]

        if masks:
            mask_count += 1
        else:
            missing_masks.append(image_path)

    print("Defective images:", len(defect_images))
    print("Images with masks:", mask_count)
    print("Images without masks:", len(missing_masks))

    # 5. Inspect image dimensions and color modes.
    print("\n===== IMAGE PROPERTIES =====")

    all_images = train_images + find_images(test_dir)

    dimensions = Counter()
    color_modes = Counter()

    for image_path in all_images:
        try:
            with Image.open(image_path) as image:
                dimensions[image.size] += 1
                color_modes[image.mode] += 1
        except Exception as error:
            print("Could not read:", image_path, error)

    print("Image dimensions:", dimensions)
    print("Color modes:", color_modes)

    # 6. Display a sample image's properties.
    if train_images:
        with Image.open(train_images[0]) as image:
            print("\n===== SAMPLE IMAGE =====")
            print("Path:", train_images[0])
            print("Size:", image.size)
            print("Mode:", image.mode)
            print("Format:", image.format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to engine_wiring directory"
    )

    args = parser.parse_args()

    inspect_dataset(args.data)
