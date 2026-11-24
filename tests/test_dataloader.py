
import torch

from torch.utils.data import DataLoader

from tests.dataset import AutoVIGoodDataset


# 1. Load our Dataset
dataset = AutoVIGoodDataset(
    root="Data",
    image_size=128
)


# 2. Create DataLoader
loader = DataLoader(

    dataset,

    batch_size=8,

    shuffle=True,

    num_workers=0

)


# 3. Inspect batches
print("Number of images:", len(dataset))
print("Number of batches:", len(loader))


for batch_index, images in enumerate(loader):

    print(
        f"Batch {batch_index + 1}:",
        images.shape
    )

    print("Data type:", images.dtype)

    print("Min:", images.min().item())
    print("Max:", images.max().item())

    # Inspect just the first batch
    if batch_index == 0:
        break
