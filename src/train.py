
import argparse
import random
import csv

from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from .models.tiny_assembly_ae import TinyAssemblyAE
from .dataset import AutoVIGoodDataset

# ==========================================
# SSIM
# ==========================================

def calculate_ssim(x, y, window_size=7):

    mu_x = F.avg_pool2d(x, window_size, stride=1)
    mu_y = F.avg_pool2d(y, window_size, stride=1)

    sigma_x = (
        F.avg_pool2d(x * x, window_size, stride=1)
        - mu_x.square()
    ).clamp_min(0)

    sigma_y = (
        F.avg_pool2d(y * y, window_size, stride=1)
        - mu_y.square()
    ).clamp_min(0)

    sigma_xy = (
        F.avg_pool2d(x * y, window_size, stride=1)
        - mu_x * mu_y
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (2 * mu_x * mu_y + c1)
        * (2 * sigma_xy + c2)
    )

    denominator = (
        (mu_x.square() + mu_y.square() + c1)
        * (sigma_x + sigma_y + c2)
    )

    ssim = numerator / (denominator + 1e-8)

    return ssim.mean().clamp(-1, 1)


# ==========================================
# RECONSTRUCTION LOSS
# ==========================================

def reconstruction_loss(original, reconstructed):

    l1 = F.l1_loss(reconstructed, original)

    ssim = calculate_ssim(original, reconstructed)

    loss = 0.5 * l1 + 0.5 * (1.0 - ssim)

    return loss


# ==========================================
# TRAINING
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="Data")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--out",
        default="checkpoints/tiny_assembly_ae_128.pt"
    )

    args = parser.parse_args()

    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------
    # LOAD DATASET
    # --------------------------------------

    dataset = AutoVIGoodDataset(
        root=args.data,
        image_size=args.image_size
    )

    val_size = max(1, round(len(dataset) * 0.1))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Training batches:", len(train_loader))

    # --------------------------------------
    # INITIALIZE MODEL
    # --------------------------------------

    model = TinyAssemblyAE().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # --------------------------------------
    # CHECKPOINT SETUP
    # --------------------------------------

    checkpoint_path = Path(args.out)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")
    history = []

    # ======================================
    # EPOCH LOOP
    # ======================================

    for epoch in range(args.epochs):

        # -------- TRAINING --------

        model.train()

        train_total = 0.0

        for images in train_loader:

            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            reconstructed = model(images)

            loss = reconstruction_loss(
                images,
                reconstructed
            )

            loss.backward()

            optimizer.step()

            train_total += loss.item() * images.size(0)

        train_loss = train_total / len(train_dataset)

        # -------- VALIDATION --------

        model.eval()

        val_total = 0.0

        with torch.no_grad():

            for images in val_loader:

                images = images.to(device)

                reconstructed = model(images)

                loss = reconstruction_loss(
                    images,
                    reconstructed
                )

                val_total += loss.item() * images.size(0)

        val_loss = val_total / len(val_dataset)

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"| Train: {train_loss:.5f} "
            f"| Val: {val_loss:.5f}"
        )

        # -------- SAVE BEST MODEL --------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({

                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),

                "epoch": epoch + 1,
                "val_loss": val_loss,
                "image_size": args.image_size

            }, checkpoint_path)

            print("  Best checkpoint saved!")

    # --------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------

    with open(
        "training_history.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "val_loss"]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\nTRAINING COMPLETED!")
    print("Best validation loss:", best_val_loss)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
