
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

from src.dataset import AutoVIGoodDataset, AutoVITestDataset
from src.models.tiny_assembly_ae import TinyAssemblyAE


# ==========================================
# 1. ANOMALY SCORING
# ==========================================

def calculate_anomaly_score(heatmap):

    # heatmap: B x 1 x H x W

    flattened = heatmap.flatten(1)

    # Number of highest-error pixels to consider
    k = max(
        1,
        int(np.ceil(flattened.shape[1] * 0.01))
    )

    top_pixels = torch.topk(
        flattened,
        k=k,
        dim=1
    ).values

    # Average top 1% reconstruction error
    return top_pixels.mean(dim=1)


# ==========================================
# 2. GENERATE RECONSTRUCTION HEATMAP
# ==========================================

def generate_heatmap(model, images):

    reconstructed = model(images)

    heatmap = (
        images - reconstructed
    ).abs().mean(dim=1, keepdim=True)

    return reconstructed, heatmap


# ==========================================
# 3. CALIBRATE PASS / REJECT THRESHOLD
# ==========================================

def calibrate_threshold(model, loader, device, quantile):

    model.eval()

    validation_scores = []

    with torch.inference_mode():

        for images in loader:

            images = images.to(device)

            _, heatmap = generate_heatmap(
                model,
                images
            )

            scores = calculate_anomaly_score(heatmap)

            validation_scores.extend(
                scores.cpu().numpy().tolist()
            )

    if not validation_scores:
        raise RuntimeError("No validation scores generated.")

    threshold = float(
        np.quantile(validation_scores, quantile)
    )

    return threshold, validation_scores


# ==========================================
# 4. INFERENCE LATENCY BENCHMARK
# ==========================================

def benchmark_latency(
    model,
    sample_image,
    device,
    warmup=5,
    repeats=30
):

    model.eval()

    images = sample_image.unsqueeze(0).to(device)

    with torch.inference_mode():

        # Warmup
        for _ in range(warmup):

            _, heatmap = generate_heatmap(
                model,
                images
            )

            calculate_anomaly_score(heatmap)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repeats):

            _, heatmap = generate_heatmap(
                model,
                images
            )

            calculate_anomaly_score(heatmap)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / repeats) * 1000

    return latency_ms


# ==========================================
# 5. VISUALIZATION
# ==========================================

def save_visualizations(panels, output_dir):

    if not panels:
        print("No visualization panels available.")
        return

    rows = len(panels)

    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(14, 3.5 * rows),
        squeeze=False
    )

    for i, panel in enumerate(panels):

        image, reconstructed, heatmap, mask, label, category = panel

        image = image.permute(1, 2, 0).numpy()

        reconstructed = (
            reconstructed.permute(1, 2, 0).numpy()
        )

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Original Image")

        axes[i, 1].imshow(reconstructed)
        axes[i, 1].set_title("Reconstruction")

        axes[i, 2].imshow(
            heatmap,
            cmap="inferno"
        )
        axes[i, 2].set_title("Predicted Heatmap")

        axes[i, 3].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[i, 3].set_title("Ground Truth Mask")

        axes[i, 0].set_ylabel(
            f"{category} | {'DEFECT' if label else 'GOOD'}"
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        output_dir / "inspection_panels.png",
        dpi=150
    )

    plt.close(fig)


# ==========================================
# 6. MAIN EVALUATION
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="Data")

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tiny_assembly_ae_128.pt"
    )

    parser.add_argument(
        "--out",
        default="evaluation/128"
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95
    )

    parser.add_argument("--max-panels", type=int, default=12)

    parser.add_argument(
        "--pixels-per-class-per-image",
        type=int,
        default=512
    )

    args = parser.parse_args()

    if not 0 < args.threshold_quantile < 1:
        raise ValueError("Threshold quantile must be between 0 and 1.")

    if args.pixels_per_class_per_image < 1:
        raise ValueError("Pixel sampling limit must be positive.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # ======================================
    # LOAD CHECKPOINT
    # ======================================

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    image_size = int(checkpoint["image_size"])

    model = TinyAssemblyAE().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\n===== MODEL INFORMATION =====")

    print("Checkpoint:", checkpoint_path)
    print("Training epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["val_loss"])
    print("Image resolution:", image_size)

    model_size_mb = (
        checkpoint_path.stat().st_size / (1024 * 1024)
    )

    print(f"Checkpoint file size: {model_size_mb:.4f} MB")

    # ======================================
    # RECREATE VALIDATION SPLIT
    # ======================================

    print("\nLoading normal validation images...")

    normal_dataset = AutoVIGoodDataset(
        root=args.data,
        image_size=image_size
    )

    val_size = max(
        1,
        round(len(normal_dataset) * 0.1)
    )

    train_size = len(normal_dataset) - val_size

    _, val_dataset = random_split(
        normal_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    # ======================================
    # CALIBRATE THRESHOLD
    # ======================================

    threshold, calibration_scores = calibrate_threshold(
        model,
        val_loader,
        device,
        args.threshold_quantile
    )

    print("\n===== THRESHOLD CALIBRATION =====")

    print("Validation images:", len(calibration_scores))
    print("Quantile:", args.threshold_quantile)
    print(f"PASS / REJECT threshold: {threshold:.6f}")

    print(
        "Note: Threshold is preliminary; "
        "calibration uses a small validation set."
    )

    # ======================================
    # LOAD TEST DATASET
    # ======================================

    test_dataset = AutoVITestDataset(
        root=args.data,
        image_size=image_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    print("\n===== TEST DATA =====")

    print("Total test images:", len(test_dataset))

    # ======================================
    # TEST INFERENCE
    # ======================================

    image_scores = []
    image_labels = []
    image_categories = []
    image_paths = []

    pixel_normal_samples = []
    pixel_defect_samples = []

    panels = []
    seen_categories = set()

    rng = np.random.default_rng(42)

    model.eval()

    with torch.inference_mode():

        for images, masks, labels, paths, categories in test_loader:

            images_device = images.to(device)

            reconstructed, heatmap = generate_heatmap(
                model,
                images_device
            )

            scores = calculate_anomaly_score(heatmap)

            image_scores.extend(
                scores.cpu().numpy().tolist()
            )

            image_labels.extend(
                labels.numpy().tolist()
            )

            image_categories.extend(list(categories))
            image_paths.extend(list(paths))

            # Collect bounded pixel samples per image.
            # This avoids holding every full-resolution pixel
            # from the test set in memory.

            heatmaps_np = heatmap[:, 0].cpu().numpy()
            masks_np = masks[:, 0].numpy()

            for i in range(len(images)):

                pixel_scores = heatmaps_np[i].reshape(-1)

                pixel_labels = (
                    masks_np[i].reshape(-1) > 0.5
                ).astype(np.uint8)

                normal_indices = np.flatnonzero(
                    pixel_labels == 0
                )

                defect_indices = np.flatnonzero(
                    pixel_labels == 1
                )

                limit = args.pixels_per_class_per_image

                if len(normal_indices) > limit:

                    normal_indices = rng.choice(
                        normal_indices,
                        size=limit,
                        replace=False
                    )

                if len(defect_indices) > limit:

                    defect_indices = rng.choice(
                        defect_indices,
                        size=limit,
                        replace=False
                    )

                pixel_normal_samples.append(
                    pixel_scores[normal_indices]
                )

                pixel_defect_samples.append(
                    pixel_scores[defect_indices]
                )

                # One visual example per category
                category = categories[i]

                panel_key = (
                    "good" if labels[i].item() == 0
                    else category
                )

                if (
                    panel_key not in seen_categories
                    and len(panels) < args.max_panels
                ):

                    panels.append((

                        images[i].cpu(),

                        reconstructed[i].cpu(),

                        heatmaps_np[i],

                        masks_np[i],

                        int(labels[i].item()),

                        category

                    ))

                    seen_categories.add(panel_key)

    # ======================================
    # IMAGE-LEVEL METRICS
    # ======================================

    scores_np = np.asarray(
        image_scores,
        dtype=np.float64
    )

    labels_np = np.asarray(
        image_labels,
        dtype=np.uint8
    )

    predictions = (
        scores_np >= threshold
    ).astype(np.uint8)

    image_auc = (
        roc_auc_score(labels_np, scores_np)
        if len(np.unique(labels_np)) == 2
        else float("nan")
    )

    accuracy = accuracy_score(
        labels_np,
        predictions
    )

    precision = precision_score(
        labels_np,
        predictions,
        zero_division=0
    )

    defect_recall = recall_score(
        labels_np,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels_np,
        predictions,
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        labels_np,
        predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        labels_np,
        predictions,
        labels=[0, 1]
    ).ravel()

    false_reject_rate = fp / max(1, fp + tn)

    # ======================================
    # PIXEL-LEVEL AUROC
    # ======================================

    normal_pixels = (
        np.concatenate(pixel_normal_samples)
        if pixel_normal_samples
        else np.array([])
    )

    defect_pixels = (
        np.concatenate(pixel_defect_samples)
        if pixel_defect_samples
        else np.array([])
    )

    pixel_auc = float("nan")

    if len(normal_pixels) and len(defect_pixels):

        sampled_scores = np.concatenate([
            normal_pixels,
            defect_pixels
        ])

        sampled_labels = np.concatenate([

            np.zeros(
                len(normal_pixels),
                dtype=np.uint8
            ),

            np.ones(
                len(defect_pixels),
                dtype=np.uint8
            )

        ])

        pixel_auc = roc_auc_score(
            sampled_labels,
            sampled_scores
        )

    # ======================================
    # PER-CATEGORY RECALL
    # ======================================

    categories_array = np.asarray(image_categories)

    category_results = {}

    for category in sorted(set(image_categories)):

        if category.lower() == "good":
            continue

        category_mask = (
            categories_array == category
        )

        category_recall = float(
            np.mean(
                predictions[category_mask] == 1
            )
        )

        category_results[category] = category_recall

    # ======================================
    # INFERENCE LATENCY
    # ======================================

    print("\nBenchmarking inference latency...")

    sample_image = test_dataset[0][0]

    latency_ms = benchmark_latency(
        model,
        sample_image,
        device
    )

    throughput = 1000 / latency_ms

    # ======================================
    # PRINT RESULTS
    # ======================================

    print("\n" + "=" * 45)
    print("       ASSEMBLYEYE EVALUATION REPORT")
    print("=" * 45)

    print("\n--- IMAGE-LEVEL METRICS ---")

    print(f"Image AUROC:          {image_auc:.4f}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Defect Recall:        {defect_recall:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"Balanced Accuracy:    {balanced_acc:.4f}")
    print(f"False Reject Rate:    {false_reject_rate:.4f}")

    print("\n--- CONFUSION MATRIX ---")

    print(f"True Negatives:       {tn}")
    print(f"False Positives:      {fp}")
    print(f"False Negatives:      {fn}")
    print(f"True Positives:       {tp}")

    print("\n--- PIXEL-LEVEL METRICS ---")

    print(f"Sampled Pixel AUROC:  {pixel_auc:.4f}")
    print(f"Normal pixels sampled: {len(normal_pixels)}")
    print(f"Defect pixels sampled: {len(defect_pixels)}")

    print("\n--- DEFECT CATEGORY RECALL ---")

    for category, value in category_results.items():

        print(f"{category:20s}: {value:.4f}")

    print("\n--- EDGE PERFORMANCE ---")

    print(f"Device:               {device}")
    print(f"Checkpoint size:      {model_size_mb:.4f} MB")
    print(f"Single-image latency: {latency_ms:.2f} ms")
    print(f"Inference throughput: {throughput:.2f} images/sec")

    print("\n--- DECISION THRESHOLD ---")

    print(f"Threshold:            {threshold:.6f}")

    # ======================================
    # SAVE REPORTS
    # ======================================

    output_dir = Path(args.out)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save visualizations
    save_visualizations(
        panels,
        output_dir
    )

    # Save per-image scores
    scores_file = output_dir / "image_scores.csv"

    with open(
        scores_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "image_path",
            "category",
            "actual_label",
            "anomaly_score",
            "predicted_label",
            "threshold"

        ])

        for i in range(len(scores_np)):

            writer.writerow([

                image_paths[i],
                image_categories[i],
                int(labels_np[i]),
                float(scores_np[i]),
                int(predictions[i]),
                threshold

            ])

    # Save summary metrics
    report_file = output_dir / "metrics.txt"

    with open(report_file, "w") as file:

        file.write("ASSEMBLYEYE EVALUATION REPORT\n\n")

        file.write(f"Image AUROC: {image_auc:.4f}\n")
        file.write(f"Pixel AUROC: {pixel_auc:.4f}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Defect Recall: {defect_recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        file.write(f"False Reject Rate: {false_reject_rate:.4f}\n")

        file.write(f"\nThreshold: {threshold:.6f}\n")
        file.write(f"Threshold quantile: {args.threshold_quantile}\n")

        file.write(f"\nCheckpoint size MB: {model_size_mb:.4f}\n")
        file.write(f"Latency ms: {latency_ms:.2f}\n")
        file.write(f"Throughput images/sec: {throughput:.2f}\n")

        file.write("\nCONFUSION MATRIX\n")

        file.write(f"TN: {tn}\n")
        file.write(f"FP: {fp}\n")
        file.write(f"FN: {fn}\n")
        file.write(f"TP: {tp}\n")

        file.write("\nCATEGORY RECALL\n")

        for category, value in category_results.items():

            file.write(
                f"{category}: {value:.4f}\n"
            )

    print("\n" + "=" * 45)
    print("REPORTS SAVED SUCCESSFULLY")
    print("=" * 45)

    print("Visualization:", output_dir / "inspection_panels.png")
    print("Image scores:", scores_file)
    print("Metrics:", report_file)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()
