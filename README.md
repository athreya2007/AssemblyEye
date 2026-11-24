# AssemblyEye 🔍
### Lightweight Computer Vision for Automotive Assembly Anomaly Detection

AssemblyEye is an experimental Computer Vision project for **unsupervised visual anomaly detection in automotive manufacturing**.

The project investigates whether a compact, scratch-built convolutional autoencoder can learn the appearance of normal automotive assemblies and flag unusual regions through image reconstruction errors.

The initial experiment uses Renault AutoVI v1.0.0 — `engine_wiring`.

> **Current status:** The baseline model has been trained and evaluated. Initial results show useful pixel-level anomaly ranking, but poor image-level defect recall. The current model is an experimental baseline, not a production-ready inspection system.

---

## 1. Problem Statement

Automotive assembly lines require inspection for issues such as:

- Missing or incorrectly fastened clips.
- Misrouted wiring.
- Incorrectly positioned components.
- Other visual abnormalities in an assembly.

Manual inspection can be difficult to scale consistently. AssemblyEye explores an automated visual inspection pipeline that learns from defect-free examples rather than requiring defective examples during model training.

## 2. Proposed Approach

AssemblyEye uses a **one-class reconstruction-based anomaly detection** approach:

1. Train an autoencoder using only normal assembly images.
2. Reconstruct an incoming assembly image.
3. Calculate the pixel-wise difference between the input and reconstruction.
4. Aggregate the largest residual errors into an image-level anomaly score.
5. Compare the score with a threshold to produce a PASS / REJECT decision.
6. Compare the predicted residual heatmap with dataset ground-truth masks during evaluation.

The model is not trained as a supervised defect classifier. Ground-truth masks are reserved for evaluation.

### Anomaly map and image score

The residual heatmap is calculated as:

\[
E(i,j) = \frac{1}{3}\sum_{c=1}^{3}|X(c,i,j)-\hat X(c,i,j)|
\]

where \(X\) is the input image and \(\hat X\) is its reconstruction.

The image-level anomaly score is the mean of the top 1% residual pixels. A validation-derived threshold converts that score into a PASS / REJECT prediction.

## 3. Model Architecture — TinyAssemblyAE

TinyAssemblyAE is a compact Depthwise-Separable Convolutional Autoencoder implemented in PyTorch.

| Component | Design |
|---|---|
| Input | 3-channel RGB image |
| Baseline input resolution | 128 × 128 |
| Encoder | Conv2D followed by 3 depthwise-separable convolution blocks |
| Encoder channels | 3 → 16 → 32 → 64 → 128 |
| Bottleneck | 1 × 1 convolutions: 128 → 32 → 128 |
| Decoder | Bilinear upsampling and depthwise-separable convolution blocks |
| Output | Reconstructed RGB image |
| Activations | SiLU in feature blocks; Sigmoid at output |
| Skip connections | None |
| Trainable parameters | 35,283 |
| FP32 parameter storage | Approximately 137.82 KiB |

The architecture's forward pass has been tested with 128 × 128, 256 × 256 and 400 × 400 inputs. **The reported training and evaluation results in this README are for 128 × 128 only.** Testing the forward pass at another resolution does not establish its anomaly-detection performance at that resolution.

## 4. Dataset

**Dataset:** Renault AutoVI v1.0.0  
**Selected category:** `engine_wiring`

Dataset counts observed during local inspection:

| Split | Category | Images |
|---|---|---:|
| Training | good | 285 |
| Testing | good | 285 |
| Testing | blue_hoop | 5 |
| Testing | cardboard | 5 |
| Testing | fastening | 277 |
| Testing | multiple | 33 |
| Testing | obstruction | 2 |
| **Total** | | **892** |

All 322 defective test images had corresponding ground-truth masks in the inspected dataset.

Image properties observed:

- Resolution: 400 × 400 pixels.
- Color mode: RGB.
- Format: PNG.
- Training samples: normal (`train/good`) only.
- Evaluation samples: normal and defective test images, with masks for defective images.

The test set is imbalanced across defect categories: fastening accounts for 277 of the 322 defective samples. Per-category results are therefore important; overall metrics can hide weak performance on less represented categories.

**Dataset usage:** Refer to the dataset's included `license.txt` and original dataset documentation for attribution and permitted use.

## 5. Technology Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- scikit-learn

ONNX, INT8 quantization and OpenCV conveyor simulation are planned engineering steps, not part of the current benchmark.

## 6. Repository Structure

```text
AssemblyEye/
│
├── Data/
│   ├── train/
│   ├── test/
│   └── ground_truth/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       ├── __init__.py
│       └── tiny_assembly_ae.py
│
├── tests/
│   ├── test_dataloader.py
│   └── test_model.py
│
├── checkpoints/
│   └── tiny_assembly_ae_128.pt
│
├── evaluation/
│   └── 128/
│       ├── inspection_panels.png
│       ├── image_scores.csv
│       └── metrics.txt
│
├── visualization/
├── scripts/
├── training_history.csv
├── requirements.txt
└── README.md
```

The dataset and checkpoint are local artifacts and should not be committed to GitHub unless their redistribution is explicitly permitted.

## 7. Setup and Execution

Run commands from the repository root (`AssemblyEye/`).

### 7.1 Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 7.2 Install dependencies

```powershell
pip install -r requirements.txt
```

The project requires PyTorch, Torchvision, NumPy, Pillow, Matplotlib and scikit-learn.

### 7.3 Prepare the dataset

Place the extracted `engine_wiring` contents under `Data/`, so that the repository contains:

```text
Data/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── blue_hoop/
│   ├── cardboard/
│   ├── fastening/
│   ├── multiple/
│   └── obstruction/
└── ground_truth/
```

### 7.4 Train the baseline

```powershell
python -m src.train --data Data --image-size 128 --epochs 30 --batch-size 8
```

The training pipeline:

- Uses only normal training images.
- Creates a 90/10 training-validation split.
- Uses AdamW optimization.
- Uses a combined L1 and SSIM-style reconstruction loss.
- Selects the best checkpoint by validation loss.
- Saves training history to `training_history.csv`.

Default checkpoint:

```text
checkpoints/tiny_assembly_ae_128.pt
```

### 7.5 Evaluate the trained model

```powershell
python -m src.evaluate --data Data --checkpoint checkpoints/tiny_assembly_ae_128.pt
```

Evaluation outputs are saved under `evaluation/128/`.

## 8. Baseline Evaluation Results — 128 × 128

The following results are from the first reported CPU evaluation run.

### 8.1 Image-level metrics

| Metric | Result |
|---|---:|
| Image AUROC | **0.6064** |
| Accuracy | **0.4992** |
| Precision | **0.8462** |
| Defect recall | **0.0683** |
| F1 score | **0.1264** |
| Balanced accuracy | **0.5271** |
| False reject rate | **0.0140** |
| Decision threshold | 0.510755 |

### 8.2 Confusion matrix

The evaluation contained 285 normal and 322 defective test images.

| | Predicted PASS | Predicted REJECT |
|---|---:|---:|
| Actual GOOD | 281 (TN) | 4 (FP) |
| Actual DEFECT | 300 (FN) | 22 (TP) |

At the selected threshold, the system rejected 26 images: 22 were defective and 4 were normal. It detected 22 of 322 defective images and missed 300.

### 8.3 Defect recall by category

| Defect category | Test images | Recall |
|---|---:|---:|
| blue_hoop | 5 | 0.2000 |
| cardboard | 5 | 0.6000 |
| fastening | 277 | 0.0505 |
| multiple | 33 | 0.0909 |
| obstruction | 2 | 0.5000 |

The category sample sizes for blue_hoop, cardboard and obstruction are very small, so their recall estimates are unstable. Fastening dominates the defective test set and currently has very low recall.

### 8.4 Pixel-level localization

| Metric | Result |
|---|---:|
| Sampled pixel AUROC | **0.7517** |
| Normal pixels sampled | 310,272 |
| Defect pixels sampled | 44,654 |

Pixel AUROC measures how well residual values rank annotated defect pixels above normal pixels. The reported value is based on sampled pixels, not a full-resolution evaluation of every pixel.

It does **not** mean that 75.17% of defect pixels were correctly segmented. A thresholded pixel-level segmentation metric would need to be evaluated separately.

### 8.5 CPU performance and storage

| Measurement | Result |
|---|---:|
| Device | CPU |
| Single-image latency | 18.96 ms |
| Reciprocal throughput | 52.75 images/second |
| Checkpoint file size | 0.4790 MB |
| Trainable parameters | 35,283 |
| FP32 parameter storage | 137.82 KiB |

The checkpoint file is larger than the raw FP32 parameter storage because it also contains checkpoint metadata and optimizer state. The reported latency measures model inference and anomaly scoring, not the entire camera-to-decision factory pipeline.

## 9. Interpretation and Current Limitations

The first experiment establishes a working end-to-end baseline, but its detection performance is not yet satisfactory.

Key observations:

1. **Image-level discrimination is weak.** Image AUROC of 0.6064 is only modestly above random ranking.
2. **Defect recall is very low.** At the selected threshold, the model detects 22 of 322 defective images.
3. **Precision is high at this threshold, but this comes with many missed defects.** A high precision value alone is not evidence of a reliable inspection system.
4. **Pixel ranking is more promising than image-level decisions.** Sampled pixel AUROC is 0.7517, suggesting residual maps contain some localization signal.
5. **The test set is category-imbalanced.** The large fastening category dominates the overall defect results.

### Threshold and validation caveats

The threshold was set to the 95th percentile of anomaly scores from 28 held-out normal validation images. The same validation split was also used for best-checkpoint selection during training.

Therefore, this threshold is preliminary and its false-alarm behavior is not a production guarantee. The small calibration sample also makes the estimated upper tail uncertain. Future experiments should use a separate calibration set or a more robust validation protocol, freeze the threshold before final testing, and report the operating-point trade-off.

The current model is an experimental baseline. It should not be described as production-ready or as achieving reliable automotive defect detection.

## 10. Planned Experiments

The next development steps are:

1. Inspect reconstruction panels and per-image anomaly scores to understand failure cases.
2. Investigate image-level anomaly scoring and threshold calibration.
3. Compare 128 × 128, 256 × 256 and 400 × 400 training runs.
4. Report per-category AUROC and recall where sample sizes support meaningful interpretation.
5. Add pixel-level precision, recall, IoU and Dice at a separately calibrated pixel threshold.
6. Compare TinyAssemblyAE with a custom lightweight, Inception-inspired autoencoder.
7. Export the selected model to ONNX and investigate INT8 quantization.
8. Benchmark model size, memory use and CPU latency under a consistent deployment setup.
9. Build an OpenCV conveyor-line inspection demonstration.

## 11. Author

**Athreya A.**  
B.Tech — Artificial Intelligence and Machine Learning

Interests: Computer Vision, Deep Learning, MLOps and Edge AI.

---

*AssemblyEye is an educational and experimental Computer Vision project. It is not a certified automotive safety or production inspection system.*

