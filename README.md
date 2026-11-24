# AssemblyEye 🔍
### Lightweight AI-Powered Visual Anomaly Detection for Automotive Manufacturing

AssemblyEye is a lightweight, scratch-built Computer Vision system designed for **unsupervised visual anomaly detection in automotive assembly lines**.

The project explores how deep learning can automate visual quality inspection by learning the appearance of normal automotive assemblies and identifying potential defects through reconstruction errors.

Unlike conventional supervised defect classifiers, AssemblyEye trains its baseline model using only defect-free assembly images.

---

## 🎯 Problem Statement

Automotive manufacturing lines require accurate inspection to identify assembly defects such as:

- Missing or incorrectly fastened clips.
- Misrouted electrical wiring.
- Incorrectly positioned components.
- Other visual abnormalities in automotive assemblies.

Manual inspection can be time-consuming and difficult to scale consistently.

AssemblyEye explores an automated visual inspection approach using a compact neural network suitable for resource-constrained computing environments.

## 💡 Proposed Solution

We use an **unsupervised convolutional autoencoder** to learn the visual characteristics of normal automotive assemblies.

The system follows this workflow:

1. Train the model exclusively on normal assembly images.
2. Reconstruct incoming assembly images.
3. Calculate pixel-wise reconstruction differences.
4. Generate an anomaly heatmap highlighting suspicious regions.
5. Apply an anomaly threshold to produce a PASS / REJECT decision.

## 🧠 Model Architecture — TinyAssemblyAE

TinyAssemblyAE is a custom-built Depthwise-Separable Convolutional Autoencoder implemented from scratch using PyTorch.

| Component | Architecture |
|---|---|
| Input | 3 × 128 × 128 RGB |
| Encoder | Conv2D + 3 Depthwise-Separable Conv blocks |
| Encoder channels | 3 → 16 → 32 → 64 → 128 |
| Bottleneck | 128 → 32 → 128 |
| Decoder | Bilinear upsampling + Depthwise-Separable Conv blocks |
| Output | Reconstructed RGB image |
| Activation | SiLU / Sigmoid |
| Skip connections | None |

**Initial architecture benchmark:**

- Trainable parameters: 35,283
- FP32 parameter storage: approximately 137.82 KiB
- Supports forward-pass tests at 128 × 128, 256 × 256 and 400 × 400.

## 📊 Dataset

**Dataset:** Renault AutoVI v1.0.0

The project uses the `engine_wiring` category.

Dataset distribution:

| Dataset split | Images |
|---|---:|
| Training — Good | 285 |
| Testing — Good | 285 |
| Testing — Defective | 322 |
| Total | 892 |

Defect categories:

- blue_hoop
- cardboard
- fastening
- multiple
- obstruction

All 322 defective test images were found to have corresponding ground-truth masks during dataset inspection.

Ground-truth masks are reserved for evaluation and are not used to train the baseline autoencoder.

## ⚙️ Technology Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- OpenCV (planned for conveyor simulation)
- ONNX (planned for deployment)

## 🏗️ Project Structure

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
│   └── models/
│       ├── __init__.py
│       └── tiny_assembly_ae.py
│
├── tests/
│   ├── test_model.py
│   └── test_dataloader.py
│
├── scripts/
│   └── inspect_dataset.py
│
├── visualization/
│   └── visualize_dataset.py
│
├── checkpoints/
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AssemblyEye.git
cd AssemblyEye
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download Renault AutoVI v1.0.0 and extract the `engine_wiring` category.

Arrange the files as follows:

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

### 5. Test the model

```bash
python -m tests.test_model
```

### 6. Train TinyAssemblyAE

Run from the project root:

```bash
python -m src.train --data Data --image-size 128 --epochs 30 --batch-size 8
```

The best validation checkpoint is saved under `checkpoints/`.

Training history is exported to `training_history.csv`.

## 📈 Evaluation Strategy

The planned evaluation pipeline will measure:

- Image-level AUROC.
- Pixel-level AUROC.
- Defect recall.
- Normal false-reject rate.
- Reconstruction quality.
- CPU inference latency.
- Model storage size.

The system will also generate visual comparisons containing:

`Input Image | Reconstruction | Anomaly Heatmap | Ground Truth Mask`

Thresholds will be calibrated using held-out normal validation images before final test evaluation.

## 🖥️ Edge Deployment — Planned

The deployment phase will investigate:

- ONNX model export.
- INT8 quantization.
- CPU inference optimization.
- Model size and latency benchmarking.
- OpenCV-based conveyor-line inspection simulation.

## 🔬 Future Experiments

1. Compare input resolutions: 128 × 128, 256 × 256 and 400 × 400.
2. Develop a lightweight Inception-inspired autoencoder.
3. Compare reconstruction quality and anomaly localization.
4. Investigate quantization and edge inference performance.
5. Build a real-time-style automotive inspection demonstration.

## 📌 Project Status

**Currently implemented:**
- AutoVI dataset exploration.
- Ground-truth mask inspection and visualization.
- PyTorch Dataset and DataLoader.
- TinyAssemblyAE architecture.
- Forward-pass validation across three resolutions.
- Initial training and checkpointing pipeline.

**Still in progress:**
- First successful training run.
- Anomaly heatmap evaluation.
- Architecture comparison.
- ONNX and INT8 deployment.
- Conveyor-line demonstration.

## 👨‍💻 Author

**Athreya A.**

B.Tech — Artificial Intelligence and Machine Learning

Interested in Computer Vision, Deep Learning, MLOps and Edge AI.

---

*AssemblyEye is an educational and experimental Computer Vision engineering project. It is not a certified automotive safety or production inspection system.*

**Dataset attribution:** Renault AutoVI v1.0.0. Refer to the original dataset documentation and included license for attribution and usage conditions.
