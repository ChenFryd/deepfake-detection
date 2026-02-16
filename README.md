# Deepfake Generation Method Classification Using Artifact-Oriented Features

A research project that goes beyond binary deepfake detection (real vs. fake) to identify **which specific manipulation method** was used to generate a deepfake video. This enables forensic attribution and supports targeted defense strategies.

**Course:** Offensive AI
**Instructor:** Dr. Yisroel Mirsky
**Authors:** Yuval Felendler, Chen Frydman, Amit Avraham, and Mustafa Hijazi

## Problem Statement

Most existing deepfake research focuses on binary classification (real vs. fake). This project tackles the harder problem of **multi-class attribution** — classifying which generation method produced a given manipulated video among four techniques:

| Method | Description |
|---|---|
| **Deepfakes** | Autoencoder-based identity swap |
| **Face2Face** | Expression reenactment transfer |
| **FaceSwap** | Classical 3D face alignment and texture transfer |
| **NeuralTextures** | Neural rendering-based texture synthesis |

## Dataset

- **FaceForensics++** (c23 compression)
- 120 videos per manipulation method (480 total)
- Only manipulated videos are used (focus is on attribution, not detection)

## Methodology

### Pipeline

1. **Input** — Videos from FaceForensics++
2. **Preprocessing** — Uniform frame sampling (24 frames/video) and face detection via MTCNN (160x160 crops). Videos with fewer than 18 valid face detections are excluded.
3. **Feature Extraction** — Six artifact-oriented feature sets (see below)
4. **Classification** — Traditional ML classifiers evaluated with stratified 10-fold cross-validation

### Feature Sets

| # | Feature | Signal Captured |
|---|---|---|
| 1 | **FFT (Radial)** | Spectral energy distribution across low/mid/high frequency bands — captures upsampling noise from generator models |
| 2 | **Geometric Jitter** | Temporal facial edge centroid displacement — measures geometric instability across frames |
| 3 | **Pixel Temporal** | Frame-to-frame intensity differences and high-frequency spectral energy — pixel-level inconsistencies |
| 4 | **Face Embeddings** | Temporal descriptors from InceptionResNetV1 (VGGFace2) — captures identity and texture-level variations |
| 5 | **Background** | Temporal embeddings of background regions — analyzes manipulation artifacts outside the face area |
| 6 | **Optical Flow** | Pixel-wise warping residuals between consecutive frames — quantifies temporal incoherence |

### Classifiers

- **Random Forest** (300 trees, balanced weights) — best overall performer
- SVM (RBF kernel)
- KNN
- XGBoost

### Evaluation Tasks

- **One-vs-All** — Binary classification per method
- **Pairwise** — Discriminating between all 6 unique method pairs
- **Multi-class** — Full 4-way classification

## Results

### Multi-Class Classification (Random Forest)

| Feature Type | Accuracy (%) | AUC (%) |
|---|---|---|
| Face Embeddings | **38.8 +/- 6.2** | **66.9 +/- 4.5** |
| FFT (Radial) | 26.0 +/- 6.3 | 54.3 +/- 4.3 |
| Pixel Temporal | 24.0 +/- 4.9 | 47.8 +/- 5.3 |
| Landmark Jitter | 24.6 +/- 5.7 | 47.6 +/- 6.6 |
| Optical Flow | 25.6 +/- 5.7 | 49.7 +/- 4.8 |
| Background | 20.8 +/- 4.6 | 39.9 +/- 3.1 |
| *Random Baseline* | *25.0* | *50.0* |

- **Face Embeddings** achieved the best performance (38.8% accuracy, 66.9% AUC)
- **FFT (Radial)** was the second-best feature (54.3% AUC)
- Optical Flow and Landmark Jitter performed near chance level

## Project Structure

```
.
├── Final_Project.ipynb          # Full experiment notebook (Google Colab)
├── group 3 - Deepfake Generation method classification.pdf   # Presentation slides
├── group 3 - OAI deepfakes.pdf  # Supplementary document
└── README.md
```

## Requirements

```
torch==2.1.2
torchvision==0.16.2
facenet-pytorch==2.5.3
numpy==1.24.4
opencv-python-headless==4.8.1.78
scipy
scikit-learn
xgboost
tqdm
matplotlib
pandas
```

## Usage

The notebook is designed to run on **Google Colab** with GPU support.

1. Upload `Final_Project.ipynb` to Google Colab
2. Mount your Google Drive when prompted
3. Run the dataset download cells to fetch FaceForensics++ videos (requires access credentials)
4. Execute the feature extraction pipeline (two loops: temporal features and embedding features)
5. Run the evaluation and visualization cells

Extracted features are cached as `.npz` files to Google Drive to avoid recomputation.
