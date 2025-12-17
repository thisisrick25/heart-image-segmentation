# Heart Segmentation with Deep Learning: Technical Deep Dive

> **Author**: thisisrick25  
> **Audience**: Intermediate software engineers and ML engineers

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Medical Image Segmentation](#2-medical-image-segmentation)
3. [Dataset: Medical Segmentation Decathlon](#3-dataset-medical-segmentation-decathlon)
4. [Data Loading & Transforms](#4-data-loading--transforms)
5. [The UNet Architecture](#5-the-unet-architecture)
6. [Loss Functions & Metrics](#6-loss-functions--metrics)
7. [Optimizer & Learning Rate](#7-optimizer--learning-rate)
8. [The Training Loop](#8-the-training-loop)
9. [Visualization & Analysis](#9-visualization--analysis)
10. [Debugging Guide](#10-debugging-guide)
11. [Improvements](#11-improvements)

---

## 1. Project Overview

This project trains a deep learning model to segment the heart in 3D MRI scans. The model classifies each voxel as either "heart" (1) or "background" (0).

**Clinical applications**:

- Measuring heart volume and wall thickness
- Detecting cardiomyopathy and other abnormalities
- Pre-surgical planning

**Input**: 3D MRI scan (grayscale volume)  
**Output**: Binary mask (same dimensions)

---

## 2. Medical Image Segmentation

Segmentation assigns a label to every pixel, unlike classification which assigns one label to the entire image.

### Traditional Methods Fail Here

| Method                        | Failure Mode                                                  |
| ----------------------------- | ------------------------------------------------------------- |
| Thresholding                  | Heart and surrounding tissue have similar intensities         |
| Edge Detection (Sobel, Canny) | Noisy images create false edges; heart boundaries are gradual |
| Region Growing                | Leaks into adjacent tissues with similar values               |
| Atlas Registration            | Fails with abnormal anatomy; computationally expensive        |

Deep learning solves this by learning discriminative features directly from labeled examples. No hand-crafted rules required.

---

## 3. Dataset: Medical Segmentation Decathlon

### Task 02: Heart

| Property         | Value          |
| ---------------- | -------------- |
| Modality         | MRI            |
| Target           | Left Atrium    |
| Training Samples | 20             |
| Test Samples     | 10             |
| Format           | NIfTI (`.nii`) |

### NIfTI vs DICOM vs PNG

| Format    | Use Case                                                             |
| --------- | -------------------------------------------------------------------- |
| DICOM     | Hospital systems (one file per slice, vendor metadata)               |
| PNG/JPEG  | Web images (2D only, lossy compression)                              |
| **NIfTI** | Research (single file for entire 3D volume, preserves voxel spacing) |

NIfTI stores the image data and critical metadata (voxel spacing, orientation) in one file. This is essential for accurate physical measurements.

### Directory Structure

```text
datasets/
├── imagesTr/          # Training MRI scans
├── labelsTr/          # Ground truth masks
└── imagesTs/          # Test scans (no labels)
```

### Small Dataset Problem

20 training samples is typical for medical imaging due to privacy regulations, annotation costs ($50-200/hour for radiologist time), and data rarity. The solution is aggressive data augmentation.

---

## 4. Data Loading & Transforms

### MONAI vs Raw PyTorch

| Capability          | PyTorch                        | MONAI                         |
| ------------------- | ------------------------------ | ----------------------------- |
| Load NIfTI          | Requires nibabel + custom code | `LoadImaged` built-in         |
| 3D transforms       | Manual implementation          | Ready-to-use                  |
| Dictionary pipeline | Handle image/label separately  | Single transform handles both |
| Dataset caching     | DIY                            | `CacheDataset` built-in       |

MONAI eliminates weeks of boilerplate code. Use it.

### Transform Pipeline

```python
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    DivisiblePadd(keys=["image", "label"], k=16),
    ToTensord(keys=["image", "label"]),
])
```

Each transform has a specific purpose:

#### LoadImaged

Loads NIfTI files from disk into NumPy arrays with associated metadata.

#### EnsureChannelFirstd

PyTorch expects `(C, H, W, D)` format. Some loaders return `(H, W, D, C)`. This transform standardizes the format.

#### Spacingd — Voxel Normalization

Different scanners produce different voxel sizes. A 1mm × 1mm × 2mm scan looks different from a 0.5mm × 0.5mm × 1mm scan even for the same patient.

**Target spacing (1.5, 1.5, 1.0) mm** was chosen because:

- Heart is a large structure — sub-millimeter resolution is unnecessary
- Fits comfortably in 16GB GPU memory
- Close to the original dataset's mean resolution

**Interpolation modes**:

- `bilinear` for images: Produces smooth intensity values
- `nearest` for labels: Prevents fractional labels (keeps binary 0/1)

#### NormalizeIntensityd — Z-score Normalization

MRI intensities are arbitrary. One scan might range 0-1000, another 0-5000.

```text
normalized = (x - mean) / std
```

**Z-score vs Min-Max**:

- Z-score: Robust to outliers; preferred for neural networks
- Min-Max: One extreme pixel distorts entire scale; use only when bounded output is required

`nonzero=True` computes statistics only from body voxels, ignoring background.

#### CropForegroundd

MRI scans contain large black borders. Cropping them:

- Reduces memory usage
- Focuses the model on relevant anatomy
- Speeds up training

#### DivisiblePadd

UNet uses 4 downsampling layers with stride 2. This reduces spatial dimensions by 2^4 = 16×. Input dimensions must be divisible by 16 to avoid dimension mismatches during upsampling.

### Data Augmentation

```python
RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
RandAffined(keys=['image', 'label'], prob=0.5, rotate_range=(0.26, 0.26, 0.26), scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.1),
RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
```

| Augmentation          | Simulates                              |
| --------------------- | -------------------------------------- |
| `RandFlipd`           | Patient orientation (left/right lying) |
| `RandRotate90d`       | Scanner orientation                    |
| `RandAffined`         | Patient positioning, slight motion     |
| `RandGaussianNoised`  | Scanner noise, acquisition artifacts   |
| `RandShiftIntensityd` | Scanner calibration differences        |

**Augmentation is applied only during training**. Validation and test sets use deterministic transforms.

**Not included: Elastic deformation**. It's computationally expensive for 3D and the heart doesn't deform as dramatically as brain structures. Add it for further improvement if needed.

---

## 5. The UNet Architecture

### Architecture Selection

| Architecture         | Designed For             | Trade-off                                                       |
| -------------------- | ------------------------ | --------------------------------------------------------------- |
| VGG/ResNet           | Classification           | No upsampling; outputs single label                             |
| FCN                  | Segmentation             | Loses fine spatial details                                      |
| **UNet**             | Biomedical segmentation  | Skip connections preserve boundaries; works with small datasets |
| SegResNet            | Medical Decathlon        | Better performance; more parameters; needs more data            |
| Transformers (UNETR) | Large-scale segmentation | Requires millions of samples; heavy compute                     |

UNet is the right choice for this dataset size. Skip connections are critical for preserving exact heart boundaries.

### Architecture

```python
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2,
)
```

```bash
Input (1 channel)
    │
┌───┴───────────────────────────────────────────────────────┐
│ ENCODER                                                   │
│                                                           │
│  [16] ─────────────────────────────────────────► Skip 1   │
│    │ stride 2                                             │
│  [32] ─────────────────────────────────────► Skip 2       │
│    │ stride 2                                             │
│  [64] ─────────────────────────────────► Skip 3           │
│    │ stride 2                                             │
│  [128] ─────────────────────────► Skip 4                  │
│    │ stride 2                                             │
│  [256] BOTTLENECK                                         │
└───────────────────────────────────────────────────────────┘
    │
┌───┴───────────────────────────────────────────────────────┐
│ DECODER                                                   │
│                                                           │
│  [256] → Upsample → Concat(Skip 4) → [128]                │
│  [128] → Upsample → Concat(Skip 3) → [64]                 │
│  [64]  → Upsample → Concat(Skip 2) → [32]                 │
│  [32]  → Upsample → Concat(Skip 1) → [16]                 │
│                                                           │
│  Final Conv → [2 channels]                                │
└───────────────────────────────────────────────────────────┘
    │
Output (2 channels: background, heart)
```

### Skip Connections

The encoder extracts "what" features at increasingly abstract levels but loses spatial precision during downsampling. The decoder reconstructs spatial resolution but has lost fine details.

Skip connections copy high-resolution features from encoder to decoder. The decoder combines:

- Coarse semantic information (from bottleneck): "This region is heart tissue"
- Fine spatial information (from skips): "The exact boundary is here"

Without skip connections, segmentation boundaries are blurry.

### Parameter Choices

| Parameter                     | Value                     | Rationale                                                                           |
| ----------------------------- | ------------------------- | ----------------------------------------------------------------------------------- |
| `spatial_dims=3`              | 3D convolutions           | 2D loses depth context; heart structure is 3D                                       |
| `in_channels=1`               | Grayscale                 | MRI is single-channel                                                               |
| `out_channels=2`              | Background + Heart        | Standard binary segmentation; extendable to multi-class                             |
| `channels=(16,32,64,128,256)` | Doubling pattern          | Standard CNN practice; more channels at lower resolutions capture abstract features |
| `strides=(2,2,2,2)`           | 4 downsample levels       | 16× reduction fits entire heart in receptive field                                  |
| `num_res_units=2`             | Residual blocks per level | Helps gradient flow in deeper networks                                              |
| `norm=Norm.BATCH`             | Batch normalization       | Stabilizes training; enables higher learning rates                                  |
| `dropout=0.2`                 | 20% dropout               | Regularization; prevents overfitting on small dataset                               |

### 2 Output Channels vs 1

Two channels (background probability, heart probability) is preferred over single channel because:

- Directly compatible with softmax and cross-entropy
- Model explicitly learns to distinguish both classes
- Extends naturally to multi-class (e.g., 4 heart chambers)

---

## 6. Loss Functions & Metrics

### The Class Imbalance Problem

Typical cardiac MRI:

- Background: 95% of voxels
- Heart: 5% of voxels

A model predicting "background everywhere" achieves 95% accuracy but 0% usefulness.

### Loss Function Comparison

| Loss          | Formula                    | Handles Imbalance | Use Case                   |
| ------------- | -------------------------- | ----------------- | -------------------------- |
| Cross-Entropy | `-Σ y·log(p)`              | No                | Balanced classes           |
| **Dice Loss** | `1 - 2·intersection/union` | Yes               | Imbalanced segmentation    |
| IoU Loss      | `1 - intersection/union`   | Yes               | Alternative to Dice        |
| Focal Loss    | `-α(1-p)^γ·log(p)`         | Yes (extreme)     | Extreme imbalance (1000:1) |

**Dice Loss is used** because it directly optimizes the overlap metric, making it immune to class imbalance.

### Dice Coefficient

```text
Dice = (2 × |A ∩ B|) / (|A| + |B|)
```

- A = predicted mask
- B = ground truth mask
- |A ∩ B| = intersection (correctly predicted voxels)

Range: 0 (no overlap) to 1 (perfect overlap)

```python
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
```

| Parameter           | Purpose                                       |
| ------------------- | --------------------------------------------- |
| `to_onehot_y=True`  | Converts label to one-hot encoding            |
| `sigmoid=True`      | Applies sigmoid to raw model outputs          |
| `squared_pred=True` | Uses pred² in denominator; smoother gradients |

### DiceCELoss (Recommended Upgrade)

```python
loss = DiceCELoss(to_onehot_y=True, softmax=True)
```

Combines Dice (handles imbalance) with Cross-Entropy (stable gradients). This typically outperforms pure Dice.

---

## 7. Optimizer & Learning Rate

### Optimizer Selection

| Optimizer | Characteristics                                                | Best For                          |
| --------- | -------------------------------------------------------------- | --------------------------------- |
| SGD       | Requires careful LR tuning; best final performance with budget | Large-scale training              |
| **Adam**  | Adaptive per-parameter LR; works out-of-box; fast convergence  | Small datasets, quick experiments |
| AdamW     | Adam with decoupled weight decay; better generalization        | Recommended upgrade               |

Adam is used because the dataset is small and fast iteration is prioritized over marginal performance gains from tuned SGD.

### Learning Rate

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

| LR       | Behavior                                |
| -------- | --------------------------------------- |
| 1e-2     | Too high; loss explodes                 |
| 1e-3     | Aggressive; can work but often unstable |
| **1e-4** | Standard for Adam with batch norm       |
| 1e-5     | Too low; very slow convergence          |

**Weight decay (1e-5)**: L2 regularization. Penalizes large weights to prevent overfitting.

### Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

| Scheduler         | Behavior                                                   |
| ----------------- | ---------------------------------------------------------- |
| StepLR            | Reduces LR every N epochs; requires knowing when to reduce |
| ReduceLROnPlateau | Reduces when validation metric plateaus; adaptive          |
| CosineAnnealing   | Smooth cosine decay; can escape local minima               |
| OneCycleLR        | Warmup → peak → decay; SOTA but complex setup              |

**ReduceLROnPlateau** is chosen for simplicity. It watches validation Dice and reduces LR by 50% if no improvement for 5 epochs.

---

## 8. The Training Loop

### Training Step

```python
model.train()
for batch_data in train_loader:
    image = batch_data["image"].to(device)
    label = batch_data["label"].to(device)
    label = label != 0  # Binary conversion

    optimizer.zero_grad()
    outputs = model(image)
    loss = loss_function(outputs, label)
    loss.backward()
    optimizer.step()
```

| Step               | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `model.train()`    | Enables dropout and batch norm updates               |
| `.to(device)`      | Moves tensors to GPU                                 |
| `label != 0`       | Converts multi-class labels to binary                |
| `zero_grad()`      | Clears accumulated gradients from previous iteration |
| `model(image)`     | Forward pass                                         |
| `loss.backward()`  | Backpropagation                                      |
| `optimizer.step()` | Weight update                                        |

### Automatic Mixed Precision (AMP)

```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type='cuda'):
    outputs = model(image)
    loss = loss_function(outputs, label)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

| Mode           | Memory | Speed |
| -------------- | ------ | ----- |
| FP32           | 100%   | 1×    |
| **FP16 (AMP)** | ~50%   | ~2×   |

AMP automatically uses FP16 for convolutions and FP32 for sensitive operations (loss, accumulation). The GradScaler prevents gradient underflow in FP16.

### Validation

```python
model.eval()
with torch.no_grad():
    for test_data in test_loader:
        outputs = model(test_image)
        dice = dice_metric(outputs, test_label)
```

| Mode                   | Gradient Tracking | Dropout  | BatchNorm             |
| ---------------------- | ----------------- | -------- | --------------------- |
| `train()`              | On                | Active   | Updates running stats |
| `eval()` + `no_grad()` | Off               | Disabled | Uses running stats    |

`torch.no_grad()` saves memory and speeds up computation by disabling gradient tracking.

### Early Stopping

```python
if epoch_metric > best_metric:
    best_metric = epoch_metric
    epochs_no_improve = 0
    torch.save(model.state_dict(), "best_metric_model.pth")
else:
    epochs_no_improve += 1
    if epochs_no_improve >= 15:
        break
```

**Patience = 15**: Stop training if validation Dice doesn't improve for 15 consecutive epochs. This prevents overfitting.

Lower patience (5) risks stopping during temporary plateaus. Higher patience (50) wastes time if truly overfitting.

---

## 9. Visualization & Analysis

### Validation Set

| Panel        | Content                         |
| ------------ | ------------------------------- |
| MRI Slice    | Raw input                       |
| Ground Truth | Radiologist annotation          |
| Prediction   | Model output                    |
| Overlay      | Green=GT, Red=Pred, Yellow=Both |

Dice score is displayed in the title.

### Test Set

| Metric     | Meaning                                 |
| ---------- | --------------------------------------- |
| Confidence | Mean probability in predicted regions   |
| Volume %   | Percentage of image classified as heart |

### Extended Visualization

| View                | Purpose                      |
| ------------------- | ---------------------------- |
| Probability Heatmap | Identify uncertain regions   |
| Contour Overlay     | Compare boundaries precisely |
| Hybrid              | Combined analysis            |

---

## 10. Debugging Guide

### Out of Memory

| Cause      | Solution                                |
| ---------- | --------------------------------------- |
| Batch size | Already 1; cannot reduce further        |
| Image size | Reduce spatial dimensions in transforms |
| Model size | Reduce `channels` tuple values          |
| No AMP     | Enable AMP for 50% memory reduction     |

### NaN Loss

| Cause            | Solution                                     |
| ---------------- | -------------------------------------------- |
| LR too high      | Reduce by 10×                                |
| Division by zero | Add epsilon: `x / (y + 1e-8)`                |
| Bad input data   | Visualize inputs; check for all-zero volumes |

### Low Dice Score

| Training  | Validation | Problem      | Solution                          |
| --------- | ---------- | ------------ | --------------------------------- |
| Low       | Low        | Underfitting | More epochs, larger model         |
| High      | Low        | Overfitting  | More augmentation, early stopping |
| Both slow | —          | LR too low   | Increase LR                       |

---

## 11. Improvements

### Tier 1: Quick Wins (Code Changes Only)

#### DiceCELoss Upgrade

```python
from monai.losses import DiceCELoss

loss = DiceCELoss(to_onehot_y=True, softmax=True)
```

| Component     | Role                                                        |
| ------------- | ----------------------------------------------------------- |
| Cross-Entropy | Stable gradients early in training; pixel-level supervision |
| Dice          | Handles class imbalance; optimizes overlap directly         |

Combined loss typically outperforms either alone by 1-3% Dice.

#### Optimizer

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

| Adam                                       | AdamW                           |
| ------------------------------------------ | ------------------------------- |
| Weight decay coupled with gradient updates | Weight decay applied separately |
| Can lead to suboptimal regularization      | Better generalization           |

Note: AdamW uses higher weight decay (1e-2 vs 1e-5) because the decoupling makes it more effective.

#### CosineAnnealing Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

| ReduceLROnPlateau             | CosineAnnealingWarmRestarts             |
| ----------------------------- | --------------------------------------- |
| Reactive (waits for plateau)  | Proactive (scheduled restarts)          |
| Can get stuck in local minima | Periodic "restarts" escape local minima |
| Simple                        | Requires tuning T_0 and T_mult          |

Parameters:

- `T_0=10`: First restart after 10 epochs
- `T_mult=2`: Each subsequent period doubles (10, 20, 40, ...)

---

### Tier 2: Architecture Upgrade

#### SegResNet

```python
from monai.networks.nets import SegResNet

model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=32,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
)
```

| UNet                                     | SegResNet                          |
| ---------------------------------------- | ---------------------------------- |
| Skip connections between encoder/decoder | Residual connections throughout    |
| 4 pooling levels                         | Configurable depth                 |
| Good baseline                            | Won Medical Segmentation Decathlon |

SegResNet adds residual connections within each block, enabling deeper networks without vanishing gradients. Expect 2-5% Dice improvement.

---

### Tier 3: Data Strategy

#### Elastic Deformation

```python
from monai.transforms import Rand3DElasticd

Rand3DElasticd(
    keys=["image", "label"],
    sigma_range=(5, 8),
    magnitude_range=(100, 200),
    prob=0.5,
    mode=("bilinear", "nearest"),
)
```

| Simple Augmentation (flip, rotate) | Elastic Deformation            |
| ---------------------------------- | ------------------------------ |
| Rigid transformations              | Non-rigid, realistic warping   |
| Fast                               | Computationally expensive      |
| Limited variety                    | Simulates anatomical variation |

Hearts deform non-linearly. Elastic deformation creates realistic variations that rigid transforms cannot.

#### 5-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(data_dicts)):
    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]

    # Train model for this fold
    model = train_model(train_files, val_files)
    models.append(model)

# Inference: average predictions from all 5 models
def ensemble_predict(image):
    predictions = [model(image) for model in models]
    return torch.mean(torch.stack(predictions), dim=0)
```

| Single Split (80/20)              | 5-Fold Cross-Validation              |
| --------------------------------- | ------------------------------------ |
| 16 training samples, 4 validation | Each sample used for validation once |
| High variance in results          | More robust evaluation               |
| One model                         | 5 models for ensemble                |

Cross-validation is the most reliable way to improve leaderboard scores. Ensemble of 5 models typically adds 1-3% Dice.

#### Test-Time Augmentation (TTA)

```python
def predict_with_tta(model, image):
    predictions = []

    # Original
    predictions.append(model(image))

    # Flipped versions
    for axis in [2, 3, 4]:  # H, W, D
        flipped = torch.flip(image, dims=[axis])
        pred = model(flipped)
        pred = torch.flip(pred, dims=[axis])  # Flip back
        predictions.append(pred)

    # Average all predictions
    return torch.mean(torch.stack(predictions), dim=0)
```

| Standard Inference  | TTA                                        |
| ------------------- | ------------------------------------------ |
| Single forward pass | Multiple forward passes with augmentations |
| Fast                | 4-8× slower                                |
| No extra accuracy   | +0.5-1.5% Dice                             |

TTA averages predictions over augmented versions of the input. The ensemble effect reduces noise in predictions.

---

### Tier 4: Post-Processing

#### Connected Component Analysis

```python
from scipy.ndimage import label
import numpy as np

def keep_largest_component(mask):
    """Remove small spurious predictions."""
    labeled, num_features = label(mask)
    if num_features == 0:
        return mask

    # Find size of each component
    sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    largest = np.argmax(sizes) + 1

    return (labeled == largest).astype(mask.dtype)
```

The model sometimes predicts small blobs far from the heart. This keeps only the largest connected region.

#### Conditional Random Field (CRF)

```python
import pydensecrf.densecrf as dcrf

def apply_crf(image, probabilities):
    """Refine segmentation using CRF."""
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)

    # Unary potentials (from model)
    U = -np.log(probabilities + 1e-8)
    d.setUnaryEnergy(U.reshape((2, -1)).astype(np.float32))

    # Pairwise potentials (appearance + smoothness)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape(image.shape[:2])
```

| Raw Model Output           | CRF Post-Processing                  |
| -------------------------- | ------------------------------------ |
| May have jagged boundaries | Smoother, more coherent boundaries   |
| Fast                       | Adds processing time                 |
| Grid-like artifacts        | Uses image appearance for refinement |

CRF is useful when model boundaries are noisy. It uses the original image appearance to refine predictions.

---

### Improvement Priority

| Priority | Improvement         | Expected Gain  | Effort                  |
| -------- | ------------------- | -------------- | ----------------------- |
| 1        | DiceCELoss          | +1-3% Dice     | 1 line                  |
| 2        | AdamW               | +0.5-1% Dice   | 1 line                  |
| 3        | SegResNet           | +2-5% Dice     | Architecture swap       |
| 4        | 5-Fold CV           | +1-3% Dice     | Training infrastructure |
| 5        | Elastic Deformation | +1-2% Dice     | Add transform           |
| 6        | TTA                 | +0.5-1.5% Dice | Inference wrapper       |
| 7        | CosineAnnealing     | +0.5-1% Dice   | 1 line                  |
| 8        | Post-processing     | +0.3-0.5% Dice | Post-inference step     |

Start from the top. Each improvement is additive.

---

## Summary

| Component      | Choice                   | Rationale                                              |
| -------------- | ------------------------ | ------------------------------------------------------ |
| Framework      | MONAI                    | Built for medical imaging                              |
| Architecture   | 3D UNet                  | Skip connections for boundaries; works with small data |
| Loss           | Dice                     | Handles class imbalance                                |
| Optimizer      | Adam                     | Fast convergence, minimal tuning                       |
| Scheduler      | ReduceLROnPlateau        | Adaptive to training progress                          |
| Regularization | Dropout + Early Stopping | Prevents overfitting                                   |

Each choice is deliberate. Adapt based on your specific constraints and dataset.
