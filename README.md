# Heart Image Segmentation with 3D UNet

A deep learning pipeline for automatic heart segmentation from 3D cardiac MRI images using MONAI and PyTorch. Designed for seamless local development and Kaggle GPU training.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Kaggle Setup](#kaggle-setup)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [Kaggle Training](#kaggle-training)
  - [Kaggle CLI Commands](#kaggle-cli-commands)
- [Configuration](#configuration)
- [Training Results](#training-results)
- [Model Architecture](#model-architecture)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project implements a 3D UNet model for cardiac MRI segmentation using the Medical Segmentation Decathlon Task02_Heart dataset. The pipeline automatically detects the execution environment (local vs Kaggle) and adjusts configurations accordingly.

**Key Highlights:**

- Single `train.py` file works on both local machine and Kaggle
- Auto-installs dependencies on Kaggle
- Early stopping to prevent overfitting
- Automatic Mixed Precision (AMP) for faster GPU training
- TensorBoard logging for real-time monitoring

---

## Features

| Feature | Description |
|---------|-------------|
| **3D UNet** | State-of-the-art architecture for volumetric medical image segmentation |
| **MONAI Framework** | Medical imaging-specific transforms and utilities |
| **Auto Environment Detection** | Automatically switches between local/Kaggle configurations |
| **Mixed Precision Training** | 2-3x faster training with AMP on GPU |
| **Early Stopping** | Stops training when validation stops improving (patience=5) |
| **Checkpoint Resuming** | Resume training from last saved checkpoint |
| **Data Augmentation** | Random flips, rotations, intensity shifts |
| **TensorBoard Logging** | Real-time loss and metric visualization |

---

## Dataset

### Medical Segmentation Decathlon - Task02_Heart

| Property | Value |
|----------|-------|
| **Images** | 20 cardiac MRI scans |
| **Format** | NIfTI (.nii) |
| **Task** | Binary segmentation (heart vs background) |
| **Source** | [Kaggle Dataset](https://www.kaggle.com/datasets/thisisrick25/medical-segmentation-decathlon-heart) |

The dataset is automatically downloaded when running locally (requires Kaggle API credentials).

---

## Project Structure

```bash
heart-image-segmentation/
├── train.py                 # Main training script (standalone, works on Kaggle)
├── config.py                # Configuration for local development
├── test.ipynb               # Interactive notebook for visualization
├── requirements.txt         # Python dependencies
├── kernel-metadata.json     # Kaggle kernel configuration
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── results/                 # Training outputs (model, metrics, logs)
│   ├── best_metric_model.pth
│   ├── last_checkpoint.pth
│   ├── loss_train.npy
│   ├── loss_test.npy
│   ├── metric_train.npy
│   ├── metric_test.npy
│   ├── training_metrics.png
│   └── tensorboard_logs/
└── datasets/                # Downloaded dataset (local only, gitignored)
    └── Task02_Heart/
        ├── imagesTr/
        └── labelsTr/
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for local training)
- Kaggle account (for dataset access and GPU training)

### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/thisisrick25/heart-image-segmentation.git
   cd heart-image-segmentation
   ```

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Kaggle Setup

### Step 1: Create Kaggle API Token

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll to **API** section
3. Click **Create New Token**
4. Download `kaggle.json` file

### Step 2: Configure Kaggle CLI

**Windows:**

```powershell
# Create .kaggle directory
mkdir $env:USERPROFILE\.kaggle

# Move kaggle.json to .kaggle folder
move kaggle.json $env:USERPROFILE\.kaggle\

# Verify installation
kaggle --version
```

**Linux/macOS:**

```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move kaggle.json and set permissions
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Verify installation
kaggle --version
```

### Step 3: Verify Setup

```bash
# List your kernels (should work without errors)
kaggle kernels list --mine
```

---

## Usage

### Local Training

Run a quick 1-epoch test locally:

```bash
python train.py
```

**What happens:**

1. Downloads dataset from Kaggle (first run only)
2. Trains for 1 epoch (configured in `train.py`)
3. Saves outputs to `results/` folder

**Monitor training with TensorBoard:**

```bash
tensorboard --logdir=results/tensorboard_logs
```

Open <http://localhost:6006> in your browser.

---

### Kaggle Training

#### Method 1: Push via CLI (Recommended)

```bash
# Push kernel to Kaggle
kaggle kernels push -p .

# Check status
kaggle kernels status thisisrick25/heart-image-segmentation

# Download results when complete
kaggle kernels output thisisrick25/heart-image-segmentation -p ./results
```

#### Method 2: Kaggle Web UI

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Create new notebook
3. Copy contents of `train.py`
4. Add dataset: `thisisrick25/medical-segmentation-decathlon-heart`
5. Enable GPU in Settings
6. Run all cells

---

### Kaggle CLI Commands

#### Push Kernel

Upload your code to run on Kaggle:

```bash
# Push from current directory (uses kernel-metadata.json)
kaggle kernels push -p .

# Push from specific path
kaggle kernels push -p /path/to/project
```

**Required files:**

- `kernel-metadata.json` - Kernel configuration
- `train.py` - Your training script

#### Check Kernel Status

Monitor your running kernel:

```bash
# Check status
kaggle kernels status <username>/<kernel-slug>

# Example
kaggle kernels status thisisrick25/heart-image-segmentation
```

**Status values:**

| Status | Meaning |
|--------|---------|
| `queued` | Waiting to start |
| `running` | Currently executing |
| `complete` | Finished successfully |
| `error` | Failed with error |
| `cancelAcknowledged` | Cancelled by user |

#### Download Kernel Output

Get training results after completion:

```bash
# Download to specific folder
kaggle kernels output <username>/<kernel-slug> -p <output-path>

# Example: Download to results folder
kaggle kernels output thisisrick25/heart-image-segmentation -p ./results

# Example: Download to kaggle_output folder
kaggle kernels output thisisrick25/heart-image-segmentation -p ./kaggle_output
```

**Downloaded files include:**

- `best_metric_model.pth` - Best trained model
- `last_checkpoint.pth` - Latest checkpoint
- `loss_train.npy`, `loss_test.npy` - Loss history
- `metric_train.npy`, `metric_test.npy` - Dice score history
- `training_metrics.png` - Training plots
- `tensorboard_logs/` - TensorBoard events
- `heart-image-segmentation.log` - Execution log

#### View Kernel Log

Check execution output without downloading:

```bash
# Pull kernel and view log (creates local copy)
kaggle kernels pull <username>/<kernel-slug>
```

#### List Your Kernels

```bash
# List all your kernels
kaggle kernels list --mine

# List with more details
kaggle kernels list --mine --page-size 20
```

#### Other Useful Commands

```bash
# Cancel a running kernel
kaggle kernels status <username>/<kernel-slug>  # Note: No direct cancel via CLI

# List available datasets
kaggle datasets list -s "medical segmentation"

# Download a dataset
kaggle datasets download -d thisisrick25/medical-segmentation-decathlon-heart
```

---

## Configuration

### Training Parameters (in `train.py`)

```python
# Training configuration
SEED = 0                      # Random seed for reproducibility
BATCH_SIZE = 1                # Batch size (1 due to 3D volume memory)
MAX_EPOCHS_KAGGLE = 30        # Max epochs on Kaggle
MAX_EPOCHS_LOCAL = 1          # Quick test locally
EARLY_STOPPING_PATIENCE = 5   # Stop if no improvement for 5 epochs
LEARNING_RATE = 1e-5          # Initial learning rate
WEIGHT_DECAY = 1e-5           # L2 regularization
TEST_INTERVAL = 1             # Validate every epoch
TRAIN_RATIO = 0.8             # 80% train, 20% validation

# Preprocessing
PIXDIM = (1.5, 1.5, 1.0)      # Voxel spacing (mm)
A_MIN = 0                     # Intensity window min
A_MAX = 2000                  # Intensity window max
```

### Kaggle Kernel Configuration (`kernel-metadata.json`)

```json
{
    "id": "thisisrick25/heart-image-segmentation",
    "title": "Heart Image Segmentation",
    "code_file": "train.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": false,
    "enable_gpu": true,
    "enable_internet": true,
    "dataset_sources": [
        "thisisrick25/medical-segmentation-decathlon-heart"
    ]
}
```

---

## Training Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Best Validation Dice** | 0.427 |
| **Best Epoch** | 15 |
| **Final Training Loss** | 0.574 |
| **Final Validation Loss** | 0.575 |
| **Training Time (Kaggle P100)** | ~10 minutes |

### Training Curves

The training produces four key plots:

1. **Training Loss** - Decreases smoothly from 0.60 to 0.57
2. **Validation Loss** - Follows training loss closely (minimal overfitting)
3. **Training Dice** - Increases from 0.40 to 0.43
4. **Validation Dice** - Peaks at 0.427 around epoch 15

### Interpretation

- **Dice Score 0.427**: Indicates 42.7% overlap between predicted and ground truth segmentation
- **Convergence**: Model converges smoothly without instability
- **Overfitting**: Minimal - training and validation curves stay close
- **Early Stopping**: Prevents wasted computation after plateau

---

## Model Architecture

### 3D UNet

```bash
Input (1, D, H, W)
    │
    ├── Encoder
    │   ├── Conv Block (16 channels) + ResUnit
    │   ├── Conv Block (32 channels) + ResUnit
    │   ├── Conv Block (64 channels) + ResUnit
    │   ├── Conv Block (128 channels) + ResUnit
    │   └── Conv Block (256 channels) + ResUnit [Bottleneck]
    │
    ├── Decoder (with skip connections)
    │   ├── UpConv + Concat + Conv Block (128 channels)
    │   ├── UpConv + Concat + Conv Block (64 channels)
    │   ├── UpConv + Concat + Conv Block (32 channels)
    │   └── UpConv + Concat + Conv Block (16 channels)
    │
    └── Output Conv (2 channels) → Sigmoid
    
Output (2, D, H, W) [Background, Heart]
```

### Key Components

- **Normalization**: Batch Normalization
- **Activation**: PReLU
- **Loss Function**: Dice Loss (handles class imbalance)
- **Optimizer**: Adam with AMSGrad
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)

---

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'monai'`

**Cause**: Dependencies not installed on Kaggle

**Solution**: The script auto-installs via pip. If it fails, add manually:

```python
import subprocess
subprocess.check_call(['pip', 'install', 'monai'])
```

#### 2. `TypeError: autocast.__init__() missing 'device_type'`

**Cause**: PyTorch version incompatibility

**Solution**: Already fixed in `train.py`:

```python
with torch.amp.autocast(device_type=device.type):
```

#### 3. `'charmap' codec can't encode characters`

**Cause**: Windows encoding issue with emoji/unicode

**Solution**: Set encoding in PowerShell before running:

```powershell
$env:PYTHONIOENCODING = "utf-8"
kaggle kernels output ...
```

#### 4. Empty log file after download

**Cause**: Kaggle sometimes doesn't capture stdout to log file

**Solution**: Check the actual output files instead:

```bash
dir results
# Look for .npy files and .pth models
```

#### 5. Kernel status shows "error"

**Solution**: Download output to check logs:

```bash
kaggle kernels output thisisrick25/heart-image-segmentation -p ./debug_output
type debug_output\heart-image-segmentation.log
```

### Environment Variables

If Kaggle CLI has issues, ensure credentials are set:

```bash
# Windows PowerShell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY = "your_api_key"

# Linux/macOS
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

---

## References

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- [MONAI Documentation](https://docs.monai.io/)
- [3D U-Net Paper](https://arxiv.org/abs/1606.06650)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Dataset**: Medical Segmentation Decathlon Challenge
- **Framework**: MONAI - Medical Open Network for AI
- **Platform**: Kaggle for free GPU resources
