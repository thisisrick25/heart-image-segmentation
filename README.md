# Heart Image Segmentation

This project implements an automatic heart segmentation algorithm using deep learning (UNet) with the MONAI framework. It processes medical imaging data (DICOM/NIfTI), trains a segmentation model, and evaluates its performance.

## Features

- **Data Preparation**: Converts DICOM series to NIfTI format, groups slices, and cleans up empty labels.
- **Preprocessing**: Loads, normalizes, and transforms 3D medical images for training.
- **Model**: Uses a 3D UNet architecture implemented via MONAI.
- **Training**: Custom training loop with Dice Loss and Adam optimizer.
- **Evaluation**: Calculates Dice metric for model performance assessment.
- **Configuration**: Centralized configuration for easy parameter tuning.

## Dataset

This project uses the **Task02_Heart** dataset from the [Medical Segmentation Decathlon](https://medicaldecathlon.com/).

- **Download**: [Direct link](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar)
- **Automated Download**: Run `python download_data.py` to download and extract the dataset automatically.

## Project Structure

```text
.
├── config.py           # Configuration parameters (paths, hyperparameters)
├── prepare.py          # Data preparation (DICOM to NIfTI, grouping)
├── preprocess.py       # Data loading and augmentation pipeline
├── train.py            # Main training script
├── utilities.py        # Helper functions for training and metrics
├── requirements.txt    # Project dependencies
└── datasets/           # Dataset directory (configured in config.py)
```

## Installation

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Configuration

Edit `config.py` to set your dataset paths and training parameters (batch size, learning rate, epochs).

### 2. Data Preparation

Run `prepare.py` to organize your DICOM files and convert them to NIfTI format:

```bash
python prepare.py
```

### 3. Visualization (Optional)

Run `preprocess.py` to visualize the data loader and transformations:

```bash
python preprocess.py
```

### 4. Training

Run `train.py` to start training the model:

```bash
python train.py
```

The model checkpoints and metrics will be saved in the result directory specified in `config.py`.

## Requirements

- Python 3.x
- PyTorch
- MONAI
- Nibabel
- Dicom2Nifti
- Matplotlib
- Tqdm
- Numpy
