"""
Training Script for Heart Segmentation
Works both locally and on Kaggle (auto-detects environment)
"""

import os
import torch
from pathlib import Path
import config

# Detect if running on Kaggle
KAGGLE_ENV = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

if KAGGLE_ENV:
    print("🌐 Running on Kaggle environment")
    KAGGLE_INPUT = Path("/kaggle/input")
    KAGGLE_WORKING = Path("/kaggle/working")
    # Use the Kaggle dataset - dataset is inside Task02_Heart subdirectory
    DATASET_DIR = KAGGLE_INPUT / "medical-segmentation-decathlon-heart" / "Task02_Heart"
    # On Kaggle: save to working dir, then copy to repo results/ for GitHub
    MODEL_RESULT_PATH = KAGGLE_WORKING / "results"
else:
    print("💻 Running locally")
    # Download dataset from Kaggle using Kaggle API
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATASET_DIR = Path(config.DATASET_DIR)

    # Download dataset directly to datasets directory if not already there
    if not DATASET_DIR.exists() or not (DATASET_DIR / "imagesTr").exists():
        print(f"Downloading dataset to: {DATASET_DIR}")

        # Workaround for Kaggle API bug - set empty env var to prevent KeyError
        if "KAGGLE_API_TOKEN" not in os.environ:
            os.environ["KAGGLE_API_TOKEN"] = ""

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        import shutil

        DATASET_DIR.mkdir(parents=True, exist_ok=True)

        api.dataset_download_files(
            config.KAGGLE_DATASET,
            path=DATASET_DIR,
            unzip=True,
            quiet=False
        )

        # Move Task02_Heart contents up to DATASET_DIR if needed
        task_dir = DATASET_DIR / "Task02_Heart"
        if task_dir.exists():
            for item in task_dir.iterdir():
                shutil.move(str(item), str(DATASET_DIR / item.name))
            task_dir.rmdir()

        # Clean up macOS/archive artifacts (._* files)
        artifact_files = list(DATASET_DIR.rglob("._*"))
        if artifact_files:
            print(f"Cleaning up {len(artifact_files)} artifact files (._*)...")
            for artifact in artifact_files:
                artifact.unlink()
            print("✓ Artifact files removed")

        print("✓ Dataset downloaded to datasets/ directory")
    else:
        print("✓ Using existing dataset in project directory")

    # Locally: results/ dir exists but is tracked in git
    MODEL_RESULT_PATH = Path(config.MODEL_RESULT_PATH)
SEED = config.SEED
BATCH_SIZE = config.BATCH_SIZE
MAX_EPOCHS = config.MAX_EPOCHS_LOCAL if not KAGGLE_ENV else config.MAX_EPOCHS
LEARNING_RATE = config.LEARNING_RATE
WEIGHT_DECAY = config.WEIGHT_DECAY
TEST_INTERVAL = config.TEST_INTERVAL
TRAIN_RATIO = config.TRAIN_RATIO
SPATIAL_SIZE = config.SPATIAL_SIZE
PIXDIM = config.PIXDIM
A_MIN = config.A_MIN
A_MAX = config.A_MAX

print(f"Training for {MAX_EPOCHS} epoch(s)")


def check_dataset():
    """Verify dataset is available"""
    if DATASET_DIR.exists() and (DATASET_DIR / "imagesTr").exists():
        print(f"✓ Dataset found at {DATASET_DIR}")
        return
    else:
        print("⚠️ Dataset not found!")
        if KAGGLE_ENV:
            print("Please attach the dataset: https://www.kaggle.com/datasets/vivekprajapati2048/medical-segmentation-decathlon-heart")
            print(f"Expected path: {DATASET_DIR}")
            raise FileNotFoundError(
                "Dataset not attached to Kaggle kernel. Please add it in the kernel settings.")
        else:
            print(f"Expected path: {DATASET_DIR}")
            raise FileNotFoundError(
                "Dataset not found. Make sure kagglehub downloaded it correctly.")


def prepare_data():
    """Prepare data loaders for training"""
    import glob
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
        Orientationd, Spacingd, ToTensord
    )
    from monai.data import CacheDataset, DataLoader
    from monai.utils import set_determinism

    set_determinism(seed=SEED)

    images_path = DATASET_DIR / "imagesTr"
    labels_path = DATASET_DIR / "labelsTr"

    # Get all image and label files
    all_images = sorted(glob.glob(str(images_path / "*.nii.gz")))
    all_labels = sorted(glob.glob(str(labels_path / "*.nii.gz")))

    print(f"Found {len(all_images)} images and {len(all_labels)} labels")

    # Create data dicts
    data_dicts = [
        {"image": image, "label": label}
        for image, label in zip(all_images, all_labels)
    ]

    # Shuffle and split
    import random
    random.seed(SEED)
    random.shuffle(data_dicts)

    split_idx = int(len(data_dicts) * TRAIN_RATIO)
    train_files = data_dicts[:split_idx]
    test_files = data_dicts[split_idx:]

    print(
        f"Training samples: {len(train_files)}, Validation samples: {len(test_files)}")

    # Import augmentation transforms
    from monai.transforms import RandFlipd, RandRotate90d, RandShiftIntensityd

    # Define transforms with augmentation for training
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM,
                 mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=A_MIN, a_max=A_MAX, b_min=0.0, b_max=1.0, clip=True),
        # Data augmentation (only for training)
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])

    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM,
                 mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=A_MIN, a_max=A_MAX, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
    ])

    # Create datasets - use CacheDataset for faster training
    print("Creating training dataset with caching...")
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("Creating validation dataset with caching...")
    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=4
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    return train_loader, test_loader


def train_model(train_loader, test_loader):
    """Train the segmentation model"""
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.losses import DiceLoss
    from tqdm import tqdm
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create model
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # Loss and optimizer
    loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(
        model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)

    # Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Automatic Mixed Precision (AMP) scaler for faster training
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    use_amp = scaler is not None
    if use_amp:
        print("✓ Using Automatic Mixed Precision (AMP) for faster training")

    # Helper function for dice metric
    def dice_metric(predicted, target):
        dice_value = DiceLoss(
            to_onehot_y=True, sigmoid=True, squared_pred=True)
        return 1 - dice_value(predicted, target).item()

    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    start_epoch = 0

    # Try to resume from checkpoint
    checkpoint_path = MODEL_RESULT_PATH / "last_checkpoint.pth"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        best_metric_epoch = checkpoint['best_metric_epoch']
        save_loss_train = checkpoint.get('loss_train', [])
        save_loss_test = checkpoint.get('loss_test', [])
        save_metric_train = checkpoint.get('metric_train', [])
        save_metric_test = checkpoint.get('metric_test', [])
        if use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(
            f"✓ Resumed from epoch {start_epoch}, best Dice: {best_metric:.4f}")

    MODEL_RESULT_PATH.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_log_dir = MODEL_RESULT_PATH / "tensorboard_logs"
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    print(f"📊 TensorBoard logs: {tensorboard_log_dir}")
    print(f"   View with: tensorboard --logdir={tensorboard_log_dir}\n")

    for epoch in range(start_epoch, MAX_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
        print(f"{'='*50}")

        model.train()
        train_epoch_loss = 0
        epoch_metric_train = 0
        train_step = 0

        for batch_data in tqdm(train_loader, desc="Training"):
            train_step += 1
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)
            label = label != 0  # Convert to binary

            optimizer.zero_grad()

            # Use AMP if available
            if use_amp:
                with torch.amp.autocast():
                    outputs = model(image)
                    train_loss = loss_function(outputs, label)
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(image)
                train_loss = loss_function(outputs, label)
                train_loss.backward()
                optimizer.step()

            train_epoch_loss += train_loss.item()
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric

        train_epoch_loss /= train_step
        epoch_metric_train /= train_step
        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)

        print(f"Training Loss: {train_epoch_loss:.4f}")
        print(f"Training Dice: {epoch_metric_train:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Dice/train', epoch_metric_train, epoch)

        # Save training metrics
        np.save(MODEL_RESULT_PATH / 'loss_train.npy', save_loss_train)
        np.save(MODEL_RESULT_PATH / 'metric_train.npy', save_metric_train)

        # Validation
        if (epoch + 1) % TEST_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in tqdm(test_loader, desc="Validation"):
                    test_step += 1
                    test_image = test_data["image"].to(device)
                    test_label = test_data["label"].to(device)
                    test_label = test_label != 0

                    test_outputs = model(test_image)
                    test_loss = loss_function(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()

                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric

                test_epoch_loss /= test_step
                epoch_metric_test /= test_step
                save_loss_test.append(test_epoch_loss)
                save_metric_test.append(epoch_metric_test)

                print(f"Validation Loss: {test_epoch_loss:.4f}")
                print(f"Validation Dice: {epoch_metric_test:.4f}")

                # Log to TensorBoard
                writer.add_scalar('Loss/validation', test_epoch_loss, epoch)
                writer.add_scalar('Dice/validation', epoch_metric_test, epoch)

                # Save test metrics
                np.save(MODEL_RESULT_PATH / 'loss_test.npy', save_loss_test)
                np.save(MODEL_RESULT_PATH / 'metric_test.npy', save_metric_test)

                # Update learning rate based on validation performance
                scheduler.step(epoch_metric_test)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               MODEL_RESULT_PATH / "best_metric_model.pth")
                    print(f"✓ New best model saved! Dice: {best_metric:.4f}")

                print(
                    f"Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

        # Save checkpoint after each epoch for resuming
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
            'loss_train': save_loss_train,
            'loss_test': save_loss_test,
            'metric_train': save_metric_train,
            'metric_test': save_metric_test,
        }
        if use_amp:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, MODEL_RESULT_PATH / "last_checkpoint.pth")

    # Close TensorBoard writer
    writer.close()

    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"Best model: {MODEL_RESULT_PATH / 'best_metric_model.pth'}")
    print(f"Last checkpoint: {MODEL_RESULT_PATH / 'last_checkpoint.pth'}")
    print(f"TensorBoard logs: {tensorboard_log_dir}")
    print(f"{'='*50}")

    return save_loss_train, save_loss_test, save_metric_train, save_metric_test


def plot_metrics(loss_train, loss_test, metric_train, metric_test):
    """Plot training metrics"""
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        ax1.plot(loss_train)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        ax2.plot(loss_test)
        ax2.set_title("Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True)

        ax3.plot(metric_train)
        ax3.set_title("Training Dice Metric")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Dice")
        ax3.grid(True)

        ax4.plot(metric_test)
        ax4.set_title("Validation Dice Metric")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Dice")
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(MODEL_RESULT_PATH / "training_metrics.png",
                    dpi=150, bbox_inches='tight')
        print(
            f"\n✓ Training plots saved to: {MODEL_RESULT_PATH / 'training_metrics.png'}")

        if not KAGGLE_ENV:
            plt.show()
    except Exception as e:
        print(f"Could not generate plots: {e}")


def main():
    """Main training pipeline"""
    print("🚀 Starting Heart Segmentation Training Pipeline")
    print(f"Environment: {'Kaggle' if KAGGLE_ENV else 'Local'}")

    # Step 1: Verify dataset is available
    check_dataset()

    # Step 2: Prepare data
    print("\n📊 Preparing data...")
    train_loader, test_loader = prepare_data()

    # Step 3: Train model
    print("\n🏋️ Starting training...")
    loss_train, loss_test, metric_train, metric_test = train_model(
        train_loader, test_loader)

    # Step 4: Plot results
    print("\n📈 Generating plots...")
    plot_metrics(loss_train, loss_test, metric_train, metric_test)

    print("\n✓ Pipeline completed successfully!")


if __name__ == '__main__':
    main()
