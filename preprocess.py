import os
from glob import glob
from typing import Tuple, List, Dict, Any

from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)

from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism, first
import matplotlib.pyplot as plt
import config


def prepare(data_path: str,
            pixdim: Tuple[float, float, float] = config.PIXDIM,
            a_min: float = config.A_MIN,
            a_max: float = config.A_MAX,
            spatial_size: List[int] = config.SPATIAL_SIZE,
            cache: bool = False) -> Tuple[DataLoader, DataLoader]:
    '''
    Prepares the data loaders for training and testing using logical splitting.

    Args:
        data_path: Path to the dataset directory (containing imagesTr and labelsTr).
        pixdim: Pixel dimensions for spacing.
        a_min: Minimum intensity for scaling.
        a_max: Maximum intensity for scaling.
        spatial_size: Spatial size for resizing (unused currently).
        cache: Whether to use CacheDataset.

    Returns:
        A tuple of (train_loader, test_loader).
    '''
    set_determinism(seed=config.SEED)

    # Load all files from the MSD structure
    images_dir = config.IMAGES_TR_PATH
    labels_dir = config.LABELS_TR_PATH

    # Get all image files
    all_images = sorted(glob(os.path.join(images_dir, '*.nii.gz')))
    all_labels = sorted(glob(os.path.join(labels_dir, '*.nii.gz')))

    # Ensure we have matching pairs
    if len(all_images) != len(all_labels):
        raise ValueError(
            f"Mismatch in number of images ({len(all_images)}) and labels ({len(all_labels)})")

    # Create dictionary of all files
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(all_images, all_labels)
    ]

    # Shuffle and split logically
    import random
    random.seed(config.SEED)
    random.shuffle(data_dicts)

    split_idx = int(len(data_dicts) * config.TRAIN_RATIO)
    train_files = data_dicts[:split_idx]
    test_files = data_dicts[split_idx:]

    print(f"Total samples: {len(data_dicts)}")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(test_files)}")

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim,
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Resized(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim,
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Resized(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )

    if cache:
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

        test_ds = CacheDataset(
            data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

        return train_loader, test_loader


def show_patient(data: Tuple[DataLoader, DataLoader], slice_number: int = 1, train: bool = True, test: bool = False) -> None:
    '''
    Visualizes a patient's image and label.

    Args:
        data: Tuple of (train_loader, test_loader).
        slice_number: The slice index to visualize.
        train: Whether to visualize from the training set.
        test: Whether to visualize from the test set.
    '''
    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    if train:
        plt.figure("Visualization Train", (10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"image {slice_number}")
        plt.imshow(view_train_patient["image"]
                   [0, 0, :, :, slice_number], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {slice_number}")
        plt.imshow(view_train_patient["label"][0, 0, :, :, slice_number])
        plt.show()

    if test:
        plt.figure("Visualization Test", (10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"image {slice_number}")
        plt.imshow(view_test_patient["image"]
                   [0, 0, :, :, slice_number], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {slice_number}")
        plt.imshow(view_test_patient["label"][0, 0, :, :, slice_number])
        plt.show()


if __name__ == "__main__":
    # data_path is now unused in prepare() as it uses config, but we pass a dummy or None
    # However, prepare() signature still expects data_path.
    # Let's pass config.DATASET_DIR just to satisfy the signature, though it ignores it for finding files.
    # Actually, let's update the signature to default to config.DATASET_DIR or remove it.
    # For now, I'll pass config.DATASET_DIR.
    patient = prepare(config.DATASET_DIR)
    show_patient(patient)
