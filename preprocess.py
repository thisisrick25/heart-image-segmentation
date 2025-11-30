import os
from glob import glob
from typing import Tuple, List, Dict, Any

from monai.transforms import(
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
    Prepares the data loaders for training and testing.
    
    Args:
        data_path: Path to the dataset directory containing 'train' and 'test' folders.
        pixdim: Pixel dimensions for spacing.
        a_min: Minimum intensity for scaling.
        a_max: Maximum intensity for scaling.
        spatial_size: Spatial size for resizing (unused currently).
        cache: Whether to use CacheDataset.
        
    Returns:
        A tuple of (train_loader, test_loader).
    '''
    set_determinism(seed=config.SEED)

    train_images = sorted(glob(os.path.join(data_path, 'train', 'images', '*.nii.gz')))
    train_labels = sorted(glob(os.path.join(data_path, 'train', 'labels', '*.nii.gz')))

    test_images = sorted(glob(os.path.join(data_path, 'test', 'images', '*.nii.gz')))
    test_labels = sorted(glob(os.path.join(data_path, 'test', 'labels', '*.nii.gz')))

    # create dictionary
    train_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
    test_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Resized(keys=["image", "label"], spatial_size=spatial_size),   
            ToTensord(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Resized(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE)

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
        plt.imshow(view_train_patient["image"][0, 0, :, :, slice_number], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {slice_number}")
        plt.imshow(view_train_patient["label"][0, 0, :, :, slice_number])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"image {slice_number}")
        plt.imshow(view_test_patient["image"][0, 0, :, :, slice_number], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {slice_number}")
        plt.imshow(view_test_patient["label"][0, 0, :, :, slice_number])
        plt.show()

if __name__ == "__main__":
    data_path = config.DATA_TRAIN_TEST_PATH
    patient = prepare(data_path)
    show_patient(patient)