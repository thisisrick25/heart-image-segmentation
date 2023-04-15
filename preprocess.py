import os
from glob import glob

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

def prepare(data_path, pixdim=(1.5, 1.5, 1.0), a_min=0, a_max=2000, spatial_size=[128,128, 64], cache=False):
    #[128, 128, 38], [512, 512, 38], (1.5, 1.5, 1.0)

    set_determinism(seed=0)

    train_images = sorted(glob(os.path.join(data_path, 'train/images', '*.nii.gz')))
    train_labels = sorted(glob(os.path.join(data_path, 'train/labels', '*.nii.gz')))

    test_images = sorted(glob(os.path.join(data_path, 'test/images', '*.nii.gz')))
    test_labels = sorted(glob(os.path.join(data_path, 'test/labels', '*.nii.gz')))

    # create dictionary
    train_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
    test_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]

    # print('train images: ', len(train_images))
    # print('train labels: ', len(train_labels))
    # print(train_files)
    # print(test_files)

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
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader


def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    
    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)
    
    if train:
        plt.figure("Visualization Train", (10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"image {SLICE_NUMBER}")
        plt.imshow(view_train_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {SLICE_NUMBER}")
        plt.imshow(view_train_patient["label"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"image {SLICE_NUMBER}")
        plt.imshow(view_test_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {SLICE_NUMBER}")
        plt.imshow(view_test_patient["label"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

if __name__ == "__main__":
    data_path ='C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/data_train_test'
    patient = prepare(data_path)
    show_patient(patient)