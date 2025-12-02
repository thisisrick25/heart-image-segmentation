import os
import shutil
import tarfile
from monai.apps import download_url
import config


def download_and_extract():
    """
    Downloads the Task02_Heart dataset from the Medical Segmentation Decathlon
    and extracts it to the datasets directory.
    """
    # Ensure datasets directory exists
    datasets_root = os.path.dirname(config.DATASET_DIR)
    if not os.path.exists(datasets_root):
        os.makedirs(datasets_root)

    # Define paths
    tar_file_path = os.path.join(datasets_root, "Task02_Heart.tar")

    # Download
    print(f"Downloading dataset from {config.DATASET_URL}...")
    download_url(url=config.DATASET_URL, filepath=tar_file_path)

    # Extract
    print("Extracting dataset...")
    with tarfile.open(tar_file_path, "r:") as tar:
        tar.extractall(path=datasets_root)

    print(f"Dataset extracted to {config.DATASET_DIR}")

    # Cleanup tar file
    os.remove(tar_file_path)
    print("Cleanup complete.")


if __name__ == "__main__":
    download_and_extract()
