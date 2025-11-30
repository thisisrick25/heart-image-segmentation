import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets', 'Task02_Heart')

# Data paths
DATA_TRAIN_TEST_PATH = os.path.join(DATASET_DIR, 'data_train_test')
DICOM_FILES_PATH = os.path.join(DATASET_DIR, 'dicom_files')
DICOM_GROUPS_PATH = os.path.join(DATASET_DIR, 'dicom_groups')
NIFTI_FILES_PATH = os.path.join(DATASET_DIR, 'nifti_files')
MODEL_RESULT_PATH = os.path.join(DATASET_DIR, 'result')

# Subdirectories
IMAGES_DIR = 'images'
LABELS_DIR = 'labels'

# Training configuration
SEED = 0
BATCH_SIZE = 1
MAX_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
TEST_INTERVAL = 1

# Model configuration
SPATIAL_SIZE = [128, 128, 64]
PIXDIM = (1.5, 1.5, 1.0)
A_MIN = 0
A_MAX = 2000
