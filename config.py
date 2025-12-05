import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths (will be overridden by kagglehub in train.py for local environment)
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')

# Data paths (relative to actual dataset directory)
IMAGES_TR_PATH = 'imagesTr'
LABELS_TR_PATH = 'labelsTr'

# Results directory - shared for local and Kaggle
MODEL_RESULT_PATH = os.path.join(BASE_DIR, 'results')

# Training configuration
SEED = 0
BATCH_SIZE = 1
MAX_EPOCHS = 20
MAX_EPOCHS_LOCAL = 1  # For local testing - just 1 epoch to verify everything works
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
TEST_INTERVAL = 1
TRAIN_RATIO = 0.8  # 80% training, 20% validation

# Model configuration
SPATIAL_SIZE = [128, 128, 64]
PIXDIM = (1.5, 1.5, 1.0)
A_MIN = 0
A_MAX = 2000

# Dataset configuration
# Original dataset source (deprecated - use Kaggle dataset instead)
# DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar"

# Kaggle dataset
# Note: Dataset is inside Task02_Heart subdirectory
KAGGLE_DATASET = "thisisrick25/medical-segmentation-decathlon-heart"
