"""
Configuration settings for the Apple Disease Classifier
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [UPLOADS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# Dataset configuration
DATASET_NAME = 'plant_village'
APPLE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy'
]

# Model save path
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'apple_disease_model.h5')

# Training parameters
SHUFFLE_BUFFER_SIZE = 1000
LEARNING_RATE = 0.001

# Display settings
SHOW_IMAGE_BY_DEFAULT = True  # Set to False to disable image display by default