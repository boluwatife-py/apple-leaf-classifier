"""
Data loading functions for Plant Village dataset
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from config.config import DATASET_NAME, APPLE_CLASSES


def load_dataset():
    """
    Load the Plant Village dataset with info
    
    Returns:
        tuple: (dataset, info) where dataset contains train/test splits
    """
    print(f"Loading {DATASET_NAME} dataset...")
    dataset, info = tfds.load(
        DATASET_NAME,
        with_info=True,
        as_supervised=True
    )
    print("Dataset loaded successfully!")
    return dataset, info


def filter_apple_classes(dataset, label_names):
    """
    Filter dataset to only include apple disease classes
    
    Args:
        dataset: TensorFlow dataset
        label_names: List of all class names from dataset info
        
    Returns:
        Filtered dataset containing only apple images
    """
    def is_apple(image, label):
        apple_indices = [label_names.index(c) for c in APPLE_CLASSES]
        return tf.reduce_any(tf.equal(label, apple_indices))
    
    print("Filtering for apple classes only...")
    filtered_ds = dataset.filter(is_apple)
    print(f"Filtered dataset to {len(APPLE_CLASSES)} apple classes")
    return filtered_ds


def get_label_names(info):
    """
    Extract label names from dataset info
    
    Args:
        info: Dataset info object
        
    Returns:
        list: List of all label names
    """
    return info.features['label'].names