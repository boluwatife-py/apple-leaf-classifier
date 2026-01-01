"""
Data preprocessing functions
"""
import tensorflow as tf
from config.config import IMG_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE


def preprocess_image(image, label):
    """
    Preprocess a single image and label
    
    Args:
        image: Input image tensor
        label: Corresponding label
        
    Returns:
        tuple: (preprocessed_image, label)
    """
    # Resize image to target size
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    return image, label


def prepare_dataset(dataset, shuffle=True):
    """
    Prepare dataset for training by applying preprocessing, shuffling, and batching
    
    Args:
        dataset: TensorFlow dataset
        shuffle: Whether to shuffle the dataset (default: True)
        
    Returns:
        Prepared and batched dataset
    """
    # Apply preprocessing
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    
    # Batch the dataset
    dataset = dataset.batch(BATCH_SIZE)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"Dataset prepared with batch size: {BATCH_SIZE}")
    return dataset


def preprocess_single_image(image_path):
    """
    Preprocess a single image from file path for prediction
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image tensor ready for prediction
    """
    # Read image file
    image = tf.io.read_file(image_path)
    
    # Decode image (supports jpg, png, etc.)
    image = tf.image.decode_image(image, channels=3)
    
    # Resize to model input size
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    image = image / 255.0
    
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    
    return image