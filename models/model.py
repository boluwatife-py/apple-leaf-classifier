"""
Model architecture definition
"""
import sys
import os

# Suppress warnings before importing tensorflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.suppress_warnings import SuppressOutput

import tensorflow as tf
from config.config import IMG_SIZE


def create_model(num_classes):
    """
    Create a CNN model for apple disease classification
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = tf.keras.models.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(),

        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    print("Model architecture created")
    print(f"Input shape: ({IMG_SIZE}, {IMG_SIZE}, 3)")
    print(f"Output classes: {num_classes}")
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model compiled successfully")
    return model


def load_trained_model(model_path):
    """
    Load a previously trained model from disk
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded Keras model
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model