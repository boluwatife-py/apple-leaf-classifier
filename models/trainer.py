"""
Model training functions
"""
import os
import json
from datetime import datetime
from config.config import EPOCHS, MODEL_SAVE_PATH, LOGS_DIR


def train_model(model, dataset, epochs=EPOCHS):
    """
    Train the model on the given dataset
    
    Args:
        model: Compiled Keras model
        dataset: Prepared training dataset
        epochs: Number of training epochs
        
    Returns:
        Training history object
    """
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 50)
    
    history = model.fit(
        dataset,
        epochs=epochs,
        verbose=1
    )
    
    print("=" * 50)
    print("Training completed!")
    
    return history


def save_model(model, save_path=MODEL_SAVE_PATH):
    """
    Save the trained model to disk
    
    Args:
        model: Trained Keras model
        save_path: Path where to save the model
    """
    print(f"\nSaving model to {save_path}...")
    model.save(save_path)
    print("Model saved successfully!")


def save_training_history(history, dataset_info):
    """
    Save training history and metadata to a JSON file
    
    Args:
        history: Training history object
        dataset_info: Information about the dataset
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"training_log_{timestamp}.json")
    
    # Prepare log data
    log_data = {
        'timestamp': timestamp,
        'epochs': len(history.history['loss']),
        'final_loss': float(history.history['loss'][-1]),
        'final_accuracy': float(history.history['accuracy'][-1]),
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']]
        }
    }
    
    # Save to file
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Training log saved to {log_file}")
    print(f"Final accuracy: {log_data['final_accuracy']:.4f}")