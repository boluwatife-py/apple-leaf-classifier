"""
Prediction utilities for classifying apple diseases
"""
import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data.preprocessor import preprocess_single_image
from config.config import UPLOADS_DIR


def predict_image(model, image_path, label_names):
    """
    Predict the class of a single image
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        label_names: List of class names
        
    Returns:
        tuple: (predicted_class_name, confidence_score, all_probabilities)
    """
    # Preprocess the image
    processed_image = preprocess_single_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get the predicted class index
    predicted_index = np.argmax(predictions[0])
    
    # Get confidence score
    confidence = predictions[0][predicted_index]
    
    # Get class name
    predicted_class = label_names[predicted_index]
    
    return predicted_class, float(confidence), predictions[0]


def find_image_in_uploads(filename):
    """
    Find an image file in the uploads directory
    
    Args:
        filename: Name of the image file
        
    Returns:
        Full path to the image if found, None otherwise
    """
    # Check if it's already a full path
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    # Check in uploads directory
    image_path = os.path.join(UPLOADS_DIR, filename)
    
    if os.path.exists(image_path):
        return image_path
    
    # Try to find with common image extensions if no extension provided
    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            test_path = os.path.join(UPLOADS_DIR, filename + ext)
            if os.path.exists(test_path):
                return test_path
    
    return None


def display_prediction_results(filename, predicted_class, confidence, all_probs, label_names, 
                              show_confidence=False, show_all_probs=False, show_image=False, image_path=None):
    """
    Display prediction results in a formatted way
    
    Args:
        filename: Name of the predicted image
        predicted_class: Predicted class name
        confidence: Confidence score
        all_probs: All class probabilities
        label_names: List of all class names
        show_confidence: Whether to show confidence score (default: False)
        show_all_probs: Whether to show all probabilities (default: False)
        show_image: Whether to display the image (default: False)
        image_path: Path to the image file
    """
    print("\n" + "=" * 60)
    print(f"PREDICTION RESULTS FOR: {filename}")
    print("=" * 60)
    print(f"\nPredicted Class: {predicted_class}")
    
    if show_confidence:
        print(f"Confidence: {confidence * 100:.2f}%")
    
    if show_all_probs:
        print("\nAll Probabilities:")
        print("-" * 60)
        
        # Sort by probability
        sorted_indices = np.argsort(all_probs)[::-1]
        
        for idx in sorted_indices:
            class_name = label_names[idx]
            prob = all_probs[idx]
            bar = "█" * int(prob * 50)
            print(f"{class_name:40} {prob*100:5.2f}% {bar}")
    
    print("=" * 60 + "\n")
    
    if show_image and image_path:
        display_image(image_path)


def display_image(image_path):
    """
    Display an image using matplotlib
    
    Args:
        image_path: Path to the image file
    """
    try:
        import matplotlib
        
        # Check if we're in a headless environment (like Codespaces)
        import subprocess
        result = subprocess.run(['which', 'display'], capture_output=True)
        has_display = result.returncode == 0
        
        # Also check DISPLAY environment variable
        import os as os_module
        display_env = os_module.environ.get('DISPLAY', '')
        
        if not has_display and not display_env:
            # Headless environment - just show path
            print(f"\n📷 Image saved at: {image_path}")
            print("   (Cannot display GUI in headless environment)")
            print("   You can download and view it locally.")
            return
        
        # Try to display if we have a display
        try:
            matplotlib.use('TkAgg')
        except:
            try:
                matplotlib.use('Qt5Agg')
            except:
                pass
        
        img = plt.imread(image_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(image_path), fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(3)
        print("\n[Image displayed for 3 seconds]")
    except Exception as e:
        print(f"\n📷 Image saved at: {image_path}")
        print(f"   (Could not display: {str(e)[:50]}...)")
        print("   You can download and view it locally.")


def get_random_image_from_uploads():
    """
    Get a random image from the uploads directory
    
    Returns:
        Path to a random image file, or None if no images found
    """
    # List all files in uploads directory
    if not os.path.exists(UPLOADS_DIR):
        return None
    
    files = os.listdir(UPLOADS_DIR)
    
    # Filter for image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        return None
    
    # Select random image
    random_image = random.choice(image_files)
    return os.path.join(UPLOADS_DIR, random_image)


def get_random_image_from_dataset(dataset, label_names):
    """
    Get a random image from the dataset
    
    Args:
        dataset: TensorFlow dataset
        label_names: List of class names
        
    Returns:
        tuple: (image_array, true_label, temp_file_path)
    """
    import tempfile
    from PIL import Image
    
    # Take random batch
    for images, labels in dataset.take(1):
        # Get random index from batch
        batch_size = images.shape[0]
        random_idx = random.randint(0, batch_size - 1)
        
        # Extract single image and label
        image = images[random_idx]
        label = labels[random_idx]
        
        # Convert to numpy and denormalize
        img_array = image.numpy()
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        Image.fromarray(img_array).save(temp_file.name)
        
        true_label = label_names[label.numpy()]
        
        return temp_file.name, true_label