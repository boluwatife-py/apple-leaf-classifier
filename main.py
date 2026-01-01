"""
Main entry point for the Apple Disease Classifier
Usage:
    python main.py train                              - Train a new model
    python main.py predict <image_name>               - Predict disease from an image
    python main.py predict --random                   - Predict on random image from uploads
    python main.py predict --random-dataset           - Predict on random image from dataset
    python main.py predict <image> --confidence       - Show confidence score
    python main.py predict <image> --all-probs        - Show all class probabilities
    python main.py predict <image> --show-image       - Display the image
"""
import os
import sys

# Suppress TensorFlow warnings before importing - MUST be before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA to suppress GPU warnings

# Redirect stderr to suppress C++ level errors
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import argparse
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from data import load_dataset, filter_apple_classes, get_label_names, prepare_dataset
from models import create_model, compile_model, load_trained_model, train_model, save_model, save_training_history
from utils import predict_image, find_image_in_uploads, display_prediction_results, get_random_image_from_uploads, get_random_image_from_dataset
from config.config import MODEL_SAVE_PATH, UPLOADS_DIR, LEARNING_RATE, SHOW_IMAGE_BY_DEFAULT

# Restore stderr after imports
sys.stderr.close()
sys.stderr = stderr


def train_new_model():
    """Train a new model from scratch"""
    print("\n" + "=" * 60)
    print("TRAINING NEW MODEL")
    print("=" * 60 + "\n")
    
    # Load dataset
    dataset, info = load_dataset()
    label_names = get_label_names(info)
    
    # Filter for apple classes
    train_ds = dataset['train']
    apple_ds = filter_apple_classes(train_ds, label_names)
    
    # Prepare dataset
    apple_ds = prepare_dataset(apple_ds, shuffle=True)
    
    # Create and compile model
    model = create_model(num_classes=len(label_names))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    # Display model summary
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    history = train_model(model, apple_ds)
    
    # Save model and history
    save_model(model)
    save_training_history(history, info)
    
    print("\n✓ Training completed successfully!")
    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")


def predict_disease(image_name=None, use_random=False, use_random_dataset=False, 
                   show_confidence=False, show_all_probs=False, show_image=None):
    """Predict disease from an image"""
    print("\n" + "=" * 60)
    print("APPLE DISEASE PREDICTION")
    print("=" * 60 + "\n")
    
    # Use config default if show_image not explicitly set
    if show_image is None:
        show_image = SHOW_IMAGE_BY_DEFAULT
    
    # Check if model exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ Error: No trained model found at {MODEL_SAVE_PATH}")
        print("Please train a model first using: python main.py train")
        return
    
    # Load model
    model = load_trained_model(MODEL_SAVE_PATH)
    
    # Load label names
    dataset, info = load_dataset()
    label_names = get_label_names(info)
    
    image_path = None
    true_label = None
    
    # Handle random dataset selection
    if use_random_dataset:
        print("Selecting random image from dataset...")
        train_ds = dataset['train']
        apple_ds = filter_apple_classes(train_ds, label_names)
        apple_ds = prepare_dataset(apple_ds, shuffle=True)
        
        image_path, true_label = get_random_image_from_dataset(apple_ds, label_names)
        image_name = "random_from_dataset.png"
        print(f"✓ Selected random image (True label: {true_label})")
    
    # Handle random uploads selection
    elif use_random:
        print("Selecting random image from uploads folder...")
        image_path = get_random_image_from_uploads()
        if image_path is None:
            print(f"❌ Error: No images found in {UPLOADS_DIR}")
            return
        image_name = os.path.basename(image_path)
        print(f"✓ Selected: {image_name}")
    
    # Handle specific image or interactive mode
    else:
        if image_name is None:
            print(f"Upload your image to: {UPLOADS_DIR}")
            image_name = input("Enter the image filename: ").strip()
        
        # Find image
        image_path = find_image_in_uploads(image_name)
        
        if image_path is None:
            print(f"\n❌ Error: Image '{image_name}' not found in {UPLOADS_DIR}")
            print("Please make sure the image is in the uploads folder.")
            return
        
        print(f"✓ Found image: {image_path}")
    
    # Make prediction
    print("\nAnalyzing image...")
    predicted_class, confidence, all_probs = predict_image(model, image_path, label_names)
    
    # Display results
    display_prediction_results(
        image_name, predicted_class, confidence, all_probs, label_names,
        show_confidence=show_confidence,
        show_all_probs=show_all_probs,
        show_image=show_image,
        image_path=image_path
    )
    
    # Show true label if from dataset
    if true_label:
        print(f"True Label: {true_label}")
        if true_label == predicted_class:
            print("✓ Prediction is CORRECT!")
        else:
            print("✗ Prediction is INCORRECT")
        print()


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Apple Disease Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                              Train a new model
  python main.py predict apple_leaf.jpg             Predict disease from specific image
  python main.py predict apple_leaf.jpg --confidence --show-image
                                                    Show confidence and display image
  python main.py predict --random                   Predict on random image from uploads
  python main.py predict --random-dataset           Predict on random dataset image
  python main.py predict apple.jpg --all-probs      Show all class probabilities
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['train', 'predict'],
        help='Command to execute (train or predict)'
    )
    
    parser.add_argument(
        'image',
        nargs='?',
        help='Image filename for prediction (optional)'
    )
    
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use a random image from uploads folder'
    )
    
    parser.add_argument(
        '--random-dataset',
        action='store_true',
        help='Use a random image from the dataset'
    )
    
    parser.add_argument(
        '--confidence',
        action='store_true',
        help='Show confidence score'
    )
    
    parser.add_argument(
        '--all-probs',
        action='store_true',
        help='Show all class probabilities'
    )
    
    parser.add_argument(
        '--show-image',
        action='store_true',
        help='Display the image'
    )
    
    parser.add_argument(
        '--no-image',
        action='store_true',
        help='Do not display the image'
    )
    
    args = parser.parse_args()
    
    # If no command provided, ask user
    if args.command is None:
        print("\nApple Disease Classifier")
        print("=" * 60)
        print("1. Train new model")
        print("2. Predict disease from image")
        print("=" * 60)
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            args.command = 'train'
        elif choice == '2':
            args.command = 'predict'
        else:
            print("Invalid choice. Exiting.")
            return
    
    # Execute command
    if args.command == 'train':
        train_new_model()
    elif args.command == 'predict':
        # Determine show_image setting
        show_image_setting = None
        if args.no_image:
            show_image_setting = False
        elif args.show_image:
            show_image_setting = True
        # If neither flag is set, show_image_setting stays None and will use config default
        
        predict_disease(
            image_name=args.image,
            use_random=args.random,
            use_random_dataset=args.random_dataset,
            show_confidence=args.confidence,
            show_all_probs=args.all_probs,
            show_image=show_image_setting
        )


if __name__ == '__main__':
    main()