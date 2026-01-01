# Apple Disease Classifier

A deep learning-based image classification system for detecting diseases in apple leaves using TensorFlow and the Plant Village dataset.

## Features

- **Automated Disease Detection**: Classifies apple leaf images into 4 categories
- **CNN Architecture**: Custom convolutional neural network for accurate predictions
- **Modular Design**: Clean, well-organized code structure
- **CLI Interface**: Easy-to-use command-line interface with multiple options
- **Training Logs**: Automatic logging of training history and metrics
- **Flexible Prediction**: Multiple prediction modes including random selection
- **Image Display**: Automatic image display (configurable)

## Disease Classes

The model can identify the following apple leaf conditions:

1. **Apple Scab** - Fungal disease causing dark spots
2. **Black Rot** - Fungal infection with circular lesions
3. **Cedar Apple Rust** - Orange/yellow spots from fungal infection
4. **Healthy** - No disease present

## Project Structure

```
apple_disease_classifier/
│
├── main.py                      # Main entry point with CLI
├── requirements.txt             # Project dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore file
│
├── config/
│   └── config.py               # Configuration settings
│
├── data/
│   ├── __init__.py
│   ├── loader.py               # Dataset loading functions
│   └── preprocessor.py         # Data preprocessing
│
├── models/
│   ├── __init__.py
│   ├── model.py                # Model architecture
│   └── trainer.py              # Training logic
│
├── utils/
│   ├── __init__.py
│   ├── predictor.py            # Prediction utilities
│   └── suppress_warnings.py   # Warning suppression
│
├── uploads/                    # Place your images here
├── saved_models/               # Trained models stored here
└── logs/                       # Training logs
```

## Installation

1. **Clone or download the project**

2. **Create a virtual environment (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Required packages:**
   - TensorFlow >= 2.13.0
   - TensorFlow Datasets >= 4.9.0
   - NumPy >= 1.24.0
   - Matplotlib >= 3.7.0
   - Pillow >= 10.0.0

## Usage

### Training a Model

Train a new model from scratch using the Plant Village dataset:

```bash
python main.py train
```

This will:
- Download the Plant Village dataset automatically
- Filter for apple disease classes
- Train the CNN model for 5 epochs
- Save the trained model to `saved_models/apple_disease_model.h5`
- Log training metrics to `logs/training_log_TIMESTAMP.json`

### Making Predictions

#### Basic prediction (shows only the predicted class)
```bash
python main.py predict apple_leaf.jpg
```

#### Show confidence score
```bash
python main.py predict apple_leaf.jpg --confidence
```

#### Show all class probabilities
```bash
python main.py predict apple_leaf.jpg --all-probs
```

#### Disable image display
```bash
python main.py predict apple_leaf.jpg --no-image
```

#### Random image from uploads folder
```bash
python main.py predict --random
```

#### Random image from dataset (with accuracy check)
```bash
python main.py predict --random-dataset
```

#### Combine multiple flags
```bash
python main.py predict apple_leaf.jpg --confidence --all-probs
python main.py predict --random-dataset --confidence --all-probs
```

#### Interactive mode
```bash
python main.py predict
# You'll be prompted to enter the filename

python main.py
# You'll be presented with a menu to choose train or predict
```

### Command-Line Options

**Commands:**
- `train` - Train a new model
- `predict` - Make predictions on images

**Prediction flags:**
- `--confidence` - Show confidence score for the prediction
- `--all-probs` - Show probabilities for all classes
- `--random` - Use a random image from the uploads folder
- `--random-dataset` - Use a random image from the dataset (shows true label)
- `--show-image` - Force image display (overrides config)
- `--no-image` - Disable image display (overrides config)

### Preparing Images for Prediction

1. Place your apple leaf images in the `uploads/` folder
2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
3. The system will automatically find and process your image

## Model Architecture

```
Input (128x128x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D
    ↓
Conv2D (128 filters, 3x3) + ReLU
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense (128 units) + ReLU
    ↓
Dense (num_classes) + Softmax
```

## Configuration

Edit `config/config.py` to customize:

- `IMG_SIZE`: Input image dimensions (default: 128x128)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 5)
- `LEARNING_RATE`: Optimizer learning rate (default: 0.001)
- `SHOW_IMAGE_BY_DEFAULT`: Whether to display images by default (default: True)

## Output Examples

### Basic Prediction Output
```
============================================================
APPLE DISEASE PREDICTION
============================================================

Loading model from saved_models/apple_disease_model.h5...
Model loaded successfully!
Loading plant_village dataset...
Dataset loaded successfully!
Found image: uploads/apple_leaf.jpg

Analyzing image...

============================================================
PREDICTION RESULTS FOR: apple_leaf.jpg
============================================================

Predicted Class: Apple___healthy

============================================================

Image saved at: uploads/apple_leaf.jpg
   (Cannot display GUI in headless environment)
   You can download and view it locally.
```

### Prediction with Confidence and All Probabilities
```
============================================================
PREDICTION RESULTS FOR: apple_leaf.jpg
============================================================

Predicted Class: Apple___healthy
Confidence: 94.32%

All Probabilities:
------------------------------------------------------------
Apple___healthy                          94.32% ████████████████████████████████████████████████
Apple___Apple_scab                        3.21% █
Apple___Black_rot                         1.89% ▌
Apple___Cedar_apple_rust                  0.58%
============================================================
```

### Random Dataset Prediction
```
Selecting random image from dataset...
Selected random image (True label: Apple___Black_rot)

Analyzing image...

============================================================
PREDICTION RESULTS FOR: random_from_dataset.png
============================================================

Predicted Class: Apple___Black_rot

============================================================

True Label: Apple___Black_rot
Prediction is CORRECT!
```

## Troubleshooting

### Model not found error
```bash
Error: No trained model found
```
**Solution**: Train a model first using `python main.py train`

### Image not found error
```bash
Error: Image 'filename.jpg' not found
```
**Solution**: Make sure your image is in the `uploads/` folder

### Memory issues during training
**Solution**: Reduce `BATCH_SIZE` in `config/config.py`

### Image display not working
If you're in a headless environment (SSH, Codespaces, Docker), image display won't work. The system will show the image path instead. You can:
- Use `--no-image` flag to suppress the message
- Set `SHOW_IMAGE_BY_DEFAULT = False` in `config/config.py`
- Download the image to view it locally

### TensorFlow warnings
All TensorFlow warnings are automatically suppressed. If you see any warnings, they won't affect functionality.

## Development

### Adding New Features

The modular structure makes it easy to extend:

- **New preprocessing**: Edit `data/preprocessor.py`
- **Different model**: Modify `models/model.py`
- **Custom metrics**: Update `models/trainer.py`
- **New output formats**: Enhance `utils/predictor.py`
- **Configuration options**: Add to `config/config.py`

## License

This project uses the Plant Village dataset, which is publicly available for research purposes.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## Support

For issues or questions, please check the troubleshooting section or open an issue on the project repository.

---

**Happy Classifying!**