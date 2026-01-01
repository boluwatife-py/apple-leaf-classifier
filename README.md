# 🍎 Apple Disease Classifier

A deep learning-based image classification system for detecting diseases in apple leaves using TensorFlow and the Plant Village dataset.

## 📋 Features

- **Automated Disease Detection**: Classifies apple leaf images into 4 categories
- **CNN Architecture**: Custom convolutional neural network for accurate predictions
- **Modular Design**: Clean, well-organized code structure
- **CLI Interface**: Easy-to-use command-line interface
- **Training Logs**: Automatic logging of training history and metrics
- **Flexible Prediction**: Predict from command line or interactive mode

## 🎯 Disease Classes

The model can identify the following apple leaf conditions:

1. **Apple Scab** - Fungal disease causing dark spots
2. **Black Rot** - Fungal infection with circular lesions
3. **Cedar Apple Rust** - Orange/yellow spots from fungal infection
4. **Healthy** - No disease present

## 📁 Project Structure

```
apple_disease_classifier/
│
├── main.py                      # Main entry point with CLI
├── requirements.txt             # Project dependencies
├── README.md                    # This file
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
│   └── predictor.py            # Prediction utilities
│
├── uploads/                    # Place your images here
├── saved_models/               # Trained models stored here
└── logs/                       # Training logs
```

## 🚀 Installation

1. **Clone or download the project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Required packages:**
   - TensorFlow >= 2.13.0
   - TensorFlow Datasets >= 4.9.0
   - NumPy >= 1.24.0
   - Matplotlib >= 3.7.0

## 💻 Usage

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

#### Option 1: Specify image filename
```bash
python main.py predict apple_leaf.jpg
```

#### Option 2: Interactive mode
```bash
python main.py predict
```
You'll be prompted to enter the filename.

#### Option 3: No arguments (interactive menu)
```bash
python main.py
```
Select from a menu of options.

### Preparing Images for Prediction

1. Place your apple leaf images in the `uploads/` folder
2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
3. The system will automatically find and process your image

## 📊 Model Architecture

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

## ⚙️ Configuration

Edit `config/config.py` to customize:

- `IMG_SIZE`: Input image dimensions (default: 128x128)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 5)
- `LEARNING_RATE`: Optimizer learning rate (default: 0.001)

## 📈 Output Examples

### Training Output
```
Loading plant_village dataset...
Dataset loaded successfully!
Filtering for apple classes only...
Filtered dataset to 4 apple classes
Dataset prepared with batch size: 32

Model architecture created
Input shape: (128, 128, 3)
Output classes: 54

Starting training for 5 epochs...
==================================================
Epoch 1/5
...
==================================================
Training completed!

✓ Training completed successfully!
✓ Model saved to: saved_models/apple_disease_model.h5
```

### Prediction Output
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

## 🛠️ Troubleshooting

### Model not found error
```bash
❌ Error: No trained model found
```
**Solution**: Train a model first using `python main.py train`

### Image not found error
```bash
❌ Error: Image 'filename.jpg' not found
```
**Solution**: Make sure your image is in the `uploads/` folder

### Memory issues during training
**Solution**: Reduce `BATCH_SIZE` in `config/config.py`

## 📝 Development

### Adding New Features

The modular structure makes it easy to extend:

- **New preprocessing**: Edit `data/preprocessor.py`
- **Different model**: Modify `models/model.py`
- **Custom metrics**: Update `models/trainer.py`
- **New output formats**: Enhance `utils/predictor.py`

## 📄 License

This project uses the Plant Village dataset, which is publicly available for research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📧 Support

For issues or questions, please check the troubleshooting section or open an issue on the project repository.

---

**Happy Classifying! 🍎🔬**