# Stress Detection from Facial Expressions

A deep learning system for detecting emotional stress from facial expressions using CNN and the KMU-FED dataset.

## Overview

This project is a re-implementation of:

> Li, R., Liu, Z., Zhang, J., et al. (2019). *"Detecting Negative Emotional Stress Based on Facial Expression in Real Time."* IEEE 4th International Conference on Signal and Image Processing (ICSIP), Wuxi, China, pp. 430-434.

The system classifies facial expressions into 6 emotion categories and maps them to stress/non-stress states.

## Results

| Metric | Value |
|--------|-------|
| Emotion Classification Accuracy | 92.77% |
| Stress Detection Accuracy | 95.38% |
| Stress Recall | 98.46% |
| Fear Recall | 100% |

## Requirements

- Google Colab (recommended) or Jupyter Notebook
- GPU: T4 or higher (available free on Google Colab)
- Python 3.10+
- TensorFlow 2.10+

## Setup

### 1. Open in Google Colab

Upload the notebook to Google Colab or open directly from GitHub.

### 2. Enable GPU

Go to: `Runtime` → `Change runtime type` → Select `T4 GPU`

### 3. Dataset

The KMU-FED dataset is included in this repository. After cloning, the dataset is ready to use.

### 4. Change Dataset Path (if needed)

The default path is set for this repository. If you move the dataset, update the path:

```python
# Default path (works after cloning)
DATA_DIR = "KMU-FED"

# If using Google Drive
DATA_DIR = "/content/drive/MyDrive/KMU-FED"
```

Dataset folder structure should be:
```
KMU-FED/
├── anger/
├── disgust/
├── fear/
├── happiness/
├── sadness/
└── surprise/
```

## Project Structure

```
stress-detection/
├── notebook.ipynb          # Main Jupyter notebook
├── src/
│   ├── training.py         # Training script
│   └── inference.py        # Inference with stress detection
├── models/
│   └── model.h5            # Trained model (Keras format)
├── docs/
│   └── report.pdf          # Project report
└── images/
    ├── training_curves.png
    └── confusion_matrix.png
```

## Model

- **Architecture:** Custom CNN with 4 conv blocks
- **Parameters:** ~2.5 million
- **Format:** .h5 (Keras)
- **Input:** 224x224 grayscale images

## Stress Mapping

| Stress | Non-Stress |
|--------|------------|
| Anger | Happiness |
| Fear | Surprise |
| Sadness | Disgust |

## Usage

### Training

```python
# In Jupyter notebook or Colab
# 1. Set your dataset path
DATA_DIR = "/your/path/to/KMU-FED"

# 2. Run all cells
```

### Inference

```python
from src.inference import StressDetector

detector = StressDetector('models/model.h5')
result = detector.predict('path/to/image.jpg')

print(f"Emotion: {result['emotion']}")
print(f"Stressed: {result['is_stressed']}")
```

## Author

**Anees, Uneeb**  
BTU Cottbus-Senftenberg  
Winter Semester 2025-26

## License

This project is for academic purposes.
