# Facial Stress Detection

A deep learning system for detecting emotional stress from facial expressions using the KMU-FED dataset.

## Overview

This project is a re-implementation of the methodology described in:

> Li, R., Liu, Z., Zhang, J., et al. (2019). *"Detecting Negative Emotional Stress Based on Facial Expression in Real Time."* IEEE 4th International Conference on Signal and Image Processing (ICSIP), Wuxi, China, pp. 430-434.

The system classifies facial expressions into six emotion categories and maps them to stress/non-stress states.

## Results

| Metric | Value |
|--------|-------|
| Emotion Classification Accuracy | 92.77% |
| Stress Detection Accuracy | 95.38% |
| Stress Recall | 98.46% |
| Fear Recall | 100% |

## Project Structure

```
facial-stress-detection/
├── src/
│   ├── training.py          # Model training script
│   └── inference.py         # Inference with enhanced detection
├── models/
│   └── model_metadata.json  # Model configuration
├── data/
│   └── README.md            # Dataset information
├── docs/
│   └── report.pdf           # Project report
├── images/
│   ├── training_curves.png  # Training history
│   └── confusion_matrix.png # Classification results
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/facial-stress-detection.git
cd facial-stress-detection
pip install -r requirements.txt
```

## Usage

### Training
```python
# Run in Google Colab or local environment
python src/training.py
```

### Inference
```python
from src.inference import StressDetector

detector = StressDetector('path/to/model.keras')
result = detector.predict('path/to/image.jpg')
print(f"Stress: {result['is_stressed']}, Confidence: {result['confidence']:.1%}")
```

## Stress Mapping

| Stress-Related | Non-Stress |
|----------------|------------|
| Anger | Happiness |
| Fear | Surprise |
| Sadness | Disgust |

## Requirements

- Python 3.10+
- TensorFlow 2.10+
- OpenCV 4.5+
- NumPy, Matplotlib, scikit-learn

## Author

**Anees, Uneeb**  
BTU Cottbus-Senftenberg  
Winter Semester 2025-26

## License

This project is for academic purposes.
