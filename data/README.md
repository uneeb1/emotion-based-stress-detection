# Dataset

## KMU-FED (Korea Maritime University - Facial Expression Dataset)

Place the KMU-FED dataset in this folder with the following structure:

```
data/
├── anger/
├── disgust/
├── fear/
├── happiness/
├── sadness/
└── surprise/
```

Each folder should contain grayscale facial images for that emotion class.

## Dataset Properties

| Property | Value |
|----------|-------|
| Total Samples | 3,318 (after augmentation) |
| Image Size | 224 × 224 pixels |
| Color | Grayscale |
| Classes | 6 (Anger, Disgust, Fear, Happiness, Sadness, Surprise) |

## Note

The dataset is not included in this repository due to size constraints. Please download it separately and place it in this folder before training.
