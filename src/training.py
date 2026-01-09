"""
Stress Detection Model Training Script
Based on: "Detecting Negative Emotional Stress Based on Facial Expression in Real Time" (Li et al., 2019)

Author: Anees, Uneeb
BTU Cottbus-Senftenberg, Winter 2025-26
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DATA_DIR = "data"  # Path to dataset folder

# Emotion classes
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
STRESS_EMOTIONS = ['anger', 'fear', 'sadness']


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=6):
    """Build CNN model for emotion classification."""
    
    reg = regularizers.l2(0.001)
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_data_generators(data_dir):
    """Create training and validation data generators."""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=5
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def compute_weights(train_generator):
    """Compute class weights with fear reduction."""
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    
    # Reduce fear weight by 20% to prevent over-prediction
    fear_idx = list(train_generator.class_indices.keys()).index('fear')
    class_weights[fear_idx] *= 0.8
    
    return dict(enumerate(class_weights))


def plot_history(history, save_path='images/training_curves.png'):
    """Plot and save training history."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    """Main training function."""
    
    print("=" * 50)
    print("Stress Detection Model Training")
    print("=" * 50)
    
    # Create data generators
    train_gen, val_gen = create_data_generators(DATA_DIR)
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    
    # Build model
    model = build_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Compute class weights
    class_weights = compute_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Plot results
    plot_history(history)
    
    # Save final model
    model.save('models/emotion_model_final.keras')
    print("Training complete! Model saved to models/")


if __name__ == "__main__":
    main()
