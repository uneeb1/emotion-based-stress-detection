"""
Stress Detection Inference Module
Enhanced detection with smart stress probability analysis

Author: Anees, Uneeb
BTU Cottbus-Senftenberg, Winter 2025-26
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class StressDetector:
    """Stress detection from facial expressions with enhanced inference."""
    
    CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    STRESS_EMOTIONS = ['anger', 'fear', 'sadness']
    STRESS_INDICES = [0, 2, 4]  # anger, fear, sadness
    
    # Thresholds for enhanced detection
    COMBINED_STRESS_THRESHOLD = 0.40
    INDIVIDUAL_STRESS_THRESHOLD = 0.18
    
    def __init__(self, model_path):
        """Initialize detector with trained model."""
        self.model = keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def preprocess(self, image_path):
        """Load and preprocess image for prediction."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Add padding
            pad = int(0.1 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            face = gray[y1:y2, x1:x2]
        else:
            face = gray
        
        # Resize and normalize
        face = cv2.resize(face, (224, 224))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=(0, -1))
        
        return face, img
    
    def predict(self, image_path):
        """
        Predict stress from facial expression.
        
        Returns:
            dict: {
                'emotion': str,
                'confidence': float,
                'is_stressed': bool,
                'stress_type': str,
                'combined_stress': float,
                'probabilities': dict
            }
        """
        # Preprocess
        face, original = self.preprocess(image_path)
        
        # Get predictions
        probs = self.model.predict(face, verbose=0)[0]
        
        # Get top emotion
        emotion_idx = np.argmax(probs)
        emotion = self.CLASSES[emotion_idx]
        confidence = probs[emotion_idx]
        
        # Calculate combined stress probability
        combined_stress = sum(probs[i] for i in self.STRESS_INDICES)
        
        # Enhanced stress detection logic
        is_stressed, stress_type = self._determine_stress(
            emotion_idx, confidence, probs, combined_stress
        )
        
        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'is_stressed': is_stressed,
            'stress_type': stress_type,
            'combined_stress': float(combined_stress),
            'probabilities': {c: float(p) for c, p in zip(self.CLASSES, probs)}
        }
    
    def _determine_stress(self, emotion_idx, confidence, probs, combined_stress):
        """
        Enhanced stress determination with smart detection rules.
        
        Rules:
        1. Direct stress emotion with high confidence → STRESS
        2. Low confidence but high combined stress → STRESS (Enhanced)
        3. Non-stress prediction but any stress emotion > threshold → STRESS (Enhanced)
        4. Otherwise → NO STRESS
        """
        emotion = self.CLASSES[emotion_idx]
        
        # Rule 1: Direct stress emotion
        if emotion in self.STRESS_EMOTIONS:
            return True, "STRESS"
        
        # Rule 2: High combined stress probability
        if combined_stress >= self.COMBINED_STRESS_THRESHOLD:
            return True, "STRESS (Enhanced)"
        
        # Rule 3: Any individual stress emotion above threshold
        for idx in self.STRESS_INDICES:
            if probs[idx] >= self.INDIVIDUAL_STRESS_THRESHOLD:
                return True, "STRESS (Enhanced)"
        
        # Rule 4: No stress indicators
        return False, "NO STRESS"
    
    def visualize(self, image_path, save_path=None):
        """Visualize prediction with emotion probabilities."""
        result = self.predict(image_path)
        _, img = self.preprocess(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image with result
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        color = 'red' if result['is_stressed'] else 'green'
        ax1.set_title(
            f"{result['emotion'].upper()}\n{result['stress_type']}\n"
            f"Confidence: {result['confidence']:.1%}",
            fontsize=14, color=color
        )
        ax1.axis('off')
        
        # Show probability distribution
        probs = result['probabilities']
        colors = ['#e74c3c' if c in self.STRESS_EMOTIONS else '#3498db' 
                  for c in self.CLASSES]
        
        bars = ax2.barh(list(probs.keys()), list(probs.values()), color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Probability')
        ax2.set_title(f"Emotion Probabilities\nCombined Stress: {result['combined_stress']:.1%}")
        
        # Add value labels
        for bar, val in zip(bars, probs.values()):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1%}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return result


def main():
    """Demo usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <image_path>")
        return
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    detector = StressDetector(model_path)
    result = detector.visualize(image_path)
    
    print("\n" + "=" * 40)
    print("PREDICTION RESULT")
    print("=" * 40)
    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Stress Status: {result['stress_type']}")
    print(f"Combined Stress: {result['combined_stress']:.1%}")


if __name__ == "__main__":
    main()
