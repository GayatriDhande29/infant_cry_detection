import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("models/baby_cry_model.h5")

# Define the class labels (same as used during training)
class_labels = ["discomfort", "sleepy", "hungry", "burping", "fever", "belly pain", "unknown"]

def preprocess_image(image_path, img_size=(128, 128)):
    """
    Load and preprocess the spectrogram image for prediction.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    img = cv2.resize(img, img_size)  # Resize to match training size
    img = img / 255.0  # Normalize
    
    # Ensure correct shape: (batch_size, time_steps, height, width, channels)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension -> (128, 128, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension -> (1, 128, 128, 1)
    img = np.expand_dims(img, axis=1)  # Add time step -> (1, 1, 128, 128, 1)

    print(f"Processed image shape: {img.shape}")  # Debugging line

    return img


def predict_cry_sound(image_path):
    """
    Predict the category of an infant's cry based on the spectrogram image.
    """
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    predicted_class = class_labels[np.argmax(prediction)]  # Get class with highest probability
    confidence = np.max(prediction)  # Get confidence score

    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

    return predicted_class, confidence

# Test with a sample spectrogram
if __name__ == "__main__":
    test_image = r"D:\babies\utils\new_sample2.png"  # Change this to your test spectrogram path
    if os.path.exists(test_image):
        predict_cry_sound(test_image)
    else:
        print(f"Test image not found: {test_image}")
