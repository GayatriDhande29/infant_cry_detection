import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set paths
DATASET_PATH = r"D:\babies\dataset\cry"
PROCESSED_PATH = "processed_data/"

# Ensure processed directory exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

def extract_features(file_path, save_as_image=True):
    """Extract Mel spectrogram from audio file and save as image."""
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if save_as_image:
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        file_name = os.path.basename(file_path).replace(".wav", ".png")
        plt.savefig(os.path.join(PROCESSED_PATH, file_name), bbox_inches='tight', pad_inches=0)
        plt.close()
    
    return mel_spec_db

# Process all audio files
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        extract_features(os.path.join(DATASET_PATH, file))

print("✅ Data preprocessing complete!")
