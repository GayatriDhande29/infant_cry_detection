import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.cluster import KMeans


data_path = r"D:\babies\utils\processed_data"
# ======================== STEP 1: BUILD AUTOENCODER ======================== #
def build_autoencoder():
    input_img = Input(shape=(128, 128, 1))  
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)  # Extract features
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

# ======================== STEP 2: LOAD & PREPROCESS DATA ======================== #
def load_spectrograms(data_path):
    images = []
    file_names = []

    for filename in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (128, 128)) / 255.0
            images.append(img_resized.reshape(128, 128, 1))
            file_names.append(filename)

    return np.array(images), file_names

# ======================== STEP 3: TRAIN AUTOENCODER ======================== #
def train_autoencoder(autoencoder, images):
    autoencoder.fit(images, images, epochs=50, batch_size=16, verbose=1)

# ======================== STEP 4: EXTRACT FEATURES ======================== #
def extract_features(encoder, images):
    features = encoder.predict(images)
    return features.reshape(len(features), -1)  # Flatten

# ======================== STEP 5: CLUSTERING ======================== #
def cluster_data(features, num_clusters=7):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

# ======================== STEP 6: ASSIGN LABELS ======================== #
def map_labels(cluster_labels):
    cluster_mapping = {
        0: "hungry",
        1: "discomfort",
        2: "sleepy",
        3: "burping",
        4: "fever",
        5: "belly pain",
        6: "unknown"
    }
    return [cluster_mapping[label] for label in cluster_labels]

# ======================== STEP 7: SAVE LABELED DATA ======================== #
def save_labeled_data(file_names, labels, output_file="data/labeled_data.csv"):
    df = pd.DataFrame({"File": file_names, "Label": labels})
    df.to_csv(output_file, index=False)
    print(f"Labeled data saved to {output_file}")

# ======================== MAIN FUNCTION ======================== #
def main():
    # Folder containing spectrogram images
    data_path = r"D:\babies\utils\processed_data"
    # Build autoencoder
    autoencoder, encoder = build_autoencoder()

    # Load and preprocess data
    images, file_names = load_spectrograms(data_path)

    # Train autoencoder
    train_autoencoder(autoencoder, images)

    # Extract features
    features = extract_features(encoder, images)

    # Cluster the features
    cluster_labels = cluster_data(features)

    # Map cluster labels to human-readable labels
    labeled_data = map_labels(cluster_labels)

    # Save labeled data to CSV
    save_labeled_data(file_names, labeled_data)

if __name__ == "__main__":
    main()
