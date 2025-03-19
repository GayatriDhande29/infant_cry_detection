import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set paths
data_path = r"D:\babies\utils\processed_data"
labels_file = r"D:\babies\utils\data\labeled_data.csv"
model_save_path = "models/baby_cry_model.h5"

# Load labeled data
df = pd.read_csv(labels_file)

# Map class labels to numbers
class_names = df['Label'].unique().tolist()
class_to_idx = {label: idx for idx, label in enumerate(class_names)}
df['Label'] = df['Label'].map(class_to_idx)

# Load spectrogram images
def load_images(df, img_size=(128, 128)):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = os.path.join(data_path, row["File"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, img_size) / 255.0  # Normalize
            images.append(img_resized)
            labels.append(row["Label"])
    return np.array(images), np.array(labels)

# Load images and labels
X, y = load_images(df)

# Reshape for CNN + LSTM input (batch_size, time_steps, height, width, channels)
X = X.reshape(X.shape[0], 1, 128, 128, 1)  # (samples, time_step=1, height, width, channels)

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=len(class_names))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN + LSTM Model
model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1))),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Save model
if not os.path.exists("models"):
    os.makedirs("models")
model.save(model_save_path)

print(f"Model trained and saved at {model_save_path}")
