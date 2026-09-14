"""
Brain Tumor Detection - Model Training
---------------------------------------
Trains a CNN on MRI images to classify brain scans as tumor / no tumor.

Expected folder structure (adjust DATASET_DIR below to match yours):

    Dataset/
        train/
            yes/   <- MRI images WITH a tumor
            no/    <- MRI images WITHOUT a tumor
        test/
            yes/
            no/
        prediction/
            <images to classify after training>

Run from VS Code's integrated terminal:
    python train_model.py
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------------------------------------------------------------------
# 1. CONFIG — edit this path to point at your Dataset folder
# ---------------------------------------------------------------------------
DATASET_DIR = r"./Dataset"          # e.g. r"C:\Users\you\OneDrive\brain-tumor-detection\Dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_OUT = "brain_tumor_cnn.h5"

# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

print("Class label mapping:", train_generator.class_indices)
# This tells you which folder ('yes'/'no') maps to 0 and which to 1 —
# you'll need this in predict.py

# ---------------------------------------------------------------------------
# 3. MODEL ARCHITECTURE
# ---------------------------------------------------------------------------
cnn = Sequential()

cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[224, 224, 3]))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=1, activation="sigmoid"))  # binary output

cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn.summary()

# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------
history = cnn.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
)

# ---------------------------------------------------------------------------
# 5. SAVE THE TRAINED MODEL
# ---------------------------------------------------------------------------
cnn.save(MODEL_OUT)
print(f"\nModel saved to {MODEL_OUT}")
