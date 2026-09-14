"""
Brain Tumor Detection - Prediction
------------------------------------
Loads the trained model and classifies a single MRI image from the
'prediction' folder (or any path you point it at).

Run from VS Code's integrated terminal:
    python predict.py path/to/image.jpg
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "brain_tumor_cnn.h5"
IMAGE_SIZE = (224, 224)

# IMPORTANT: set this to match the class_indices printed by train_model.py
# e.g. if it printed {'no': 0, 'yes': 1}, TUMOR_LABEL_FOR_1 stays "yes"
LABEL_FOR_0 = "no"
LABEL_FOR_1 = "yes"


def predict_image(img_path: str) -> str:
    model = tf.keras.models.load_model(MODEL_PATH)

    test_image = image.load_img(img_path, target_size=IMAGE_SIZE)
    test_image = image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)

    if result[0][0] >= 0.5:
        return LABEL_FOR_1
    else:
        return LABEL_FOR_0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    outcome = predict_image(img_path)

    if outcome == "yes":
        print("Prediction: tumor detected")
    else:
        print("Prediction: no tumor detected")
