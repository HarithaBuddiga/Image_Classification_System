import sys
import numpy as np
import tensorflow as tf
from PIL import Image


# ==========================================
# 1. CLASS NAMES
# ==========================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ==========================================
# 2. CHECK COMMAND-LINE INPUT
# ==========================================

if len(sys.argv) != 2:
    print("Usage:")
    print("python predict.py <image_path>")
    sys.exit(1)


image_path = sys.argv[1]


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

model_path = "model/cifar10_cnn.keras"

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")


# ==========================================
# 4. LOAD IMAGE
# ==========================================

try:
    image = Image.open(image_path).convert("RGB")

except Exception as e:
    print("\nError loading image:")
    print(e)
    sys.exit(1)


# ==========================================
# 5. PREPROCESS IMAGE
# ==========================================

# CIFAR-10 images are 32 x 32
image = image.resize((32, 32))

image_array = np.array(image)

# Normalize pixel values
image_array = image_array.astype("float32") / 255.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)


# ==========================================
# 6. MAKE PREDICTION
# ==========================================

print("\nMaking prediction...")

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n===================================")
print("       IMAGE CLASSIFICATION")
print("===================================")

print("Image:", image_path)
print("Predicted Class:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print("===================================")