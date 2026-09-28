import os
import tarfile

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix


# ==========================================
# 1. CHECK DEVICE
# ==========================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU Available:", gpus)
else:
    print("GPU not available. Using CPU.")


# ==========================================
# 2. CREATE PROJECT FOLDERS
# ==========================================

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# 3. CIFAR-10 DATASET PATH
# ==========================================

dataset_path = os.path.expanduser(
    "~/.keras/datasets/cifar-10-binary.tar.gz"
)

extract_path = os.path.expanduser(
    "~/.keras/datasets/cifar-10-batches-bin"
)


# ==========================================
# 4. EXTRACT DATASET
# ==========================================

if not os.path.exists(extract_path):

    print("\nExtracting CIFAR-10 dataset...")

    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall(
            path=os.path.expanduser("~/.keras/datasets")
        )

    print("Dataset extracted successfully.")
else:
    print("\nCIFAR-10 dataset already extracted.")


# ==========================================
# 5. FUNCTION TO LOAD BINARY FILE
# ==========================================

def load_batch(file_path):

    with open(file_path, "rb") as file:

        data = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )

    # Each CIFAR-10 record:
    # 1 byte label
    # 3072 bytes image data

    data = data.reshape(-1, 3073)

    labels = data[:, 0]

    images = data[:, 1:]

    # Convert to:
    # (number of images, 32, 32, 3)

    images = images.reshape(
        -1, 3, 32, 32
    )

    images = images.transpose(
        0, 2, 3, 1
    )

    return images, labels


# ==========================================
# 6. LOAD TRAINING DATA
# ==========================================

x_train_list = []
y_train_list = []

for i in range(1, 6):

    file_path = os.path.join(
        extract_path,
        f"data_batch_{i}.bin"
    )

    images, labels = load_batch(file_path)

    x_train_list.append(images)
    y_train_list.append(labels)


x_train = np.concatenate(x_train_list)
y_train = np.concatenate(y_train_list)


# ==========================================
# 7. LOAD TEST DATA
# ==========================================

test_file = os.path.join(
    extract_path,
    "test_batch.bin"
)

x_test, y_test = load_batch(test_file)


# ==========================================
# 8. DISPLAY DATASET INFORMATION
# ==========================================

print("\nDataset Information")
print("-------------------")

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)


# ==========================================
# 9. PREPROCESS THE IMAGES
# ==========================================

# Convert pixel values from 0-255 to 0-1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nAfter preprocessing")
print("-------------------")

print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())


# ==========================================
# 10. CLASS NAMES
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
# 11. IMAGE AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    )

])

print("\nImage augmentation created successfully.")


# ==========================================
# 12. MODEL PATHS
# ==========================================

model_path = "model/cifar10_cnn.keras"

history_path = "model/training_history.npz"


# ==========================================
# 13. BUILD CNN FUNCTION
# ==========================================

def build_model():

    model = tf.keras.Sequential([

        # Input and augmentation
        tf.keras.layers.Input(
            shape=(32, 32, 3)
        ),

        data_augmentation,

        # First convolution block
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Second convolution block
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Convert feature maps into vector
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        # Reduce overfitting
        tf.keras.layers.Dropout(
            0.5
        ),

        # 10 CIFAR-10 classes
        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================
# 14. LOAD OR TRAIN MODEL
# ==========================================

if os.path.exists(model_path):

    print("\nSaved model found.")
    print("Loading trained CNN model...")

    model = tf.keras.models.load_model(
        model_path
    )

    # Load saved training history
    if os.path.exists(history_path):

        saved_history = np.load(
            history_path
        )

        history_data = {
            key: saved_history[key]
            for key in saved_history.files
        }

    else:

        history_data = None


else:

    print("\nNo saved model found.")
    print("Building CNN model...")

    model = build_model()

    print("\nCNN Model Summary")
    print("-----------------")

    model.summary()

    # ======================================
    # TRAIN MODEL
    # ======================================

    print("\nStarting CNN training...")
    print("------------------------")

    history = model.fit(

        x_train,
        y_train,

        epochs=10,

        batch_size=64,

        validation_split=0.1,

        verbose=1
    )

    history_data = history.history

    # ======================================
    # SAVE MODEL
    # ======================================

    print("\nSaving trained model...")

    model.save(
        model_path
    )

    print(
        "Model saved to:",
        model_path
    )

    # ======================================
    # SAVE TRAINING HISTORY
    # ======================================

    np.savez(
        history_path,
        **history_data
    )

    print(
        "Training history saved to:",
        history_path
    )


# ==========================================
# 15. MODEL SUMMARY
# ==========================================

print("\nModel Information")
print("-----------------")

model.summary()


# ==========================================
# 16. EVALUATE MODEL
# ==========================================

print("\nEvaluating model on test data...")
print("-------------------------------")

test_loss, test_accuracy = model.evaluate(

    x_test,
    y_test,

    verbose=1
)


print("\nTest Results")
print("------------")

print(
    "Test Loss:",
    test_loss
)

print(
    "Test Accuracy:",
    test_accuracy
)


# ==========================================
# 17. PLOT TRAINING ACCURACY
# ==========================================

if history_data is not None:

    plt.figure(figsize=(8, 5))

    plt.plot(
        history_data["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history_data["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "CNN Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/accuracy.png"
    )

    plt.show()


# ==========================================
# 18. PLOT TRAINING LOSS
# ==========================================

if history_data is not None:

    plt.figure(figsize=(8, 5))

    plt.plot(
        history_data["loss"],
        label="Training Loss"
    )

    plt.plot(
        history_data["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "CNN Training and Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/loss.png"
    )

    plt.show()


# ==========================================
# 19. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")
print("------------------------")

y_predictions = model.predict(

    x_test,

    batch_size=64,

    verbose=1
)


# Convert probabilities to class numbers

y_pred_classes = np.argmax(
    y_predictions,
    axis=1
)


# ==========================================
# 20. CONFUSION MATRIX
# ==========================================

print("\nGenerating confusion matrix...")
print("------------------------------")

cm = confusion_matrix(
    y_test,
    y_pred_classes
)


plt.figure(figsize=(10, 8))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "CIFAR-10 Confusion Matrix"
)

plt.colorbar()


plt.xticks(
    np.arange(10),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(10),
    class_names
)


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# Display numbers inside matrix

for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


plt.savefig(
    "results/confusion_matrix.png"
)

plt.show()


# ==========================================
# 21. DISPLAY SAMPLE IMAGES
# ==========================================

plt.figure(figsize=(10, 10))

for i in range(9):

    plt.subplot(
        3,
        3,
        i + 1
    )

    plt.imshow(
        x_train[i]
    )

    plt.title(
        class_names[y_train[i]]
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "results/sample_images.png"
)

plt.show()


# ==========================================
# 22. PROJECT COMPLETE
# ==========================================

print("\n===================================")
print("IMAGE CLASSIFICATION PROJECT DONE")
print("===================================")

print("\nTest Accuracy:", test_accuracy)

print("\nGenerated files:")

print("model/cifar10_cnn.keras")
print("model/training_history.npz")
print("results/accuracy.png")
print("results/loss.png")
print("results/confusion_matrix.png")
print("results/sample_images.png")