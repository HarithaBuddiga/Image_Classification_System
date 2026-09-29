import os
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. GPU CHECK
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU available. Using GPU.")
else:
    print("GPU not available. Using CPU.")


# ============================================================
# 2. CREATE PROJECT FOLDERS
# ============================================================

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 3. CIFAR-10 DATASET SETUP
# ============================================================

dataset_url = (
    "https://www.cs.toronto.edu/~kriz/"
    "cifar-10-binary.tar.gz"
)

keras_dataset_folder = os.path.expanduser("~/.keras/datasets")
dataset_path = os.path.join(
    keras_dataset_folder,
    "cifar-10-binary.tar.gz"
)

extract_path = os.path.join(
    keras_dataset_folder,
    "cifar-10-batches-bin"
)


# Create .keras/datasets folder if it does not exist
os.makedirs(keras_dataset_folder, exist_ok=True)


# Download dataset if it does not exist
if not os.path.exists(dataset_path):

    print("\nCIFAR-10 dataset not found.")
    print("Downloading CIFAR-10 dataset...")
    print("This may take a few minutes depending on your internet speed.\n")

    try:
        urllib.request.urlretrieve(
            dataset_url,
            dataset_path,
            reporthook=lambda block_num, block_size, total_size:
                print(
                    f"\rDownloading: "
                    f"{min(block_num * block_size * 100 / total_size, 100):.1f}%",
                    end=""
                )
        )

        print("\n\nCIFAR-10 dataset downloaded successfully.")

    except Exception as e:

        print("\n\nDataset download failed.")
        print("Please check your internet connection and try again.")
        print("\nError:", e)

        # Remove incomplete download if it exists
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        raise SystemExit


# ============================================================
# 4. EXTRACT DATASET
# ============================================================

if not os.path.exists(extract_path):

    print("\nExtracting CIFAR-10 dataset...")

    try:
        with tarfile.open(dataset_path, "r:gz") as tar:
            tar.extractall(keras_dataset_folder)

        print("CIFAR-10 dataset extracted successfully.")

    except Exception as e:

        print("\nError while extracting CIFAR-10 dataset.")
        print("Error:", e)
        raise SystemExit

else:

    print("\nCIFAR-10 dataset already extracted.")


# ============================================================
# 5. LOAD CIFAR-10 BINARY DATA
# ============================================================

def load_batch(file_path):

    with open(file_path, "rb") as file:

        data = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )

    data = data.reshape(-1, 3073)

    labels = data[:, 0]

    images = data[:, 1:]

    images = images.reshape(
        -1,
        3,
        32,
        32
    )

    images = images.transpose(
        0,
        2,
        3,
        1
    )

    return images, labels


print("\nLoading CIFAR-10 images...")


train_images_list = []
train_labels_list = []


for i in range(1, 6):

    batch_file = os.path.join(
        extract_path,
        f"data_batch_{i}.bin"
    )

    images, labels = load_batch(batch_file)

    train_images_list.append(images)
    train_labels_list.append(labels)


train_images = np.concatenate(
    train_images_list,
    axis=0
)

train_labels = np.concatenate(
    train_labels_list,
    axis=0
)


# Load test data

test_file = os.path.join(
    extract_path,
    "test_batch.bin"
)

test_images, test_labels = load_batch(
    test_file
)


print("Training images:", train_images.shape)
print("Training labels:", train_labels.shape)

print("Test images:", test_images.shape)
print("Test labels:", test_labels.shape)


# ============================================================
# 6. PREPROCESSING
# ============================================================

train_images = train_images.astype("float32") / 255.0

test_images = test_images.astype("float32") / 255.0


print("\nImage preprocessing completed.")

print(
    "Minimum pixel value:",
    train_images.min()
)

print(
    "Maximum pixel value:",
    train_images.max()
)


# ============================================================
# 7. CLASS NAMES
# ============================================================

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


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.1
        ),

        tf.keras.layers.RandomZoom(
            0.1
        )
    ],
    name="data_augmentation"
)


# ============================================================
# 9. BUILD CNN MODEL
# ============================================================

def build_model():

    model = tf.keras.Sequential(
        [

            tf.keras.layers.Input(
                shape=(32, 32, 3)
            ),

            data_augmentation,

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.5
            ),

            tf.keras.layers.Dense(
                10,
                activation="softmax"
            )
        ]
    )


    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================================
# 10. MODEL PATHS
# ============================================================

model_path = os.path.join(
    "model",
    "cifar10_cnn.keras"
)

history_path = os.path.join(
    "model",
    "training_history.npz"
)


# ============================================================
# 11. LOAD OR TRAIN MODEL
# ============================================================

if os.path.exists(model_path):

    print("\nSaved model found.")

    print("Loading trained CNN model...")

    model = tf.keras.models.load_model(
        model_path
    )

    print("Model loaded successfully.")

    if os.path.exists(history_path):

        saved_history = np.load(
            history_path
        )

        history = {
            "accuracy": saved_history["accuracy"],
            "val_accuracy": saved_history["val_accuracy"],
            "loss": saved_history["loss"],
            "val_loss": saved_history["val_loss"]
        }

    else:

        history = None


else:

    print("\nNo saved model found.")

    print("Building CNN model...")

    model = build_model()

    model.summary()


    print("\nStarting model training...")

    history_object = model.fit(

        train_images,

        train_labels,

        epochs=10,

        batch_size=64,

        validation_split=0.1,

        verbose=1
    )


    # Save model

    model.save(model_path)

    print("\nTrained model saved to:")

    print(model_path)


    # Save training history

    history = history_object.history

    np.savez(

        history_path,

        accuracy=np.array(
            history["accuracy"]
        ),

        val_accuracy=np.array(
            history["val_accuracy"]
        ),

        loss=np.array(
            history["loss"]
        ),

        val_loss=np.array(
            history["val_loss"]
        )
    )


    print("Training history saved.")


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(

    test_images,

    test_labels,

    verbose=1
)


print("\n===================================")
print("        MODEL EVALUATION")
print("===================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("===================================")


# ============================================================
# 13. ACCURACY GRAPH
# ============================================================

if history is not None:

    plt.figure()

    plt.plot(
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/accuracy.png"
    )

    plt.close()


# ============================================================
# 14. LOSS GRAPH
# ============================================================

if history is not None:

    plt.figure()

    plt.plot(
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/loss.png"
    )

    plt.close()


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_images,
    verbose=1
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


cm = confusion_matrix(
    test_labels,
    predicted_classes
)


plt.figure(
    figsize=(10, 10)
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)


disp.plot(
    xticks_rotation=45,
    ax=plt.gca(),
    colorbar=False
)


plt.title(
    "CIFAR-10 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()


# ============================================================
# 16. SAMPLE IMAGES
# ============================================================

plt.figure(
    figsize=(10, 6)
)


for i in range(10):

    plt.subplot(
        2,
        5,
        i + 1
    )

    plt.imshow(
        test_images[i]
    )

    plt.title(
        class_names[
            test_labels[i]
        ]
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "results/sample_images.png"
)

plt.close()


# ============================================================
# 17. PROJECT COMPLETION MESSAGE
# ============================================================

print("\n===================================")
print("       PROJECT COMPLETED")
print("===================================")

print("Model:", model_path)

print("Accuracy graph: results/accuracy.png")

print("Loss graph: results/loss.png")

print(
    "Confusion matrix: results/confusion_matrix.png"
)

print(
    "Sample images: results/sample_images.png"
)

print("===================================")