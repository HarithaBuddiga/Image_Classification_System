# Image Classification System using CNN

## 📌 Project Overview

This project is an image classification system built using a Convolutional Neural Network (CNN) with TensorFlow and Keras.

The model is trained on the CIFAR-10 dataset to classify images into 10 different categories. The project includes image preprocessing, data augmentation, CNN model training, model evaluation, confusion matrix analysis, model saving, and command-line image prediction.

---

## 🎯 Objective

The main objective of this project is to develop a deep learning model capable of automatically identifying the class of an input image.

The system can classify images into the following categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

---

## 🧠 Deep Learning Model

A Convolutional Neural Network (CNN) is used for image classification.

### CNN Architecture

```text
Input Image (32 × 32 × 3)
        ↓
Data Augmentation
        ↓
Convolutional Layer (32 filters)
        ↓
Max Pooling
        ↓
Convolutional Layer (64 filters)
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense Layer (128 neurons)
        ↓
Dropout
        ↓
Output Layer (10 classes)