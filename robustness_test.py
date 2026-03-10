import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# Load trained model
from robustness_layer import focal_loss
model = tf.keras.models.load_model("models/melanoma_model_improved.keras", custom_objects={'loss': focal_loss()})

IMG_SIZE = (224, 224)
data_dir = "data/processed"

def load_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=False
    )
    return dataset

def add_noise(image):
    noise = np.random.normal(0, 0.1, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def add_blur(image):
    return cv2.GaussianBlur(image, (5,5), 0)

def change_brightness(image):
    return tf.image.adjust_brightness(image, delta=0.2).numpy()

def evaluate_dataset(dataset, transform=None):
    y_true = []
    y_pred = []

    for img, label in dataset:
        image = img.numpy()[0]

        if transform:
            image = transform(image)

        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image, verbose=0)[0][0]
        predicted_label = 1 if prediction > 0.5 else 0

        y_true.append(label.numpy()[0])
        y_pred.append(predicted_label)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return acc, recall

print("Loading dataset...")
dataset = load_dataset()

print("\nOriginal Data Evaluation")
acc, recall = evaluate_dataset(dataset)
print("Accuracy:", acc)
print("Melanoma Recall:", recall)

print("\nWith Noise")
acc_noise, recall_noise = evaluate_dataset(dataset, add_noise)
print("Accuracy:", acc_noise)
print("Melanoma Recall:", recall_noise)

print("\nWith Blur")
acc_blur, recall_blur = evaluate_dataset(dataset, add_blur)
print("Accuracy:", acc_blur)
print("Melanoma Recall:", recall_blur)

print("\nWith Brightness Change")
acc_bright, recall_bright = evaluate_dataset(dataset, change_brightness)
print("Accuracy:", acc_bright)
print("Melanoma Recall:", recall_bright)