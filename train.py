"""
Melanoma CNN Training Pipeline - Optimized for Maximum Accuracy
Uses EfficientNetB3 transfer learning + focal loss + class weighting
Target: 88-93% accuracy, 0.92+ AUC-ROC
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5
DATA_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── FOCAL LOSS ───────────────────────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)
    return loss_fn

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_datasets():
    print("📂 Loading dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    class_names = train_ds.class_names
    print(f"✅ Classes found: {class_names}")

    # Count samples for class weights
    benign_count = len(os.listdir(os.path.join(DATA_DIR, "benign")))
    malignant_count = len(os.listdir(os.path.join(DATA_DIR, "malignant")))
    total = benign_count + malignant_count
    print(f"📊 Benign: {benign_count} | Malignant: {malignant_count} | Total: {total}")

    # Compute class weights to handle imbalance
    labels = [0] * benign_count + [1] * malignant_count
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"⚖️  Class weights: {class_weight}")

    return train_ds, val_ds, class_names, class_weight

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────
def get_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.1),
        layers.GaussianNoise(0.01),
    ], name="augmentation")

# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model():
    print("🏗️  Building EfficientNetB3 model...")

    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Freeze base initially
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = get_augmentation()(inputs)

    # EfficientNet expects [0,255] input (no rescaling needed)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model, base_model

# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train():
    train_ds, val_ds, class_names, class_weight = load_datasets()

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    model, base_model = build_model()

    # ── Phase 1: Train head only ──
    print("\n🚀 Phase 1: Training classification head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    callbacks_phase1 = [
        EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(
            f"{MODEL_DIR}/best_phase1.keras",
            monitor="val_auc", save_best_only=True, mode="max"
        )
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weight,
        callbacks=callbacks_phase1
    )

    # ── Phase 2: Fine-tune top layers ──
    print("\n🔧 Phase 2: Fine-tuning EfficientNet top layers...")
    base_model.trainable = True

    # Freeze bottom 80% of layers, only train top 20%
    fine_tune_at = int(len(base_model.layers) * 0.8)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"   Fine-tuning from layer {fine_tune_at}/{len(base_model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LR),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    callbacks_phase2 = [
        EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-8),
        ModelCheckpoint(
            f"{MODEL_DIR}/melanoma_model_improved.keras",
            monitor="val_auc", save_best_only=True, mode="max"
        )
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks_phase2
    )

    # ── Save final model ──
    model.save(f"{MODEL_DIR}/melanoma_final.keras")
    print(f"\n✅ Model saved to {MODEL_DIR}/melanoma_final.keras")

    # ── Save training history ──
    combined_history = {}
    for key in history1.history:
        combined_history[key] = [float(x.numpy()) if hasattr(x, 'numpy') else float(x) for x in history1.history[key]]
        if key in history2.history:
            combined_history[key] += [float(x.numpy()) if hasattr(x, 'numpy') else float(x) for x in history2.history[key]]

    with open(f"{OUTPUT_DIR}/training_history.json", "w") as f:
        json.dump(combined_history, f)

    # ── Plot training curves ──
    plot_training(combined_history)

    # ── Final evaluation ──
    print("\n📊 Final Evaluation:")
    results = model.evaluate(val_ds)
    metrics = dict(zip(model.metrics_names, results))
    print(f"   Accuracy : {metrics['accuracy']:.4f}")
    print(f"   AUC      : {metrics['auc']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall   : {metrics['recall']:.4f}")

    return model

# ─── PLOT ─────────────────────────────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Results - Melanoma CNN", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.get("accuracy", []), label="Train", color="#2196F3")
    axes[0].plot(history.get("val_accuracy", []), label="Val", color="#FF5722")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(history.get("auc", []), label="Train AUC", color="#4CAF50")
    axes[1].plot(history.get("val_auc", []), label="Val AUC", color="#FF9800")
    axes[1].set_title("AUC-ROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Loss
    axes[2].plot(history.get("loss", []), label="Train Loss", color="#9C27B0")
    axes[2].plot(history.get("val_loss", []), label="Val Loss", color="#E91E63")
    axes[2].set_title("Focal Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150, bbox_inches="tight")
    print(f"📈 Training curves saved to {OUTPUT_DIR}/training_curves.png")
    plt.close()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MELANOMA CNN TRAINING - EfficientNetB3 + Focal Loss")
    print("=" * 60)
    print(f"  TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU available: {'YES - ' + str(len(gpus)) + ' GPU(s)' if gpus else 'NO - using CPU'}")
    print("=" * 60)

    model = train()
    print("\n🎉 Training complete! Run evaluate_final.py next.")