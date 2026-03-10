"""
Melanoma CNN - Final Evaluation Script
Generates: Accuracy, AUC-ROC, Confusion Matrix, Classification Report,
           Multi-threshold analysis, ROC curve plot
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score
)
import json
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data/processed"
MODEL_PATH = "models/melanoma_model_improved.keras"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
def load_model():
    # Try improved first, fall back to final
    for path in [MODEL_PATH, "models/melanoma_final.keras", "models/best_phase1.keras"]:
        if os.path.exists(path):
            print(f"✅ Loading model: {path}")
            return tf.keras.models.load_model(path, compile=False)
    raise FileNotFoundError("❌ No model found in models/ folder. Run train.py first.")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
def load_test_data():
    print("📂 Loading validation data...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )
    class_names = val_ds.class_names
    print(f"   Classes: {class_names}")

    # Extract all labels and predictions
    all_labels = []
    all_preds = []

    model = load_model()

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy().flatten())
        all_preds.extend(preds.flatten())

    return np.array(all_labels), np.array(all_preds), class_names

# ─── MULTI-THRESHOLD ANALYSIS ─────────────────────────────────────────────────
def multi_threshold_analysis(y_true, y_pred_proba):
    print("\n📊 Multi-Threshold Analysis:")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Sensitivity':>13} {'Specificity':>13} {'F1':>8} {'AUC':>8}")
    print("-" * 70)

    results = []
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    auc = roc_auc_score(y_true, y_pred_proba)

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        sensitivity = tp / (tp + fn + 1e-9)  # Recall for malignant
        specificity = tn / (tn + fp + 1e-9)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{t:>10.2f} {acc:>10.4f} {sensitivity:>13.4f} {specificity:>13.4f} {f1:>8.4f} {auc:>8.4f}")
        results.append({
            "threshold": t, "accuracy": float(acc),
            "sensitivity": float(sensitivity), "specificity": float(specificity),
            "f1": float(f1), "auc": float(auc)
        })

    return results

# ─── GENERATE PLOTS ───────────────────────────────────────────────────────────
def generate_evaluation_plots(y_true, y_pred_proba, class_names):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0D1117')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    dark_bg = '#0D1117'
    card_bg = '#161B22'
    accent = '#58A6FF'
    green = '#3FB950'
    red = '#F85149'
    orange = '#D29922'

    # ── 1. Confusion Matrix ──
    ax1 = fig.add_subplot(gs[0, 0])
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    ax1.set_title('Confusion Matrix', color='white', fontsize=13, pad=10)
    ax1.set_xlabel('Predicted', color='#8B949E')
    ax1.set_ylabel('Actual', color='#8B949E')
    ax1.tick_params(colors='#8B949E')
    ax1.set_facecolor(card_bg)
    fig.axes[0].spines[:].set_color('#30363D')

    # ── 2. ROC Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(card_bg)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    ax2.plot(fpr, tpr, color=accent, lw=2.5, label=f'AUC = {auc_score:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', color='#30363D', lw=1)
    ax2.fill_between(fpr, tpr, alpha=0.1, color=accent)
    ax2.set_title('ROC Curve', color='white', fontsize=13, pad=10)
    ax2.set_xlabel('False Positive Rate', color='#8B949E')
    ax2.set_ylabel('True Positive Rate', color='#8B949E')
    ax2.legend(loc='lower right', facecolor=card_bg, labelcolor='white')
    ax2.tick_params(colors='#8B949E')
    ax2.spines[:].set_color('#30363D')

    # ── 3. Precision-Recall Curve ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(card_bg)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    ax3.plot(recall, precision, color=green, lw=2.5, label=f'AP = {ap:.4f}')
    ax3.fill_between(recall, precision, alpha=0.1, color=green)
    ax3.set_title('Precision-Recall Curve', color='white', fontsize=13, pad=10)
    ax3.set_xlabel('Recall', color='#8B949E')
    ax3.set_ylabel('Precision', color='#8B949E')
    ax3.legend(facecolor=card_bg, labelcolor='white')
    ax3.tick_params(colors='#8B949E')
    ax3.spines[:].set_color('#30363D')

    # ── 4. Prediction Distribution ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(card_bg)
    benign_preds = y_pred_proba[y_true == 0]
    malignant_preds = y_pred_proba[y_true == 1]
    ax4.hist(benign_preds, bins=30, alpha=0.7, color=green, label='Benign', density=True)
    ax4.hist(malignant_preds, bins=30, alpha=0.7, color=red, label='Malignant', density=True)
    ax4.axvline(x=0.5, color=orange, lw=2, linestyle='--', label='Threshold=0.5')
    ax4.set_title('Prediction Distribution', color='white', fontsize=13, pad=10)
    ax4.set_xlabel('Predicted Probability', color='#8B949E')
    ax4.legend(facecolor=card_bg, labelcolor='white')
    ax4.tick_params(colors='#8B949E')
    ax4.spines[:].set_color('#30363D')

    # ── 5. Metrics Summary ──
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_facecolor(card_bg)
    ax5.axis('off')

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    acc = (y_pred == y_true).mean()
    sensitivity = report.get('malignant', report.get(class_names[1], {})).get('recall', 0)
    specificity_val = report.get('benign', report.get(class_names[0], {})).get('recall', 0)
    f1 = report.get('malignant', report.get(class_names[1], {})).get('f1-score', 0)

    summary_text = (
        f"  EVALUATION SUMMARY\n\n"
        f"  Accuracy    :  {acc:.4f}  ({acc*100:.1f}%)\n"
        f"  AUC-ROC     :  {auc_score:.4f}\n"
        f"  Sensitivity :  {sensitivity:.4f}  (malignant recall)\n"
        f"  Specificity :  {specificity_val:.4f}  (benign recall)\n"
        f"  F1-Score    :  {f1:.4f}\n"
        f"  Avg Precision:  {ap:.4f}\n\n"
        f"  Total Samples: {len(y_true)}\n"
        f"  Benign: {int((y_true==0).sum())}  |  Malignant: {int((y_true==1).sum())}"
    )

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=13, verticalalignment='top',
             fontfamily='monospace', color='white',
             bbox=dict(boxstyle='round', facecolor='#21262D', alpha=0.8))

    fig.suptitle('Melanoma Detection CNN — Evaluation Report',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    out_path = f"{OUTPUT_DIR}/evaluation_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=dark_bg)
    print(f"\n📊 Evaluation report saved: {out_path}")
    plt.close()

# ─── SAVE METRICS ─────────────────────────────────────────────────────────────
def save_metrics(y_true, y_pred_proba, class_names, threshold_results):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = (y_pred == y_true).mean()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "classification_report": report,
        "threshold_analysis": threshold_results,
        "total_samples": int(len(y_true)),
        "benign_count": int((y_true == 0).sum()),
        "malignant_count": int((y_true == 1).sum())
    }

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Metrics saved: {OUTPUT_DIR}/metrics.json")

    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"AUC-ROC: {auc:.4f}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MELANOMA CNN — FINAL EVALUATION")
    print("=" * 60)

    y_true, y_pred_proba, class_names = load_test_data()

    threshold_results = multi_threshold_analysis(y_true, y_pred_proba)
    generate_evaluation_plots(y_true, y_pred_proba, class_names)
    save_metrics(y_true, y_pred_proba, class_names, threshold_results)

    print("\n✅ Evaluation complete! Check outputs/ folder.")