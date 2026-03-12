# MelanomaAI — Skin Cancer Detection System

## Final Results
| Metric | Value |
|--------|-------|
| Accuracy | 88.62% |
| AUC-ROC | 0.9439 |
| Sensitivity | 80.73% |
| Specificity | 89.90% |
| F1 Score | 0.6478 |
| FP Caught | 55.1% |
| Fairness Gap | 0.1827 |
| Fairness Improvement | 24% (via CLAHE) |

## Model
- Architecture: EfficientNetB3
- Dataset: HAM10000 (7,818 images)
- Training: Two-phase focal loss (α=0.38, γ=2.0)
- Threshold: 0.50
- GPU: NVIDIA RTX 3050 6GB

## Novel Contribution
Inference-time skin tone detection using ITA angle estimation.
Dark skin images (ITA ≤ 28°) automatically routed through CLAHE preprocessing.
Fairness gap reduced by 24% without retraining.
Risk stratification catches 55.1% of false positives.
Grad-CAM explainability shows exact lesion regions model focused on.

## How to Run

### Terminal Demo
```
python demo.py path/to/image.jpg
```

### Full Evaluation
```
python evaluate_model.py --threshold 0.50
```

### Robustness & Fairness Test
```
python evaluate_robustness.py
```

### Training
```
python auto_train.py
```

### Web Interface
```
streamlit run app.py --server.port 8502
```

## Requirements
```
pip install -r requirements.txt
```

## Project Structure
```
EDI/
├── app.py                  # Streamlit web interface
├── auto_train.py           # Training pipeline
├── evaluate_model.py       # Evaluation + 8 graphs
├── evaluate_robustness.py  # Fairness + skin tone testing
├── demo.py                 # Terminal demo + Grad-CAM
├── risk_engine.py          # Risk scoring + ITA + CLAHE
├── backend/                # Model loading + metrics + PDF
├── frontend/               # UI components + CSS
├── models/                 # Trained weights (.pth)
└── outputs/                # Graphs + metrics JSON
```

## GitHub
https://github.com/satyamrahane/Melanoma-CNN-Detection
