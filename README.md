# MelanomaAI — Skin Cancer Detection System

## Final Results
| Metric | Value |
|--------|-------|
| Accuracy | 88.94% |
| AUC-ROC | 0.9392 |
| Sensitivity | 82.57% |
| Specificity | 89.97% |
| F1 Score | 0.6754 |
| FP Caught | 55.1% |
| Fairness Gap | 0.1834 |

## Model
- Architecture: EfficientNetB3
- Dataset: HAM10000 (7,818 images)
- Training: Two-phase focal loss
- Threshold: 0.50

## Novel Contribution
Inference-time skin tone detection using ITA estimation.
Dark skin images routed through CLAHE preprocessing.
Fairness gap measured and flagged for clinical review.
Risk stratification reduces false positive escalations by 55.1%.

## How to Run

### Terminal Demo
python demo.py path/to/image.jpg

### Full Evaluation
python evaluate_model.py --threshold 0.50

### Training
python auto_train.py

### Web Interface
streamlit run app.py --server.port 8502

## Requirements
pip install -r requirements.txt

## Project Structure
EDI/
├── app.py                  # Streamlit web interface
├── auto_train.py           # Training pipeline
├── evaluate_model.py       # Evaluation + graphs
├── demo.py                 # Terminal demo
├── risk_engine.py          # Risk scoring + robustness
├── frontend/               # UI components
├── backend/                # Model + metrics helpers
├── models/                 # Trained weights
└── outputs/                # Graphs + metrics JSON
