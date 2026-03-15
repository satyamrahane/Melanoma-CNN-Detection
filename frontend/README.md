# MelanomaAI — Flask Frontend

A dynamic, production-grade web frontend for the Melanoma CNN Detection project.

## Project Structure

```
frontend/
├── app.py                    ← Flask server (entry point)
├── requirements.txt
├── templates/
│   ├── base.html             ← Jinja2 base layout (sidebar, topbar, toast)
│   └── index.html            ← All 3 tab pages
└── static/
    ├── css/
    │   └── main.css          ← Full stylesheet
    └── js/
        ├── main.js           ← Tab routing, toast, sidebar metrics
        ├── diagnosis.js      ← Upload, /api/predict, render results
        ├── charts.js         ← All 8 evaluation charts (canvas)
        └── history.js        ← Session history, sort, export CSV
```

## Setup & Run

```bash
# 1. Go into the frontend folder
cd frontend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py

# 4. Open in browser
http://127.0.0.1:5000
```

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/` | Serves the SPA |
| POST | `/api/predict` | Accepts image → returns prediction JSON |
| GET  | `/api/metrics` | Returns model evaluation metrics |
| GET  | `/api/history` | Returns session scan history |
| POST | `/api/history/clear` | Clears session history |
| POST | `/api/report` | Generates PDF report (requires backend/pdf_report.py) |

## Backend Integration

`app.py` auto-detects your existing backend:

```python
from backend.model import predict          # ← your existing file
from backend.metrics import load_metrics   # ← your existing file
from backend.pdf_report import generate_report  # ← your existing file
```

If these are not found, it runs in **DEMO mode** with simulated predictions.

### Expected `predict()` return format:
```python
{
  "probability": 0.73,       # float 0-1
  "label": "Malignant",      # "Benign" or "Malignant"
  "ita": 22.5,               # ITA angle in degrees
  "tone": "Brown",           # Fitzpatrick tone name
  "clahe": True,             # bool
  "risk": "HIGH",            # LOW/MODERATE/HIGH/CRITICAL
  "risk_score": 73,          # int 0-100
  "abcd": {
    "asymmetry": 0.72,
    "border": 0.65,
    "color": 0.58,
    "diameter": 8.4          # mm
  }
}
```

### Expected `load_metrics()` return format:
```python
{
  "accuracy": 88.94,
  "auc": 0.9392,
  "sensitivity": 82.57,
  "specificity": 89.97,
  "precision": 86.60,
  "f1": 84.50,
  "threshold": 0.50,
  "confusion_matrix": [[341, 28], [37, 172]],
  "fairness": {
    "light_acc": 93.36,
    "dark_acc": 75.19,
    "gap": 0.1834
  }
}
```

## Features

- **Tab 1 — Diagnosis**: Upload image → animated analysis → verdict, risk score, ABCD biomarkers, skin tone + CLAHE status, Grad-CAM heatmap, PDF report download
- **Tab 2 — Evaluation**: 4 KPI cards + 8 canvas charts (ROC, CM, PR, Threshold, Risk, Histogram, Radar, Fairness)
- **Tab 3 — History**: Session KPIs, sortable table, risk score bar chart, CSV export

All connected to Flask via `fetch()` API calls — fully dynamic, no page reloads.
