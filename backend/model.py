import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_engine import robustness_preprocess, get_skin_tone, estimate_ita, compute_risk_score, estimate_abcd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
    A.pytorch.ToTensorV2()
])

def build_model():
    base = models.efficientnet_b3(weights=None)
    nf   = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(.5),      nn.Linear(nf, 512),  nn.ReLU(),
        nn.BatchNorm1d(512), nn.Dropout(.3),       nn.Linear(512, 256),
        nn.ReLU(),           nn.Dropout(.2),       nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return base

def load_model():
    model = build_model()
    for d in [".", "models", "..", "../models"]:
        for n in ["melanoma_final.pth", "best_phase2.pth", "best_phase1.pth"]:
            p = os.path.join(d, n)
            if os.path.exists(p):
                try:
                    ck = torch.load(p, map_location=device)
                    model.load_state_dict(ck["model_state_dict"])
                    return model.to(device).eval(), p
                except:
                    continue
    return None, None

def predict(model, img_arr, threshold, use_robustness=True):
    img = cv2.resize(img_arr, (224, 224))
    if use_robustness:
        processed, ita, tone_lbl, reliability, clahe_used = robustness_preprocess(img)
    else:
        processed  = img
        ita        = estimate_ita(img)
        tone_lbl, tone_col, _, reliability = get_skin_tone(ita)
        clahe_used = False
    t    = val_tf(image=processed)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(model(t).squeeze().cpu())
    _, tone_col, _, _ = get_skin_tone(ita)
    label = "MALIGNANT" if prob >= threshold else "BENIGN"
    risk  = compute_risk_score(prob, ita, threshold)
    abcd  = estimate_abcd(prob)
    return {
        "prob":      prob,
        "label":     label,
        "ita":       ita,
        "tone_lbl":  tone_lbl,
        "tone_col":  tone_col,
        "clahe":     clahe_used,
        "reliability": reliability,
        "risk":      risk,
        "abcd":      abcd,
    }
