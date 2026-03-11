import sys
import os
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    HAS_ALBU = True
except ImportError:
    HAS_ALBU = False

# Import clinical logic from local risk_engine.py
try:
    from risk_engine import (
        robustness_preprocess, 
        compute_risk_score, 
        estimate_abcd
    )
except ImportError:
    print("Error: risk_engine.py not found in the current directory.")
    sys.exit(1)

def load_model(model_path="models/melanoma_final.pth"):
    """Loads the EfficientNet-B3 model with custom classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reconstruct architecture from utils.py
    base = models.efficientnet_b3(weights=None)
    num_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    if not os.path.exists(model_path):
        # Try relative paths if absolute fails
        alt_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"Error: Model file {model_path} not found.")
            sys.exit(1)
            
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # Handle both full checkpoint and state_dict only
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        base.load_state_dict(ckpt["model_state_dict"])
    else:
        base.load_state_dict(ckpt)
        
    base = base.to(device)
    base.eval()
    return base, device

def get_threshold():
    """Loads optimal threshold from outputs/optimal_threshold.json."""
    paths = ["outputs/optimal_threshold.json", "optimal_threshold.json"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f).get("optimal_threshold", 0.48)
            except:
                pass
    return 0.48

def main():
    if len(sys.argv) < 2:
        print("\n\033[93mUsage: python demo.py path/to/image.jpg\033[0m")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"\033[91mError: Image {img_path} not found.\033[0m")
        sys.exit(1)

    # 1. Initialize
    print("\n[1/3] Loading MelanomaAI Model...")
    model, device = load_model()
    threshold = get_threshold()
    
    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"\033[91mError: Could not read image {img_path}\033[0m")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Preprocessing & Prediction
    print("[2/3] Analyzing Skin Tone & Running Prediction...")
    processed_img, ita, tone_label, reliability, was_enhanced = robustness_preprocess(img_rgb)
    
    if HAS_ALBU:
        transform = Compose([
            Resize(224, 224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        model_img = cv2.resize(processed_img, (224, 224))
        transformed = transform(image=model_img)
        inp = transformed["image"].unsqueeze(0).to(device)
    else:
        # Fallback to manual numpy/cv2 normalization if Albumentations is missing
        model_img = cv2.resize(processed_img, (224, 224)) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        model_img = (model_img - mean) / std
        inp = torch.from_numpy(model_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob = float(model(inp).squeeze().cpu().item())
    
    # 3. Clinical Scoring
    verdict = "MALIGNANT" if prob >= threshold else "BENIGN"
    risk_data = compute_risk_score(prob, ita, threshold)
    abcd = estimate_abcd(prob)

    # 4. Final Output
    print("[3/3] Generating Results...\n")
    
    # Header
    print("="*60)
    print(" " * 15 + "MELANOMA AI DIAGNOSTIC REPORT")
    print("="*60)
    
    # Core Verdict
    colors = {"MALIGNANT": "\033[91m", "BENIGN": "\033[92m", "END": "\033[0m"}
    v_color = colors.get(verdict, "")
    print(f"VERDICT:      {v_color}{verdict}{colors['END']}")
    print(f"Probability:  {prob*100:.1f}%")
    print(f"Risk Level:   {risk_data['level']}")
    print(f"Risk Score:   {int(risk_data['score'])}/100")
    
    # Technical Details
    print("-" * 60)
    print(f"Skin Tone:    {ita:.1f}° ({tone_label})")
    print(f"CLAHE:        {'ON' if was_enhanced else 'OFF'}")
    
    # ABCD Scores
    print("\nABCD BIO-MARKER ESTIMATION:")
    print(f" - Asymmetry: {abcd['asymmetry']['score']:.3f} [{abcd['asymmetry']['label']}]")
    print(f" - Border:    {abcd['border']['score']:.3f} [{abcd['border']['label']}]")
    print(f" - Color:     {abcd['color']['score']:.3f} [{abcd['color']['label']}]")
    print(f" - Diameter:  {abcd['diameter']['mm']:.1f}mm [{abcd['diameter']['label']}]")
    
    # Global Performance
    print("-" * 60)
    print("SYSTEM PERFORMANCE (GLOBAL METRICS)")
    print("-" * 60)
    
    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
                print(f"Accuracy:     {m.get('accuracy', 0)*100:.2f}%")
                print(f"AUC:          {m.get('auc_roc', 0):.4f}")
                print(f"Sensitivity:  {m.get('sensitivity', 0)*100:.2f}%")
                print(f"Specificity:  {m.get('specificity', 0)*100:.2f}%")
        except:
            print("Error reading metrics.json")
    else:
        print("metrics.json not found in outputs/.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
