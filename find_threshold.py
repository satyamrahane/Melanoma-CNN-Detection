import json, torch, numpy as np
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, indices=None):
        self.base = ImageFolder(root)
        self.transform = transform
        self.indices = indices if indices is not None else list(range(len(self.base)))
        self.classes = self.base.classes
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

base_ds = ImageFolder("data/processed")
n = len(base_ds)
torch.manual_seed(42)
perm = torch.randperm(n).tolist()
val_idx = perm[int(0.8*n):]
val_ds = SkinDataset("data/processed", transform=val_transform, indices=val_idx)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

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
ckpt = torch.load("models/melanoma_final.pth", map_location=device)
base.load_state_dict(ckpt["model_state_dict"])
model = base.to(device)
model.eval()

print("Running predictions on validation set...")
all_l, all_p = [], []
with torch.no_grad():
    for imgs, labs in val_loader:
        imgs = imgs.to(device)
        preds = model(imgs).squeeze()
        all_l.extend(labs.numpy())
        all_p.extend(preds.cpu().numpy() if preds.dim()>0 else [preds.cpu().item()])

y_true = np.array(all_l)
y_prob = np.array(all_p)

print("\n=== THRESHOLD ANALYSIS ===")
print(f"{'Threshold':>10} | {'Accuracy':>8} | {'Sensitivity':>11} | {'Specificity':>11} | {'F1':>6}")
print("-"*60)
best_thresh = 0.5
best_f1 = 0
for t in np.arange(0.20, 0.70, 0.05):
    y_pred = (y_prob >= t).astype(int)
    acc = (y_pred==y_true).mean()
    report = classification_report(y_true, y_pred, target_names=val_ds.classes, output_dict=True, zero_division=0)
    classes = list(report.keys())
    mal_key = next((k for k in classes if 'mali' in k.lower()), None)
    ben_key = next((k for k in classes if 'beni' in k.lower()), None)
    sens = report[mal_key]['recall'] if mal_key else 0
    spec = report[ben_key]['recall'] if ben_key else 0
    f1 = report[mal_key]['f1-score'] if mal_key else 0
    marker = " <-- BEST F1" if f1 > best_f1 else ""
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
    print(f"{t:>10.2f} | {acc*100:>7.2f}% | {sens*100:>10.2f}% | {spec*100:>10.2f}% | {f1:>6.4f}{marker}")

print(f"\n Recommended threshold: {best_thresh:.2f}")
print(f"   Use this in app.py for best sensitivity/specificity balance")

with open("outputs/optimal_threshold.json","w") as f:
    json.dump({"optimal_threshold": float(best_thresh), "f1_at_threshold": float(best_f1)}, f, indent=2)
print(" Saved outputs/optimal_threshold.json")
