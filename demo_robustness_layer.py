import os
import cv2
import numpy as np
from robustness_layer import MelanomaRobustnessLayer

# Load the robustness layer
model_path = "models/melanoma_model_improved.keras"
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    exit()

rlayer = MelanomaRobustnessLayer(model_path)

# Test on a few images from processed benign/malignant
test_images = []
benign_dir = "data/processed/benign"
malignant_dir = "data/processed/malignant"

if os.path.exists(benign_dir):
    test_images.extend([os.path.join(benign_dir, f) for f in os.listdir(benign_dir)[:3]])
if os.path.exists(malignant_dir):
    test_images.extend([os.path.join(malignant_dir, f) for f in os.listdir(malignant_dir)[:3]])

print("-" * 80)
print(f"{'Image':<40} | {'Skin Tone':<15} | {'Routing':<25} | {'Label'}")
print("-" * 80)

for img_path in test_images:
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = rlayer.predict(image_rgb)
    
    img_name = os.path.basename(img_path)
    print(f"{img_name:<40} | {result['skin_tone_category']:<15} | {result['routing']:<25} | {result['label']}")
