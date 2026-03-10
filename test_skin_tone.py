import os
import cv2
from skin_tone_utils import estimate_skin_tone

data_dir = "data/processed/benign"
images = os.listdir(data_dir)[:10]

print(f"{'Image ID':<20} | {'ITA':<8} | {'Category':<15}")
print("-" * 50)

for img_name in images:
    path = os.path.join(data_dir, img_name)
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ita, cat = estimate_skin_tone(image_rgb)
    print(f"{img_name:<20} | {ita:>8.2f} | {cat:<15}")
