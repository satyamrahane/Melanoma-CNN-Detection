import numpy as np
import cv2
from PIL import Image

def estimate_ita_old(img_rgb):
    try:
        u8 = np.clip(img_rgb,0,255).astype(np.uint8)
        lab = cv2.cvtColor(u8,cv2.COLOR_RGB2LAB)
        h,w = lab.shape[:2]
        m = max(int(min(h,w)*0.1),4)
        corners = [lab[:m,:m],lab[:m,-m:],lab[-m:,:m],lab[-m:,-m:]]
        px = np.vstack([c.reshape(-1,3) for c in corners])
        L = float(np.mean(px[:,0]))*100/255
        b = float(np.mean(px[:,2]))-128
        b = b if abs(b)>1e-6 else 1e-6
        return np.arctan((L-50)/b)*(180/np.pi)
    except Exception as e:
        print(e)
        return 30.0

def estimate_ita_new(img_rgb):
    try:
        u8 = np.clip(img_rgb,0,255).astype(np.uint8)
        lab = cv2.cvtColor(u8,cv2.COLOR_RGB2LAB)
        
        # Mask out black corners
        gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
        mask = gray > 20
        
        valid_lab = lab[mask]
        if len(valid_lab) == 0:
            return 30.0
            
        L_channel = valid_lab[:, 0].astype(float) * 100 / 255.0
        
        threshold_L = np.percentile(L_channel, 60)
        # Avoid glares by capping the top
        threshold_L_upper = np.percentile(L_channel, 95)
        
        skin_pixels = valid_lab[(L_channel > threshold_L) & (L_channel < threshold_L_upper)]
        
        if len(skin_pixels) == 0:
            skin_pixels = valid_lab
            
        L = float(np.mean(skin_pixels[:, 0])) * 100 / 255.0
        b = float(np.mean(skin_pixels[:, 2])) - 128.0
        b = b if abs(b) > 1e-6 else 1e-6
        
        return np.arctan((L - 50) / b) * (180 / np.pi)
    except Exception as e:
        print(e)
        return 30.0

if __name__ == "__main__":
    import os
    # find an image in data/processed
    import glob
    files = glob.glob("data/processed/*/*.jpg")
    if len(files) > 0:
        for f in files[:5]:
            img = np.array(Image.open(f).convert("RGB"))
            print(f"File: {f}")
            print(f"Old ITA: {estimate_ita_old(img):.2f}")
            print(f"New ITA: {estimate_ita_new(img):.2f}")
            print("---")
