import cv2
import numpy as np

def estimate_skin_tone(image):
    """
    Estimates the skin tone of an image using the Individual Typology Angle (ITA).
    ITA = arctan((L* - 50) / b*) * (180 / pi)
    
    Args:
        image: RGB image as a numpy array.
        
    Returns:
        ita: The calculated ITA value.
        category: Skin tone category (Very Light, Light, Intermediate, Tan, Brown, Dark).
    """
    # Convert to LAB color space
    # OpenCV expects BGR for cvtColor, but we assume RGB input from tf/numpy
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    # We want to estimate background skin tone, not the lesion.
    # We'll take a sample from the corners (usually skin)
    h, w, _ = lab.shape
    margin = 20
    corners = [
        lab[0:margin, 0:margin],
        lab[0:margin, w-margin:w],
        lab[h-margin:h, 0:margin],
        lab[h-margin:h, w-margin:w]
    ]
    
    # Combine corners
    sample = np.concatenate(corners, axis=0)
    
    # Calculate mean L and b
    L = np.mean(sample[:, :, 0]) * (100.0 / 255.0) # Scale to 0-100
    b = np.mean(sample[:, :, 2]) - 128.0 # Scale to LAB range
    
    # ITA formula
    ita = np.arctan2(L - 50, b) * (180.0 / np.pi)
    
    # Categorize
    if ita > 55:
        category = "Very Light"
    elif ita > 41:
        category = "Light"
    elif ita > 28:
        category = "Intermediate"
    elif ita > 10:
        category = "Tan"
    elif ita > -30:
        category = "Brown"
    else:
        category = "Dark"
        
    return ita, category

def is_highly_pigmented(category):
    """Returns True if the skin tone is 'Brown' or 'Dark'."""
    return category in ["Brown", "Dark"]

if __name__ == "__main__":
    # Test with a dummy image
    dummy_light = np.full((224, 224, 3), [255, 224, 189], dtype=np.uint8) # Light skin approx
    ita, cat = estimate_skin_tone(dummy_light)
    print(f"Light test - ITA: {ita:.2f}, Category: {cat}")
    
    dummy_dark = np.full((224, 224, 3), [80, 40, 20], dtype=np.uint8) # Dark skin approx
    ita, cat = estimate_skin_tone(dummy_dark)
    print(f"Dark test - ITA: {ita:.2f}, Category: {cat}")
