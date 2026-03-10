import tensorflow as tf
import numpy as np
import cv2
from skin_tone_utils import estimate_skin_tone, is_highly_pigmented

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss

class MelanomaRobustnessLayer:
    def __init__(self, model_path):
        """
        Initializes the robustness layer with a trained Keras model.
        """
        print(f"Loading diagnostic model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, custom_objects={'loss': focal_loss()})
        self.IMG_SIZE = (224, 224)

    def specialized_preprocessing(self, image_rgb):
        """
        Applies specialized preprocessing for dark skin tones.
        Uses CLAHE on the L channel of LAB space to enhance contrast.
        """
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Convert to LAB
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        
        # Convert back to RGB
        final_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        
        return final_rgb

    def predict(self, image_rgb):
        """
        Predicts melanoma using the robustness layer logic.
        
        1. Estimate skin tone.
        2. If dark skin, apply specialized preprocessing.
        3. Resize and normalize.
        4. Run model prediction.
        """
        # Ensure image is in uint8 for skin tone estimation and specialized processing
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)

        # 1. Estimate skin tone
        ita, category = estimate_skin_tone(image_rgb)
        
        processed_image = image_rgb.copy()
        routing_flag = "Standard"

        # 2. Check if highly pigmented
        if is_highly_pigmented(category):
            print(f"Detecting dark skin ({category}, ITA: {ita:.2f}). Applying specialized processing...")
            processed_image = self.specialized_preprocessing(image_rgb)
            routing_flag = "Specialized (CLAHE)"
        else:
            print(f"Detecting light/medium skin ({category}, ITA: {ita:.2f}). Using standard pipeline.")

        # 3. Final preparation for the model
        # Resize to model input size
        processed_image = cv2.resize(processed_image, self.IMG_SIZE)
        
        # Normalize (assuming model expects 0-1)
        # Note: The model itself contains layers.Rescaling(1./255), we should not divide by 255 again.
        input_tensor = processed_image.astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 4. Diagnostic Prediction
        prediction = self.model.predict(input_tensor, verbose=0)[0][0]
        
        result = {
            "prediction_score": float(prediction),
            "label": "Malignant" if prediction > 0.5 else "Benign",
            "skin_tone_category": category,
            "ita": float(ita),
            "routing": routing_flag
        }
        
        return result

if __name__ == "__main__":
    # Test with a dummy image if model exists
    model_file = "models/melanoma_model_improved.keras"
    if os.path.exists(model_file):
        layer = MelanomaRobustnessLayer(model_file)
        # Create a dummy "dark" image
        dummy_dark = np.full((300, 300, 3), [60, 30, 10], dtype=np.uint8)
        res = layer.predict(dummy_dark)
        print("Test Result:", res)
    else:
        print("Model file not found. Please train the model first.")
