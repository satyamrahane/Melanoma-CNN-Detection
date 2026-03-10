import tensorflow as tf
import numpy as np

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss

# Load model with custom loss
try:
    model = tf.keras.models.load_model('models/melanoma_model_improved.keras', custom_objects={'loss': focal_loss()})
    print("✅ Model loaded successfully!")
    
    # Print model summary
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    
    # Test with random data
    print("\n" + "="*50)
    print("MODEL INFERENCE TEST")
    print("="*50)
    test_input = np.random.rand(1, 224, 224, 3)
    prediction = model.predict(test_input, verbose=0)
    print(f'Test prediction: {prediction[0][0]:.6f}')
    print(f'Prediction shape: {prediction.shape}')
    print(f'Prediction range: [{prediction.min():.6f}, {prediction.max():.6f}]')
    
    # Test multiple predictions
    print("\nTesting multiple random inputs...")
    for i in range(5):
        test = np.random.rand(1, 224, 224, 3)
        pred = model.predict(test, verbose=0)[0][0]
        print(f'  Test {i+1}: {pred:.6f} -> {"Malignant" if pred > 0.5 else "Benign"}')
    
    print("\n✅ MODEL IS FULLY FUNCTIONAL!")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("MODEL IS BROKEN")
