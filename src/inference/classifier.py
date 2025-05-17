import joblib
import numpy as np
from tensorflow.keras.models import load_model
from src.config import MODEL_PATH, SCALER_PATH

class LetterClassifier:
    def __init__(self, model_path=None, scaler_path=None):
        self.model = load_model(model_path or MODEL_PATH)
        self.scaler = joblib.load(scaler_path or SCALER_PATH)
    def predict(self, img):
        arr = img.reshape(1, img.shape[0], img.shape[1], 1)
        flat = arr.reshape(1, -1)
        scaled = self.scaler.transform(flat)
        scaled = scaled.reshape(arr.shape)
        probs = self.model.predict(scaled)
        idx = np.argmax(probs, axis=1)[0]
        return chr(idx + ord('A'))