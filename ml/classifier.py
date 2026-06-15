"""
classifier.py — wrapper wokół wytrenowanego modelu joblib.
Używany w runtime przez główną pętlę.
"""

import os
import numpy as np
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


class SpecialGestureClassifier:
    """
    Ładuje model z pliku i udostępnia metodę predict().
    Jeśli model nie istnieje (przed treningiem), zawsze zwraca False.
    """

    def __init__(self, threshold: float = 0.75):
        """
        threshold — minimalne prawdopodobieństwo klasy 'special'
                    żeby uznać gest za wykryty (dodatkowy filtr pewności).
        """
        self.threshold = threshold
        self.model = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print(f"[Classifier] Model załadowany z {MODEL_PATH}")
        else:
            print(f"[Classifier] Model nie znaleziony ({MODEL_PATH}).")
            print("             Uruchom: python ml/train_model.py")

    def predict(self, feature_vector: np.ndarray) -> bool:
        """
        Zwraca True jeśli wykryto gest SPECIAL z wystarczającą pewnością.
        """
        if self.model is None:
            return False

        vec = feature_vector.reshape(1, -1)
        try:
            proba = self.model.predict_proba(vec)[0]
            # Klasa 1 = SPECIAL
            confidence = proba[1]
            return bool(confidence >= self.threshold)
        except Exception:
            # Fallback na predict() jeśli model nie wspiera predict_proba
            pred = self.model.predict(vec)[0]
            return bool(pred == 1)

    def confidence(self, feature_vector: np.ndarray) -> float:
        """Zwraca prawdopodobieństwo klasy SPECIAL (0.0–1.0)."""
        if self.model is None:
            return 0.0
        vec = feature_vector.reshape(1, -1)
        try:
            return float(self.model.predict_proba(vec)[0][1])
        except Exception:
            return 1.0 if self.model.predict(vec)[0] == 1 else 0.0

    def is_ready(self) -> bool:
        return self.model is not None
