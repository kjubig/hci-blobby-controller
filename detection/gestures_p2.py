"""
gestures_p2.py — Gracz 2: skok (uniesienie brwi) + aktywacja bonusu (wink lewym okiem).

Akcje:
  IDLE    — neutralna pozycja
  JUMP    — obie brwi uniesione (MAR ust poniżej progu) LUB otwarte usta
  SPECIAL — wykryty przez model ML (wink lewym okiem)
"""

import numpy as np
from collections import deque
from enum import Enum, auto

from detection.face_mesh import (
    FaceMeshDetector,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    brow_height,
)


class ActionP2(Enum):
    IDLE = auto()
    JUMP = auto()
    SPECIAL = auto()   # aktywacja bonusu — klasyfikator ML


# Domyślne progi
DEFAULT_BROW_THRESHOLD = 0.45    # uniesienie brwi (brow_height ratio)
DEFAULT_MAR_THRESHOLD = 0.4      # otwarcie ust
DEFAULT_EAR_OPEN = 0.20          # minimalne EAR dla oka otwartego
SMOOTHING_WINDOW = 4


class GestureDetectorP2:
    """
    Reguły (if/else) dla JUMP — akceptowalne wg wymagań projektu.
    SPECIAL pochodzi z zewnętrznego klasyfikatora ML (ml/classifier.py).
    """

    def __init__(self,
                 brow_threshold: float = DEFAULT_BROW_THRESHOLD,
                 mar_threshold: float = DEFAULT_MAR_THRESHOLD,
                 ear_open: float = DEFAULT_EAR_OPEN,
                 smoothing: int = SMOOTHING_WINDOW):

        self.brow_threshold = brow_threshold
        self.mar_threshold = mar_threshold
        self.ear_open = ear_open

        self._brow_history = deque(maxlen=smoothing)
        self._mar_history = deque(maxlen=smoothing)

    # ------------------------------------------------------------------
    # Obliczenia cech
    # ------------------------------------------------------------------

    def get_features(self, landmarks: dict) -> dict:
        """
        Wyciąga wszystkie cechy z landmarków.
        Zwraca słownik gotowy do ML i do reguł.
        """
        ear_left = eye_aspect_ratio(landmarks, FaceMeshDetector.LEFT_EYE)
        ear_right = eye_aspect_ratio(landmarks, FaceMeshDetector.RIGHT_EYE)
        mar = mouth_aspect_ratio(landmarks, [
            FaceMeshDetector.MOUTH_OUTER[0],   # 61  lewy kąt ust
            FaceMeshDetector.MOUTH_OUTER[1],   # 291 prawy kąt ust
            FaceMeshDetector.MOUTH_OUTER[2],   # 0   góra
            FaceMeshDetector.MOUTH_OUTER[3],   # 17  dół
        ])
        brow_left = brow_height(landmarks,
                                FaceMeshDetector.LEFT_BROW,
                                FaceMeshDetector.LEFT_EYE)
        brow_right = brow_height(landmarks,
                                 FaceMeshDetector.RIGHT_BROW,
                                 FaceMeshDetector.RIGHT_EYE)

        return {
            "ear_left": ear_left,
            "ear_right": ear_right,
            "mar": mar,
            "brow_left": brow_left,
            "brow_right": brow_right,
            # Asymetria oczu — kluczowa cecha dla wink
            "ear_asymmetry": ear_right - ear_left,
        }

    def to_feature_vector(self, features: dict) -> np.ndarray:
        """Konwertuje słownik cech na wektor numpy dla modelu ML."""
        return np.array([
            features["ear_left"],
            features["ear_right"],
            features["mar"],
            features["brow_left"],
            features["brow_right"],
            features["ear_asymmetry"],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Detekcja skoku (reguły)
    # ------------------------------------------------------------------

    def _detect_jump(self, features: dict) -> bool:
        """
        Skok = obie brwi uniesione (oba brow_height > próg)
        LUB otwarcie ust > próg.
        Wymaga obustronności, żeby odróżnić od wink.
        """
        self._brow_history.append(
            (features["brow_left"] + features["brow_right"]) / 2.0
        )
        self._mar_history.append(features["mar"])

        avg_brow = np.mean(self._brow_history)
        avg_mar = np.mean(self._mar_history)

        brow_raised = avg_brow > self.brow_threshold
        mouth_open = avg_mar > self.mar_threshold

        return brow_raised or mouth_open

    # ------------------------------------------------------------------
    # Główna metoda aktualizacji
    # ------------------------------------------------------------------

    def update(self, landmarks: dict, ml_special: bool = False) -> ActionP2:
        """
        Przetwarza landmarki z jednej klatki.
        ml_special = True gdy klasyfikator ML zwrócił klasę 'special'.
        Priorytet: SPECIAL > JUMP > IDLE (special nie jest skaczącym)
        """
        if not landmarks:
            return ActionP2.IDLE

        features = self.get_features(landmarks)

        if ml_special:
            return ActionP2.SPECIAL

        if self._detect_jump(features):
            return ActionP2.JUMP

        return ActionP2.IDLE

    # ------------------------------------------------------------------
    # Kalibracja
    # ------------------------------------------------------------------

    def set_thresholds(self, brow: float = None, mar: float = None):
        if brow is not None:
            self.brow_threshold = brow
        if mar is not None:
            self.mar_threshold = mar
        self._brow_history.clear()
        self._mar_history.clear()
