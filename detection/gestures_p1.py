"""
gestures_p1.py — Gracz 1: sterowanie ruchem (lewo/prawo) przez przechylenie głowy.

Metoda: estymacja kąta Yaw z landmarków nosa i konturów twarzy.
Zwraca enum Action: LEFT / RIGHT / IDLE
"""

import numpy as np
from collections import deque
from enum import Enum, auto

from detection.face_mesh import FaceMeshDetector, euclidean


class ActionP1(Enum):
    IDLE = auto()
    LEFT = auto()
    RIGHT = auto()


# Domyślne progi — kalibracja może je nadpisać
DEFAULT_YAW_THRESHOLD = 0.07   # znormalizowany offset nosa względem środka twarzy
SMOOTHING_WINDOW = 3            # liczba klatek do wygładzania (mniej = szybsza reakcja)


class GestureDetectorP1:
    """
    Wykrywa przechylenie głowy w lewo/prawo na podstawie poziomej pozycji
    końcówki nosa względem środka symetrii twarzy.

    Indeksy użyte:
      - 1   : czubek nosa
      - 234 : lewy kraniec twarzy
      - 454 : prawy kraniec twarzy
      - 10  : środek czoła (punkt referencyjny pionu)
      - 152 : podbródek
    """

    NOSE_TIP = 1
    FACE_LEFT = 234
    FACE_RIGHT = 454
    FOREHEAD = 10
    CHIN = 152

    def __init__(self, yaw_threshold: float = DEFAULT_YAW_THRESHOLD,
                 smoothing: int = SMOOTHING_WINDOW,
                 mirror: bool = False):
        self.yaw_threshold = yaw_threshold
        # mirror=True gdy kamera daje obraz odwrócony (lustrzany, jak selfie).
        # mirror=False (domyślnie) dla surowego obrazu OpenCV z kamery USB/laptopa.
        # Jeśli ruch głowy w lewo porusza blobbem w prawo — ustaw mirror=True.
        self.mirror = mirror
        self._history = deque(maxlen=smoothing)

    def compute_yaw_offset(self, landmarks: dict) -> float:
        """
        Oblicza znormalizowany offset poziomy nosa.
        Wartość > 0  → nos przesunięty w prawo kamery → głowa skręcona w lewo (z pov gracza)
        Wartość < 0  → nos w lewo kamery → głowa w prawo
        Normalizacja przez szerokość twarzy.
        """
        nose = landmarks[self.NOSE_TIP]
        left_cheek = landmarks[self.FACE_LEFT]
        right_cheek = landmarks[self.FACE_RIGHT]

        face_center_x = (left_cheek[0] + right_cheek[0]) / 2.0
        face_width = euclidean(left_cheek, right_cheek)

        if face_width < 1e-6:
            return 0.0

        offset = (nose[0] - face_center_x) / face_width
        return offset

    def update(self, landmarks: dict) -> ActionP1:
        """
        Przetwarza landmarki z jednej klatki i zwraca akcję.
        Używa wygładzania przez okno historii.
        """
        if not landmarks:
            return ActionP1.IDLE

        offset = self.compute_yaw_offset(landmarks)
        self._history.append(offset)
        smoothed = np.mean(self._history)

        if smoothed > self.yaw_threshold:
            # Nos w prawo kamery = głowa w lewo (zakładamy obraz NIE-lustrzany)
            return ActionP1.RIGHT if self.mirror else ActionP1.LEFT
        elif smoothed < -self.yaw_threshold:
            return ActionP1.LEFT if self.mirror else ActionP1.RIGHT
        else:
            return ActionP1.IDLE

    def set_threshold(self, threshold: float):
        """Aktualizuje próg — wywoływane przez moduł kalibracji."""
        self.yaw_threshold = threshold
        self._history.clear()

    def set_mirror(self, mirror: bool):
        """Przełącza tryb lustrzany. Wywołaj jeśli LEFT/RIGHT są zamienione."""
        self.mirror = mirror
        self._history.clear()
