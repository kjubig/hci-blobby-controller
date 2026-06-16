"""
face_mesh.py — wrapper MediaPipe FaceLandmarker (Tasks API, mediapipe ≥ 0.10.x).

Nowe API nie ma mp.solutions — używamy mediapipe.tasks.python.vision.FaceLandmarker.
Model face_landmarker.task pobierany automatycznie przy pierwszym uruchomieniu.
"""

import os
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Model FaceLandmarker — pobierany automatycznie jeśli nie istnieje
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _ensure_model():
    """Pobiera plik modelu jeśli nie istnieje lokalnie."""
    if not os.path.exists(MODEL_PATH):
        print(f"[FaceMesh] Pobieranie modelu FaceLandmarker (~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
        print(f"[FaceMesh] Model zapisany → {MODEL_PATH}")


class FaceMeshDetector:
    """Inicjalizuje MediaPipe FaceLandmarker (Tasks API) i przetwarza klatki wideo."""

    # Kluczowe indeksy landmarków — identyczne jak w starym FaceMesh (468 punktów)
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_BROW  = [336, 296, 334, 293, 300]
    RIGHT_BROW = [70, 63, 105, 66, 107]
    MOUTH_OUTER = [61, 291, 0, 17, 269, 405, 314, 82, 87, 178, 88, 95]
    HEAD_POSE_POINTS = [1, 9, 57, 130, 287, 359]

    def __init__(self, camera_id: int = 0, max_faces: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        self.camera_id = camera_id
        self.cap = None
        self._start_ms = time.monotonic() * 1000  # punkt startowy dla timestampów

        _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[FaceMesh] Nie można otworzyć kamery {self.camera_id}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[FaceMesh] Kamera {self.camera_id} otwarta.")
        return True

    def read_frame(self):
        """Odczytuje klatkę z kamery. Zwraca (success, frame_bgr)."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def process(self, frame: np.ndarray):
        """
        Przetwarza klatkę BGR i zwraca (landmarks_list, raw_result).
        landmarks_list — lista słowników {idx: (x_px, y_px, z)} dla każdej twarzy.
        raw_result     — surowy obiekt FaceLandmarkerResult (do rysowania).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video wymaga rosnącego timestampu w ms (rzeczywisty czas)
        ts_ms = int(time.monotonic() * 1000 - self._start_ms) + 1
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        landmarks_list = []
        if result.face_landmarks:
            h, w = frame.shape[:2]
            for face_lms in result.face_landmarks:
                lm_dict = {
                    idx: (lm.x * w, lm.y * h, lm.z)
                    for idx, lm in enumerate(face_lms)
                }
                landmarks_list.append(lm_dict)

        return landmarks_list, result

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Rysuje kluczowe punkty oczu/ust na klatce (do debugowania)."""
        if not results.face_landmarks:
            return frame
        h, w = frame.shape[:2]
        for face_lms in results.face_landmarks:
            for idx in (self.LEFT_EYE + self.RIGHT_EYE +
                        self.LEFT_BROW + self.RIGHT_BROW + self.MOUTH_OUTER):
                lm = face_lms[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
        self._landmarker.close()


def euclidean(p1, p2) -> float:
    """Odległość euklidesowa między dwoma punktami (x, y)."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def eye_aspect_ratio(landmarks: dict, eye_indices: list) -> float:
    """
    Oblicza Eye Aspect Ratio (EAR) dla podanych indeksów landmarków.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Wartość bliska 0 → oko zamknięte, ~0.25-0.35 → oko otwarte.
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks: dict, mouth_indices: list) -> float:
    """
    Oblicza Mouth Aspect Ratio (MAR) — miara otwarcia ust.
    Wyższe wartości → usta bardziej otwarte.
    """
    # pionowe odległości (góra-dół)
    top = landmarks[mouth_indices[2]]
    bottom = landmarks[mouth_indices[3]]
    left = landmarks[mouth_indices[0]]
    right = landmarks[mouth_indices[1]]

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)

    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def brow_height(landmarks: dict, brow_indices: list, eye_indices: list) -> float:
    """
    Zwraca znormalizowaną wysokość brwi względem oka.
    Wyższa wartość → brwi uniesione wyżej.
    """
    brow_center_y = np.mean([landmarks[i][1] for i in brow_indices])
    eye_center_y = np.mean([landmarks[i][1] for i in eye_indices])
    eye_width = euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

    if eye_width < 1e-6:
        return 0.0
    # Brwi są nad oczami (mniejszy y), więc różnica będzie ujemna — negujemy
    return (eye_center_y - brow_center_y) / eye_width
