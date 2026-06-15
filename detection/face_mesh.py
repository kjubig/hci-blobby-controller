"""
face_mesh.py — wrapper MediaPipe FaceMesh.
Zwraca 468 znormalizowanych landmarków twarzy dla jednej kamery.
"""

import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    """Inicjalizuje MediaPipe FaceMesh i przetwarza klatki wideo."""

    # Kluczowe indeksy landmarków MediaPipe FaceMesh
    # Lewe oko (z perspektywy kamery)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    # Prawe oko (z perspektywy kamery)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    # Lewa brew
    LEFT_BROW = [336, 296, 334, 293, 300]
    # Prawa brew
    RIGHT_BROW = [70, 63, 105, 66, 107]
    # Usta (zewnętrzne)
    MOUTH_OUTER = [61, 291, 0, 17, 269, 405, 314, 82, 87, 178, 88, 95]
    # Punkty do estymacji pozy głowy
    HEAD_POSE_POINTS = [1, 9, 57, 130, 287, 359]  # nos, podbródek, rogi ust, rogi oczu

    def __init__(self, camera_id: int = 0, max_faces: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        self.camera_id = camera_id
        self.cap = None

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

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
        """Odczytuje klatkę z kamery. Zwraca (success, frame)."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def process(self, frame: np.ndarray):
        """
        Przetwarza klatkę BGR i zwraca wyniki MediaPipe.
        Zwraca (landmarks_list, results) gdzie landmarks_list to lista
        słowników {idx: (x, y, z)} dla każdej wykrytej twarzy.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        landmarks_list = []
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            for face_lms in results.multi_face_landmarks:
                lm_dict = {}
                for idx, lm in enumerate(face_lms.landmark):
                    lm_dict[idx] = (lm.x * w, lm.y * h, lm.z)
                landmarks_list.append(lm_dict)

        return landmarks_list, results

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Rysuje siatkę landmarków na klatce (do debugowania)."""
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lms,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
        self.face_mesh.close()


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
