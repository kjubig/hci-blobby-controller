"""
collect_dataset.py — narzędzie do zbierania datasetu dla modelu ML.

Klasy:
  0 = idle    (neutralna twarz)
  1 = special (wink lewym okiem — lewe oko ZAMKNIĘTE, prawe OTWARTE)

Sterowanie:
  [Space]  → zapisz próbkę klasy IDLE
  [Enter]  → zapisz próbkę klasy SPECIAL
  [S]      → pokaż statystyki
  [Q]      → wyjdź i zapisz dataset

Dane zapisywane do: data/dataset.npz
"""

import os
import sys
import cv2
import numpy as np

# Dodaj root projektu do ścieżki
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.face_mesh import FaceMeshDetector
from detection.gestures_p2 import GestureDetectorP2

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.npz")
MIN_SAMPLES_PER_CLASS = 50
CAMERA_ID = 0

CLASS_NAMES = {0: "IDLE", 1: "SPECIAL (wink lewe oko)"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 100, 255)}


def load_existing(path: str):
    """Wczytuje istniejący dataset lub zwraca puste tablice."""
    if os.path.exists(path):
        data = np.load(path)
        return list(data["X"]), list(data["y"])
    return [], []


def save_dataset(path: str, X: list, y: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int32))
    print(f"[Dataset] Zapisano {len(y)} próbek → {path}")


def draw_ui(frame, n_idle, n_special, last_action, features):
    h, w = frame.shape[:2]

    # Tło panelu
    cv2.rectangle(frame, (0, 0), (w, 120), (30, 30, 30), -1)

    # Liczniki
    cv2.putText(frame, f"IDLE    [Space]: {n_idle:3d}/{MIN_SAMPLES_PER_CLASS}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLASS_COLORS[0], 2)
    cv2.putText(frame, f"SPECIAL [Enter]: {n_special:3d}/{MIN_SAMPLES_PER_CLASS}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLASS_COLORS[1], 2)

    # Ostatnia akcja
    cv2.putText(frame, f"Ostatnia: {last_action}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Cechy real-time
    if features:
        info = (f"EAR_L={features['ear_left']:.3f}  "
                f"EAR_R={features['ear_right']:.3f}  "
                f"MAR={features['mar']:.3f}  "
                f"Asym={features['ear_asymmetry']:.3f}")
        cv2.putText(frame, info,
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

    # Instrukcja
    cv2.putText(frame, "[Q] Zakoncz i zapisz    [S] Statystyki",
                (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def main():
    print("=" * 55)
    print("  Kolektor danych — MICM Blobby Volley")
    print("=" * 55)
    print(f"  Kamera: {CAMERA_ID}")
    print(f"  [Space] = IDLE   [Enter] = SPECIAL   [Q] = Wyjście")
    print("=" * 55)

    detector = FaceMeshDetector(camera_id=CAMERA_ID)
    gesture = GestureDetectorP2()

    if not detector.open_camera():
        print("Błąd: nie można otworzyć kamery!")
        return

    X, y = load_existing(DATASET_PATH)
    n_idle = sum(1 for label in y if label == 0)
    n_special = sum(1 for label in y if label == 1)
    print(f"[Dataset] Wczytano istniejący dataset: {n_idle} idle, {n_special} special")

    last_action = "—"
    current_features = None
    current_vector = None

    while True:
        ret, frame = detector.read_frame()
        if not ret:
            break

        landmarks_list, results = detector.process(frame)
        landmarks = landmarks_list[0] if landmarks_list else {}

        if landmarks:
            current_features = gesture.get_features(landmarks)
            current_vector = gesture.to_feature_vector(current_features)

        draw_ui(frame, n_idle, n_special, last_action, current_features)
        cv2.imshow("Kolektor danych", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord(' '):  # Space → IDLE
            if current_vector is not None:
                X.append(current_vector.copy())
                y.append(0)
                n_idle += 1
                last_action = f"Zapisano IDLE #{n_idle}"
                print(f"[+] IDLE #{n_idle}: {current_vector}")

        elif key == 13:  # Enter → SPECIAL
            if current_vector is not None:
                X.append(current_vector.copy())
                y.append(1)
                n_special += 1
                last_action = f"Zapisano SPECIAL #{n_special}"
                print(f"[+] SPECIAL #{n_special}: {current_vector}")

        elif key == ord('s') or key == ord('S'):
            print(f"\n--- Statystyki ---")
            print(f"  IDLE:    {n_idle} próbek")
            print(f"  SPECIAL: {n_special} próbek")
            if MIN_SAMPLES_PER_CLASS - n_idle > 0:
                print(f"  Potrzeba jeszcze {MIN_SAMPLES_PER_CLASS - n_idle} IDLE")
            if MIN_SAMPLES_PER_CLASS - n_special > 0:
                print(f"  Potrzeba jeszcze {MIN_SAMPLES_PER_CLASS - n_special} SPECIAL")
            print()

    detector.release()
    cv2.destroyAllWindows()

    if X:
        save_dataset(DATASET_PATH, X, y)
        print(f"\nPodsumowanie: IDLE={n_idle}, SPECIAL={n_special}")
        if n_idle >= MIN_SAMPLES_PER_CLASS and n_special >= MIN_SAMPLES_PER_CLASS:
            print("Dataset gotowy do treningu! Uruchom: python ml/train_model.py")
        else:
            print(f"UWAGA: potrzeba min. {MIN_SAMPLES_PER_CLASS} próbek/klasę")
    else:
        print("Brak danych do zapisania.")


if __name__ == "__main__":
    main()
