"""
calibrate.py — kalibracja progów per-user przed meczem.

Proces (ok. 20 sekund):
  1. Neutralna twarz (5s) → baseline EAR, MAR, yaw
  2. Skręt głowy w lewo (3s) → próg LEFT
  3. Skręt głowy w prawo (3s) → próg RIGHT
  4. Uniesienie brwi / otwarcie ust (3s) → próg JUMP

Wynik zapisywany do calibration/thresholds.json
"""

import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.face_mesh import FaceMeshDetector, eye_aspect_ratio, brow_height
from detection.gestures_p1 import GestureDetectorP1
from detection.gestures_p2 import GestureDetectorP2

THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "thresholds.json")
COLLECT_SECONDS = 3
NEUTRAL_SECONDS = 5


def collect_samples(detector: FaceMeshDetector,
                    gesture_p1: GestureDetectorP1,
                    gesture_p2: GestureDetectorP2,
                    message: str,
                    duration: float,
                    camera_id_label: str = "") -> list:
    """
    Zbiera próbki przez `duration` sekund z oknem instrukcji.
    Zwraca listę słowników cech.
    """
    samples = []
    end_time = time.monotonic() + duration

    while time.monotonic() < end_time:
        ret, frame = detector.read_frame()
        if not ret:
            continue

        landmarks_list, _ = detector.process(frame)
        landmarks = landmarks_list[0] if landmarks_list else {}

        if landmarks:
            yaw = gesture_p1.compute_yaw_offset(landmarks)
            features = gesture_p2.get_features(landmarks)
            samples.append({
                "yaw": yaw,
                "brow_left": features["brow_left"],
                "brow_right": features["brow_right"],
                "mar": features["mar"],
            })

        remaining = max(0.0, end_time - time.monotonic())
        cv2.putText(frame, message,
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Pozostalo: {remaining:.1f}s",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        if camera_id_label:
            cv2.putText(frame, camera_id_label,
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        cv2.imshow("Kalibracja", frame)
        cv2.waitKey(1)

    return samples


def calibrate_player(camera_id: int, player_label: str,
                     gesture_p1: GestureDetectorP1,
                     gesture_p2: GestureDetectorP2) -> dict:
    """Przeprowadza pełną kalibrację dla jednego gracza."""
    detector = FaceMeshDetector(camera_id=camera_id)
    if not detector.open_camera():
        print(f"[Kalibracja] Nie można otworzyć kamery {camera_id} dla {player_label}")
        return {}

    print(f"\n[Kalibracja] {player_label} | kamera {camera_id}")

    # Krok 1: Neutralna pozycja
    neutral = collect_samples(detector, gesture_p1, gesture_p2,
                              f"{player_label}: Patrzqj prosto, neutralna twarz",
                              NEUTRAL_SECONDS, player_label)

    # Krok 2: Skręt w lewo (tylko P1)
    left_samples = []
    right_samples = []
    jump_samples = []

    if player_label == "P1":
        left_samples = collect_samples(detector, gesture_p1, gesture_p2,
                                       "P1: Skret glowy W LEWO >>>",
                                       COLLECT_SECONDS, player_label)
        right_samples = collect_samples(detector, gesture_p1, gesture_p2,
                                        "P1: Skret glowy W PRAWO <<<",
                                        COLLECT_SECONDS, player_label)
    else:
        jump_samples = collect_samples(detector, gesture_p1, gesture_p2,
                                       "P2: Unie brwi / otworz usta (SKOK)",
                                       COLLECT_SECONDS, player_label)

    detector.release()

    # --- Obliczenie progów ---
    thresholds = {}

    if neutral:
        yaw_neutral = np.mean([s["yaw"] for s in neutral])
        mar_neutral = np.mean([s["mar"] for s in neutral])
        brow_neutral = np.mean(
            [(s["brow_left"] + s["brow_right"]) / 2 for s in neutral]
        )
        thresholds["yaw_neutral"] = float(yaw_neutral)
        thresholds["mar_neutral"] = float(mar_neutral)
        thresholds["brow_neutral"] = float(brow_neutral)

    if left_samples and neutral:
        yaw_neutral_val = thresholds.get("yaw_neutral", 0)
        yaw_left = np.mean([s["yaw"] for s in left_samples])
        gap_left = abs(yaw_left - yaw_neutral_val)

        if right_samples:
            # Użyj obu stron — symetria głowy rzadko jest idealna
            yaw_right = np.mean([s["yaw"] for s in right_samples])
            gap_right = abs(yaw_right - yaw_neutral_val)
            avg_gap = (gap_left + gap_right) / 2.0
            print(f"[Kalibracja] gap_left={gap_left:.4f}  gap_right={gap_right:.4f}  avg={avg_gap:.4f}")
        else:
            avg_gap = gap_left
            print(f"[Kalibracja] gap_left={gap_left:.4f} (brak prawej strony — używam tylko lewej)")

        # Próg = 50% średniego zakresu ruchu
        thresholds["yaw_threshold"] = float(avg_gap * 0.5)

    if jump_samples and neutral:
        brow_jump = np.mean(
            [(s["brow_left"] + s["brow_right"]) / 2 for s in jump_samples]
        )
        mar_jump = np.mean([s["mar"] for s in jump_samples])
        brow_gap = brow_jump - thresholds.get("brow_neutral", 0)
        mar_gap = mar_jump - thresholds.get("mar_neutral", 0)
        thresholds["brow_threshold"] = float(
            thresholds.get("brow_neutral", 0) + brow_gap * 0.5
        )
        thresholds["mar_threshold"] = float(
            thresholds.get("mar_neutral", 0) + mar_gap * 0.5
        )

    return thresholds


def save_thresholds(data: dict):
    os.makedirs(os.path.dirname(THRESHOLDS_PATH), exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Kalibracja] Progi zapisane → {THRESHOLDS_PATH}")


def load_thresholds() -> dict:
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            return json.load(f)
    return {}


def apply_thresholds(thresholds: dict,
                     gesture_p1: GestureDetectorP1,
                     gesture_p2: GestureDetectorP2):
    """Ustawia wyliczone progi w detektorach gestów."""
    p1 = thresholds.get("p1", {})
    p2 = thresholds.get("p2", {})

    if "yaw_threshold" in p1:
        gesture_p1.set_threshold(p1["yaw_threshold"])
        print(f"[Kalibracja] P1 yaw_threshold = {p1['yaw_threshold']:.4f}")

    if "brow_threshold" in p2 or "mar_threshold" in p2:
        gesture_p2.set_thresholds(
            brow=p2.get("brow_threshold"),
            mar=p2.get("mar_threshold"),
        )
        print(f"[Kalibracja] P2 brow_threshold = {p2.get('brow_threshold', '—')}")
        print(f"[Kalibracja] P2 mar_threshold  = {p2.get('mar_threshold', '—')}")


def run_calibration(gesture_p1: GestureDetectorP1,
                    gesture_p2: GestureDetectorP2,
                    cam_p1: int = 0, cam_p2: int = 1):
    """Uruchamia pełną kalibrację dla obu graczy."""
    print("\n" + "=" * 50)
    print("  KALIBRACJA — MICM Blobby Volley")
    print("=" * 50)

    p1_thresholds = calibrate_player(cam_p1, "P1", gesture_p1, gesture_p2)
    p2_thresholds = calibrate_player(cam_p2, "P2", gesture_p1, gesture_p2)

    all_thresholds = {"p1": p1_thresholds, "p2": p2_thresholds}
    save_thresholds(all_thresholds)
    apply_thresholds(all_thresholds, gesture_p1, gesture_p2)

    cv2.destroyAllWindows()
    print("[Kalibracja] Zakończona pomyślnie.\n")


if __name__ == "__main__":
    from detection.gestures_p1 import GestureDetectorP1
    from detection.gestures_p2 import GestureDetectorP2
    g1 = GestureDetectorP1()
    g2 = GestureDetectorP2()
    run_calibration(g1, g2)
