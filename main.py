"""
main.py — Główna pętla sterowania MICM Blobby Volley.

Uruchomienie:
  python main.py [--cam-p1 0] [--cam-p2 1] [--no-calibrate] [--no-keys]

Argumenty:
  --cam-p1 N      ID kamery dla gracza 1 (domyślnie 0)
  --cam-p2 N      ID kamery dla gracza 2 (domyślnie 1)
  --no-calibrate  Pomiń kalibrację (użyj zapisanych progów lub domyślnych)
  --no-keys       Tryb demo — nie symuluje klawiszy (do testów)
"""

import argparse
import os
import sys
import time

import cv2

from detection.face_mesh import FaceMeshDetector
from detection.gestures_p1 import ActionP1, GestureDetectorP1
from detection.gestures_p2 import ActionP2, GestureDetectorP2
from ml.classifier import SpecialGestureClassifier
from control.key_controller import KeyController
from ui.overlay import Overlay
from calibration.calibrate import load_thresholds, apply_thresholds


def parse_args():
    parser = argparse.ArgumentParser(description="MICM Blobby Volley Controller")
    parser.add_argument("--cam-p1", type=int, default=0, help="ID kamery gracza 1")
    parser.add_argument("--cam-p2", type=int, default=1, help="ID kamery gracza 2")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Pomiń kalibrację")
    parser.add_argument("--no-keys", action="store_true",
                        help="Tryb demo bez symulacji klawiszy")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  MICM Blobby Volley — Face Controller")
    print("=" * 55)
    print(f"  Kamera P1: {args.cam_p1}  |  Kamera P2: {args.cam_p2}")
    print(f"  Symulacja klawiszy: {'NIE (demo)' if args.no_keys else 'TAK'}")
    print("=" * 55)

    # --- Inicjalizacja modułów ---
    detector_p1 = FaceMeshDetector(camera_id=args.cam_p1)
    detector_p2 = FaceMeshDetector(camera_id=args.cam_p2)
    gesture_p1 = GestureDetectorP1()
    gesture_p2 = GestureDetectorP2()
    classifier = SpecialGestureClassifier(threshold=0.75)
    key_ctrl = KeyController()
    overlay = Overlay()

    # --- Otwórz kamery ---
    if not detector_p1.open_camera():
        print(f"BŁĄD: Nie można otworzyć kamery P1 ({args.cam_p1})")
        print("Spróbuj --cam-p1 z innym numerem.")
        sys.exit(1)

    # Kamera P2 — fallback: używamy tej samej kamery jeśli druga niedostępna
    p2_same_cam = False
    if not detector_p2.open_camera():
        print(f"UWAGA: Kamera P2 ({args.cam_p2}) niedostępna. Używam kamery P1 dla obu graczy.")
        detector_p2 = detector_p1
        p2_same_cam = True

    # --- Kalibracja ---
    if not args.no_calibrate:
        thresholds = load_thresholds()
        if thresholds:
            print("[Main] Wczytano istniejące progi kalibracji.")
            apply_thresholds(thresholds, gesture_p1, gesture_p2)
        else:
            print("[Main] Brak zapisanych progów — uruchamiam kalibrację...")
            from calibration.calibrate import run_calibration
            run_calibration(gesture_p1, gesture_p2,
                            cam_p1=args.cam_p1, cam_p2=args.cam_p2)
    else:
        print("[Main] Kalibracja pominięta — używam domyślnych progów.")

    print("\n[Main] Start pętli głównej. [Q] w oknie overlay = wyjście.\n")

    # --- Pętla główna ---
    running = True
    frame_count = 0
    loop_start = time.monotonic()

    try:
        while running:
            t0 = time.monotonic()

            # Odczyt klatek
            ret1, frame1 = detector_p1.read_frame()
            if not p2_same_cam:
                ret2, frame2 = detector_p2.read_frame()
            else:
                ret2, frame2 = ret1, (frame1.copy() if ret1 else None)

            if not ret1:
                print("[Main] Utracono połączenie z kamerą P1.")
                break

            # --- Przetwarzanie P1 ---
            lm_list1, results1 = detector_p1.process(frame1)
            lm1 = lm_list1[0] if lm_list1 else {}
            action_p1 = gesture_p1.update(lm1)

            # --- Przetwarzanie P2 ---
            lm_list2, results2 = detector_p2.process(frame2 if frame2 is not None else frame1)
            lm2 = lm_list2[0] if lm_list2 else {}
            features2 = gesture_p2.get_features(lm2) if lm2 else None

            # Klasyfikacja ML
            ml_special = False
            ml_confidence = 0.0
            if features2 is not None and classifier.is_ready():
                vec = gesture_p2.to_feature_vector(features2)
                ml_confidence = classifier.confidence(vec)
                ml_special = classifier.predict(vec)

            action_p2 = gesture_p2.update(lm2, ml_special=ml_special)

            # --- Symulacja klawiszy ---
            if not args.no_keys:
                key_ctrl.update(action_p1, action_p2)

            # --- Overlay ---
            key_state = key_ctrl.get_state()
            running = overlay.render(
                frame1, frame2,
                action_p1, action_p2,
                key_state, ml_confidence
            )

            frame_count += 1

            # Pomiar latencji (co 100 klatek)
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - loop_start
                fps = frame_count / elapsed
                latency_ms = (time.monotonic() - t0) * 1000
                print(f"[Main] FPS: {fps:.1f} | Latencja klatki: {latency_ms:.1f}ms")

    except KeyboardInterrupt:
        print("\n[Main] Przerwano przez użytkownika.")

    finally:
        print("[Main] Zamykanie...")
        key_ctrl.release_all()
        detector_p1.release()
        if not p2_same_cam:
            detector_p2.release()
        overlay.destroy()
        cv2.destroyAllWindows()
        print("[Main] Zakończono.")


if __name__ == "__main__":
    main()
