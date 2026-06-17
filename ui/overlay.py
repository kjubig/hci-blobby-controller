"""
overlay.py — wizualny overlay łączący podgląd obu kamer.

Wyświetla:
  - Podgląd kamer P1 i P2 obok siebie
  - Aktualną akcję każdego gracza (kolorowy baner)
  - Wciśnięte klawisze gry
  - Pewność modelu ML (pasek)
  - Latencję pętli (FPS)
"""

import cv2
import numpy as np
import time

from detection.gestures_p1 import ActionP1
from detection.gestures_p2 import ActionP2

# Kolory (BGR)
COLOR_IDLE = (80, 80, 80)
COLOR_LEFT = (50, 200, 50)
COLOR_RIGHT = (50, 50, 200)
COLOR_JUMP = (200, 200, 50)
COLOR_SPECIAL = (0, 100, 255)
COLOR_KEY_ACTIVE = (0, 230, 0)
COLOR_KEY_INACTIVE = (60, 60, 60)
COLOR_TEXT = (255, 255, 255)

PANEL_H = 160        # wysokość panelu sterowania (px)
CAM_W, CAM_H = 320, 240  # rozmiar każdego podglądu


def _resize(frame, w, h):
    if frame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(frame, (w, h))


def _action_color_p1(action: ActionP1):
    return {
        ActionP1.IDLE: COLOR_IDLE,
        ActionP1.LEFT: COLOR_LEFT,
        ActionP1.RIGHT: COLOR_RIGHT,
    }.get(action, COLOR_IDLE)


def _action_color_p2(action: ActionP2):
    return {
        ActionP2.IDLE: COLOR_IDLE,
        ActionP2.JUMP: COLOR_JUMP,
        ActionP2.SPECIAL: COLOR_SPECIAL,
    }.get(action, COLOR_IDLE)


def _action_label_p1(action: ActionP1) -> str:
    return {
        ActionP1.IDLE: "IDLE",
        ActionP1.LEFT: "<<< LEWO",
        ActionP1.RIGHT: "PRAWO >>>",
    }.get(action, "IDLE")


def _action_label_p2(action: ActionP2) -> str:
    return {
        ActionP2.IDLE: "IDLE",
        ActionP2.JUMP: "SKOK ↑",
        ActionP2.SPECIAL: "★ SPECIAL",
    }.get(action, "IDLE")


class Overlay:
    """Buduje i wyświetla okno overlay."""

    def __init__(self, window_name: str = "MICM Blobby Controller"):
        self.window_name = window_name
        self._fps_times = []
        self._fps = 0.0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, CAM_W * 2, CAM_H + PANEL_H)

    def render(self,
               frame_p1, frame_p2,
               action_p1: ActionP1,
               action_p2: ActionP2,
               key_state: dict,
               ml_confidence: float = 0.0):
        """
        Buduje i wyświetla klatkę overlay.
        Zwraca True jeśli kontynuować, False jeśli naciśnięto Q.
        """
        self._update_fps()

        # Resize klatek kamer
        cam1 = _resize(frame_p1, CAM_W, CAM_H)
        cam2 = _resize(frame_p2, CAM_W, CAM_H)

        # Dodaj kolorowe banery akcji nad kamerami
        cam1 = self._draw_player_banner(cam1, "P1: RUCH",
                                        _action_label_p1(action_p1),
                                        _action_color_p1(action_p1))
        cam2 = self._draw_player_banner(cam2, "P2: SKOK/BONUS",
                                        _action_label_p2(action_p2),
                                        _action_color_p2(action_p2))

        # Połącz klatki poziomo
        top = np.hstack([cam1, cam2])

        # Panel sterowania
        panel = self._build_panel(key_state, ml_confidence)

        # Złącz pionowo
        frame = np.vstack([top, panel])

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q') and key != ord('Q')

    # ------------------------------------------------------------------

    def _draw_player_banner(self, cam, title: str, action_label: str, color) -> np.ndarray:
        """Rysuje półprzezroczysty baner akcji na klatce kamery."""
        overlay_frame = cam.copy()
        cv2.rectangle(overlay_frame, (0, 0), (CAM_W, 55), color, -1)
        cv2.addWeighted(overlay_frame, 0.55, cam, 0.45, 0, cam)

        cv2.putText(cam, title,
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(cam, action_label,
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2)
        return cam

    def _build_panel(self, key_state: dict, ml_confidence: float) -> np.ndarray:
        """Buduje dolny panel ze wskaźnikami klawiszy i ML."""
        panel = np.full((PANEL_H, CAM_W * 2, 3), 25, dtype=np.uint8)

        # --- Klawisze ---
        keys_info = [
            ("<<< LEWO", key_state.get("LEFT", False)),
            ("PRAWO >>>", key_state.get("RIGHT", False)),
            ("SKOK ^", key_state.get("JUMP", False)),
            ("* SPECIAL", key_state.get("SPECIAL", False)),
        ]

        box_w = (CAM_W * 2) // len(keys_info)
        for i, (label, active) in enumerate(keys_info):
            x0 = i * box_w + 8
            x1 = (i + 1) * box_w - 8
            color = COLOR_KEY_ACTIVE if active else COLOR_KEY_INACTIVE
            cv2.rectangle(panel, (x0, 10), (x1, 65), color, -1)
            cv2.rectangle(panel, (x0, 10), (x1, 65), (120, 120, 120), 1)

            text_color = (10, 10, 10) if active else (160, 160, 160)
            cv2.putText(panel, label,
                        (x0 + 6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        # --- Pasek ML confidence ---
        bar_x, bar_y = 10, 85
        bar_max_w = CAM_W * 2 - 20
        bar_h = 22

        cv2.putText(panel, f"ML Confidence (SPECIAL): {ml_confidence:.0%}",
                    (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        filled_w = int(bar_max_w * min(ml_confidence, 1.0))
        bar_color = COLOR_SPECIAL if ml_confidence >= 0.75 else (80, 120, 160)
        if filled_w > 0:
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h),
                          bar_color, -1)
        # Linia progu
        threshold_x = bar_x + int(bar_max_w * 0.75)
        cv2.line(panel, (threshold_x, bar_y - 2), (threshold_x, bar_y + bar_h + 2),
                 (255, 200, 0), 2)

        # --- FPS ---
        cv2.putText(panel, f"FPS: {self._fps:.0f}",
                    (CAM_W * 2 - 90, PANEL_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # --- Instrukcja ---
        cv2.putText(panel, "[Q] Zakoncz",
                    (10, PANEL_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        return panel

    def _update_fps(self):
        now = time.monotonic()
        self._fps_times.append(now)
        # Trzymaj tylko ostatnie 30 klatek
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        self._fps = len(self._fps_times)

    def destroy(self):
        cv2.destroyWindow(self.window_name)
