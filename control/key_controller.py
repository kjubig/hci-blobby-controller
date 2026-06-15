"""
key_controller.py — symulacja naciśnięć klawiszy dla Blobby Volley.

Mapowanie klawiszy (tryb 1-player w przeglądarce):
  Lewo      → strzałka lewa  (←)
  Prawo     → strzałka prawa (→)
  Skok      → strzałka góra  (↑)  lub spacja
  Specjalny → Shift lewy

Maszyna stanów zapobiega zalewaniu gry eventami.
Throttle: min. 50ms między zmianami stanu.
"""

import time
from enum import Enum, auto
from pynput.keyboard import Key, Controller

from detection.gestures_p1 import ActionP1
from detection.gestures_p2 import ActionP2


class GameKey(Enum):
    LEFT = auto()
    RIGHT = auto()
    JUMP = auto()
    SPECIAL = auto()


# Mapowanie akcji na klawisze gry
KEY_MAP = {
    GameKey.LEFT: Key.left,
    GameKey.RIGHT: Key.right,
    GameKey.JUMP: Key.up,
    GameKey.SPECIAL: Key.shift_l,
}

THROTTLE_MS = 50  # minimalne ms między zmianami stanu


class KeyController:
    """
    Utrzymuje zestaw wciśniętych klawiszy i aktualizuje je
    na podstawie akcji P1 i P2.
    """

    def __init__(self):
        self.keyboard = Controller()
        self._pressed: set = set()
        self._last_update: float = 0.0

    # ------------------------------------------------------------------
    # Prywatne pomocniki
    # ------------------------------------------------------------------

    def _press(self, key):
        if key not in self._pressed:
            self.keyboard.press(key)
            self._pressed.add(key)

    def _release(self, key):
        if key in self._pressed:
            self.keyboard.release(key)
            self._pressed.discard(key)

    def _release_all(self):
        for key in list(self._pressed):
            self._release(key)

    # ------------------------------------------------------------------
    # Główna metoda aktualizacji
    # ------------------------------------------------------------------

    def update(self, action_p1: ActionP1, action_p2: ActionP2):
        """
        Wywoływana co klatkę. Aktualizuje stan klawiszy
        z throttlingiem 50ms.
        """
        now = time.monotonic() * 1000
        if now - self._last_update < THROTTLE_MS:
            return
        self._last_update = now

        # --- Gracz 1: ruch ---
        if action_p1 == ActionP1.LEFT:
            self._press(KEY_MAP[GameKey.LEFT])
            self._release(KEY_MAP[GameKey.RIGHT])
        elif action_p1 == ActionP1.RIGHT:
            self._press(KEY_MAP[GameKey.RIGHT])
            self._release(KEY_MAP[GameKey.LEFT])
        else:
            self._release(KEY_MAP[GameKey.LEFT])
            self._release(KEY_MAP[GameKey.RIGHT])

        # --- Gracz 2: skok i special ---
        if action_p2 == ActionP2.SPECIAL:
            self._press(KEY_MAP[GameKey.SPECIAL])
            self._release(KEY_MAP[GameKey.JUMP])
        elif action_p2 == ActionP2.JUMP:
            self._press(KEY_MAP[GameKey.JUMP])
            self._release(KEY_MAP[GameKey.SPECIAL])
        else:
            self._release(KEY_MAP[GameKey.JUMP])
            self._release(KEY_MAP[GameKey.SPECIAL])

    def release_all(self):
        """Zwalnia wszystkie klawisze — wywoływane przy zamykaniu."""
        self._release_all()

    def get_state(self) -> dict:
        """Zwraca aktualny stan (do overlay)."""
        return {
            "LEFT": KEY_MAP[GameKey.LEFT] in self._pressed,
            "RIGHT": KEY_MAP[GameKey.RIGHT] in self._pressed,
            "JUMP": KEY_MAP[GameKey.JUMP] in self._pressed,
            "SPECIAL": KEY_MAP[GameKey.SPECIAL] in self._pressed,
        }
