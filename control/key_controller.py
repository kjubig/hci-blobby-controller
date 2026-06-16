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

THROTTLE_MS = 50       # minimalne ms między zmianami stanu ruchu
JUMP_HOLD_MS = 80      # czas trzymania klawisza skoku (ms) — po tym auto-release
JUMP_COOLDOWN_MS = 500 # minimalny czas między kolejnymi skokami (ms)


class KeyController:
    """
    Utrzymuje zestaw wciśniętych klawiszy i aktualizuje je
    na podstawie akcji P1 i P2.

    JUMP działa jako jednorazowy tap (press + auto-release po JUMP_HOLD_MS),
    a nie jako trzymanie klawisza — zgodnie z mechaniką Blobby Volley.
    """

    def __init__(self):
        self.keyboard = Controller()
        self._pressed: set = set()
        self._last_update: float = 0.0
        # Stan skoku — edge detection
        self._prev_action_p2: ActionP2 = ActionP2.IDLE
        self._jump_pressed_at: float = 0.0
        self._jump_cooldown_until: float = 0.0

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
        Wywoływana co klatkę. Aktualizuje stan klawiszy.
        - Ruch (L/R): trzymanie klawisza tak długo jak gest trwa.
        - JUMP: jednorazowy tap na wznoszącym zboczu gestu + auto-release po JUMP_HOLD_MS.
        - SPECIAL: trzymanie (zależnie od mechaniki gry może wymagać korekty).
        """
        now = time.monotonic() * 1000

        # --- Auto-release skoku po JUMP_HOLD_MS ---
        jump_key = KEY_MAP[GameKey.JUMP]
        if jump_key in self._pressed and (now - self._jump_pressed_at) >= JUMP_HOLD_MS:
            self._release(jump_key)

        # Throttle dla ruchu i speciala
        if now - self._last_update < THROTTLE_MS:
            self._prev_action_p2 = action_p2
            return
        self._last_update = now

        # --- Gracz 1: ruch (trzymanie klawisza) ---
        if action_p1 == ActionP1.LEFT:
            self._press(KEY_MAP[GameKey.LEFT])
            self._release(KEY_MAP[GameKey.RIGHT])
        elif action_p1 == ActionP1.RIGHT:
            self._press(KEY_MAP[GameKey.RIGHT])
            self._release(KEY_MAP[GameKey.LEFT])
        else:
            self._release(KEY_MAP[GameKey.LEFT])
            self._release(KEY_MAP[GameKey.RIGHT])

        # --- Gracz 2: skok (tap na wznoszącym zboczu) ---
        jump_rising_edge = (
            action_p2 == ActionP2.JUMP
            and self._prev_action_p2 != ActionP2.JUMP
            and now >= self._jump_cooldown_until
        )
        if jump_rising_edge:
            self._press(jump_key)
            self._jump_pressed_at = now
            self._jump_cooldown_until = now + JUMP_COOLDOWN_MS

        # --- Gracz 2: special (trzymanie) ---
        if action_p2 == ActionP2.SPECIAL:
            self._press(KEY_MAP[GameKey.SPECIAL])
        else:
            self._release(KEY_MAP[GameKey.SPECIAL])

        self._prev_action_p2 = action_p2

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
