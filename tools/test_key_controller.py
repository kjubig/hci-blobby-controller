"""
test_key_controller.py — sprawdza, czy KeyController faktycznie wysyła
zdarzenia klawiatury na poziomie systemu, BEZ użycia kamery i BEZ gry.

Jak to działa:
  1. Tworzenie osobnego pynput.keyboard.Listener, który nasłuchuje
     WSZYSTKICH zdarzeń klawiatury w systemie (niezależnie od KeyController).
  2. Wywołanie KeyController.update(...) z różnymi kombinacjami akcji
     P1/P2, symulując to co main.py robiłby na podstawie gestów.
  3. Listener loguje każde press/release, które faktycznie dotarło
     do systemu — to dowód, że pynput działa i ma odpowiednie uprawnienia.

WAŻNE (macOS): przy pierwszym uruchomieniu system poprosi o uprawnienia
Accessibility dla terminala/IDE. Jeśli nie zobaczysz żadnych zdarzeń
w logu mimo że skrypt "działa", to prawdopodobnie ten problem — sprawdź
System Settings → Privacy & Security → Accessibility.

Uruchom z głównego katalogu repo:
    python3 tools/test_key_controller.py

Podczas testu NIE rób nic innego na klawiaturze — listener złapie
też Twoje przypadkowe naciśnięcia i to zaśmieci log.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pynput.keyboard import Listener, Key
from control.key_controller import KeyController, GameKey, KEY_MAP
from detection.gestures_p1 import ActionP1
from detection.gestures_p2 import ActionP2


# ----------------------------------------------------------------------
# Listener
# ----------------------------------------------------------------------

captured_events = []

def on_press(key):
    captured_events.append(("PRESS", key, time.monotonic()))

def on_release(key):
    captured_events.append(("RELEASE", key, time.monotonic()))


def describe_key(k):
    """Czytelna nazwa klawisza do logu."""
    for game_key, mapped in KEY_MAP.items():
        if mapped == k:
            return f"{game_key.name} ({k})"
    return str(k)


def run_scenario(name, controller, action_p1, action_p2, repeats=1, delay=0.06):
    """Wywołuje controller.update() kilka razy z tą samą akcją,
    żeby symulować kilka klatek pod rząd (tak jak w main.py)."""
    print(f"\n--- Scenariusz: {name} ---")
    print(f"    action_p1={action_p1.name}  action_p2={action_p2.name}")
    for _ in range(repeats):
        controller.update(action_p1, action_p2)
        time.sleep(delay)


def main():
    print("=" * 60)
    print("  Test KeyController — bez kamery, bez gry")
    print("=" * 60)

    print("UWAGA: Podczas testu NIE rób nic innego na klawiaturze — listener złapie też Twoje przypadkowe naciśnięcia i to zaśmieci log.")
    time.sleep(2)

    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()
    time.sleep(0.3)  # daj listenerowi chwilę na start

    if not listener.running:
        print("UWAGA: Listener się nie uruchomił — sprawdź uprawnienia Accessibility (macOS).")
        return

    controller = KeyController()

    try:
        # 1. Ruch w lewo (P1), neutralne P2
        run_scenario("Ruch LEWO", controller, ActionP1.LEFT, ActionP2.IDLE, repeats=3)

        # 2. Powrót do neutralnego — sprawdzamy czy LEWO się zwalnia
        run_scenario("Powrót do IDLE (P1)", controller, ActionP1.IDLE, ActionP2.IDLE, repeats=2)

        # 3. Ruch w prawo
        run_scenario("Ruch PRAWO", controller, ActionP1.RIGHT, ActionP2.IDLE, repeats=3)
        run_scenario("Powrót do IDLE (P1)", controller, ActionP1.IDLE, ActionP2.IDLE, repeats=2)

        # 4. Skok (P2) — sprawdzamy edge detection: powinien być JEDEN tap,
        #    nawet jeśli JUMP trwa przez kilka klatek
        run_scenario("SKOK (utrzymywany 5 klatek)", controller, ActionP1.IDLE, ActionP2.JUMP, repeats=5)
        # Czekamy dłużej niż JUMP_HOLD_MS, żeby zobaczyć auto-release
        time.sleep(0.15)
        controller.update(ActionP1.IDLE, ActionP2.IDLE)

        # 5. SPECIAL (trzymanie)
        run_scenario("SPECIAL", controller, ActionP1.IDLE, ActionP2.SPECIAL, repeats=3)
        run_scenario("Powrót do IDLE (P2)", controller, ActionP1.IDLE, ActionP2.IDLE, repeats=2)

    finally:
        print("\n--- Zwalnianie wszystkich klawiszy (cleanup) ---")
        controller.release_all()
        time.sleep(0.2)
        listener.stop()

    # ------------------------------------------------------------------
    # Raport
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RAPORT — zdarzenia faktycznie zarejestrowane przez system")
    print("=" * 60)

    if not captured_events:
        print("BRAK ZDARZEŃ! To oznacza, że pynput nie wysyła nic do systemu.")
        print("Najczęstsza przyczyna na macOS: brak uprawnień Accessibility")
        print("dla terminala/IDE w System Settings -> Privacy & Security -> Accessibility.")
        return

    relevant_keys = set(KEY_MAP.values())
    relevant_events = [e for e in captured_events if e[1] in relevant_keys]

    print(f"Łącznie zdarzeń (wszystkie): {len(captured_events)}")
    print(f"Zdarzenia dotyczące klawiszy gry (LEFT/RIGHT/JUMP/SPECIAL): {len(relevant_events)}\n")

    if not relevant_events:
        print("UWAGA: nie zarejestrowano ŻADNEGO zdarzenia dla klawiszy gry,")
        print("mimo że jakieś zdarzenia w systemie były (może to inne klawisze?).")
        return

    t0 = relevant_events[0][2]
    for action, key, t in relevant_events:
        print(f"  [{t - t0:6.3f}s] {action:8s} {describe_key(key)}")

    print("\nJeśli widzisz powyżej PRESS i RELEASE dla LEFT, RIGHT, JUMP, SPECIAL")
    print("w sensownej kolejności (np. SKOK jako jedna szybka para PRESS/RELEASE,")
    print("nie ciągłe trzymanie) — KeyController działa poprawnie na poziomie systemu.")
    print("\nJeśli to działa, a gra na blobby-online.com NIE reaguje, problem jest")
    print("po stronie przeglądarki/strony (np. focus okna, blokowanie syntetycznych")
    print("zdarzeń) — nie w kodzie Pythona.")


if __name__ == "__main__":
    main()