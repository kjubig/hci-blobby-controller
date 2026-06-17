"""
tools/check_dataset.py — inspekcja jakości etykiet w data/dataset.npz

Zamiast zgadywać próg "wyraźnej asymetrii" z góry, wyliczamy go
EMPIRYCZNIE z rozkładu danych: próg to punkt leżący pomiędzy średnią
ear_asymmetry w klasie IDLE i średnią w klasie SPECIAL, z uwzględnieniem
odchylenia standardowego obu grup (podobnie jak granica decyzyjna
między dwoma rozkładami Gaussa o różnej wariancji).

Dzięki temu próg jest dopasowany do gestu osoby grające (np. jeśli mruga
niewyraźnie i asymetria jest mniejsza niż "podręcznikowa", próg
automatycznie się obniży, zamiast fałszywie oznaczać poprawne
próbki jako podejrzane).

Uruchom z głównego katalogu repo:
    python3 tools/check_dataset.py
"""

import os
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.npz")


def compute_empirical_threshold(asym_idle: np.ndarray, asym_special: np.ndarray) -> float:
    """
    Wyznacza próg jako punkt przecięcia między rozkładami IDLE i SPECIAL,
    ważony odchyleniem standardowym (podobnie do granicy decyzyjnej LDA
    dla jednej cechy, przy założeniu rozkładów zbliżonych do normalnych).

    threshold = (mean_idle * std_special + mean_special * std_idle)
                / (std_idle + std_special)

    Gdy std_idle == std_special, redukuje się to do prostej średniej
    dwóch średnich — co jest intuicyjnym, bezpiecznym fallbackiem.
    """
    mean_idle, std_idle = asym_idle.mean(), asym_idle.std() + 1e-9
    mean_special, std_special = asym_special.mean(), asym_special.std() + 1e-9

    threshold = (mean_idle * std_special + mean_special * std_idle) / (std_idle + std_special)
    return threshold, mean_idle, std_idle, mean_special, std_special


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Nie znaleziono datasetu: {DATASET_PATH}")
        print("Najpierw zbierz dane: python3 ml/collect_dataset.py")
        return

    data = np.load(DATASET_PATH)
    X, y = data["X"], data["y"]

    print(f"Łącznie próbek: {len(y)}  (IDLE={sum(y==0)}, SPECIAL={sum(y==1)})\n")

    asym_idle = X[y == 0][:, 5]
    asym_special = X[y == 1][:, 5]

    threshold, m_idle, s_idle, m_special, s_special = compute_empirical_threshold(
        asym_idle, asym_special
    )

    print("=== Rozkład ear_asymmetry w obu klasach ===")
    print(f"  IDLE:    mean={m_idle:+.4f}  std={s_idle:.4f}")
    print(f"  SPECIAL: mean={m_special:+.4f}  std={s_special:.4f}")
    print(f"\n  Próg wyznaczony empirycznie: {threshold:.4f}")

    separates_upward = m_special > m_idle

    print("=" * 60)
    print("=== SPECIAL — próbki bliskie lub po złej stronie progu ===")
    suspicious_special = []
    for i, row in enumerate(X[y == 1]):
        a = row[5]
        is_suspicious = (a < threshold) if separates_upward else (a > threshold)
        flag = "  <-- BLISKO/ZA PROGIEM" if is_suspicious else ""
        if is_suspicious:
            suspicious_special.append(i)
        print(f"  #{i:3d}: ear_L={row[0]:.3f}  ear_R={row[1]:.3f}  asym={a:+.3f}{flag}")

    print(f"\n=== IDLE — próbki bliskie lub po złej stronie progu ===")
    suspicious_idle = []
    for i, row in enumerate(X[y == 0]):
        a = row[5]
        is_suspicious = (a >= threshold) if separates_upward else (a <= threshold)
        flag = "  <-- BLISKO/ZA PROGIEM" if is_suspicious else ""
        if is_suspicious:
            suspicious_idle.append(i)
        print(f"  #{i:3d}: ear_L={row[0]:.3f}  ear_R={row[1]:.3f}  asym={a:+.3f}{flag}")

    print(f"\n--- Podsumowanie ---")
    print(f"Próg empiryczny:                     {threshold:.4f}")
    print(f"Podejrzane SPECIAL (za progiem):      {len(suspicious_special)}/{len(asym_special)}")
    print(f"Podejrzane IDLE (za progiem):         {len(suspicious_idle)}/{len(asym_idle)}")


if __name__ == "__main__":
    main()