# MICM Blobby Volley — Face Controller

Interfejs sterowania grą [Blobby Volley Online](https://www.blobby-online.com) przy użyciu analizy obrazu z kamery w czasie rzeczywistym (MediaPipe FaceMesh).

---

## Cel projektu

Stworzenie systemu HCI (Human-Computer Interaction), w którym dwóch graczy steruje jednym "Blobbem" wyłącznie mimiką twarzy i ruchem głowy — bez użycia klawiatury ani myszy.

---

## Podział ról

| Gracz | Odpowiada za | Gest |
|-------|-------------|------|
| **P1** | Ruch (lewo / prawo) | Przechylenie głowy w lewo lub prawo |
| **P2** | Skok | Uniesienie obu brwi LUB otwarcie ust |
| **P2** | Bonus (gest specjalny) | Mrugnięcie **lewym okiem** (wink) — klasyfikator ML |

---

## Architektura

```
Kamera P1                          Kamera P2
    │                                  │
    ▼                                  ▼
MediaPipe FaceMesh             MediaPipe FaceMesh
    │                                  │
gestures_p1.py              gestures_p2.py + classifier.py (SVM)
LEFT / RIGHT / IDLE         JUMP / SPECIAL / IDLE
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
          key_controller.py
          (pynput — symulacja klawiszy)
                   │
                   ▼
        Blobby-online.com (przeglądarka)
                   │
                   ▼
            overlay.py
      (podgląd kamer + wskaźniki akcji)
```

---

## Struktura plików

```
MICM PROJEKT/
├── main.py                    ← główna pętla (uruchamiasz to)
├── requirements.txt
├── detection/
│   ├── face_mesh.py           ← wrapper MediaPipe + funkcje EAR/MAR/brow
│   ├── gestures_p1.py         ← detekcja ruchu głową (P1)
│   └── gestures_p2.py         ← detekcja brwi/ust + wektor cech dla ML (P2)
├── ml/
│   ├── collect_dataset.py     ← zbieranie datasetu (Space = idle, Enter = special)
│   ├── train_model.py         ← trening SVM + MLP, walidacja StratifiedKFold(5)
│   └── classifier.py          ← runtime wrapper wytrenowanego modelu
├── control/
│   └── key_controller.py      ← maszyna stanów + pynput, throttle 50ms
├── ui/
│   └── overlay.py             ← wizualizacja: kamery, akcje, pasek ML, FPS
├── calibration/
│   └── calibrate.py           ← kalibracja progów per-użytkownik
└── data/
    └── dataset.npz            ← (generowany) dataset do treningu
```

---

## Wymagania techniczne

- Python 3.10+
- Kamera USB (1 lub 2 sztuki)
- System: Windows (testowane) / Linux

### Instalacja zależności

```bash
pip install -r requirements.txt
```

### Zawartość requirements.txt

```
mediapipe>=0.10.0
opencv-python>=4.8.0
pynput>=1.7.6
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0
pyautogui>=0.9.54
```

---

## Uruchomienie

### Standardowe (2 kamery)
```bash
python main.py --cam-p1 0 --cam-p2 1
```

### Jedna kamera (obie osoby przed jedną kamerą — tryb testowy)
```bash
python main.py --cam-p1 0 --cam-p2 0
```

### Tryb demo — bez symulacji klawiszy
```bash
python main.py --no-keys
```

### Z pominięciem kalibracji (użyj zapisanych progów)
```bash
python main.py --no-calibrate
```

---

## Mapowanie klawiszy

| Akcja | Klawisz |
|-------|---------|
| Ruch w lewo | `←` (strzałka lewa) |
| Ruch w prawo | `→` (strzałka prawa) |
| Skok | `↑` (strzałka góra) |
| Bonus specjalny | `Shift L` |

> Klawisze można zmienić w `control/key_controller.py` → słownik `KEY_MAP`.

---

## Model ML — Gest Specjalny

- **Klasa 0:** `idle` — neutralna twarz
- **Klasa 1:** `special` — mrugnięcie lewym okiem (lewe oko zamknięte, prawe otwarte)
- **Cechy:** 6 wartości numerycznych (EAR L/P, MAR, brow L/P, asymetria oczu)
- **Modele:** SVM (rbf) i MLP (64-32), wybierany najlepszy
- **Walidacja:** StratifiedKFold(k=5), cel precision > 85%

---

## Kalibracja

System przeprowadza 4-etapową kalibrację (ok. 20s per gracz):

1. Neutralna twarz (5s) — wyznaczenie baseline
2. Skręt głowy w lewo (3s) — próg ruchu P1
3. Skręt głowy w prawo (3s) — próg ruchu P1
4. Uniesienie brwi / otwarcie ust (3s) — próg skoku P2

Wyniki zapisywane do `calibration/thresholds.json`.

---

## Parametry wydajności

| Parametr | Wartość |
|----------|---------|
| Rozdzielczość kamery | 640×480 @ 30fps |
| Throttle klawiszy | 50ms |
| Wygładzanie akcji | okno 4–5 klatek |
| Cel latencji | < 200ms |

---

## Autorzy

Projekt zaliczeniowy — Metody Interakcji Człowiek–Maszyna, 2025/2026.
