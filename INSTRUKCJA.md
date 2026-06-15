# Instrukcja uruchomienia — od zera do meczu

Wykonaj kroki **po kolei**. Każdy krok ma checkboksa — odhaczaj jak skończysz.

---

## KROK 0 — Sprawdź kamery

Zanim cokolwiek, ustal numery kamer na swoim komputerze.

```bash
python -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Kamera {i}: DOSTEPNA')
        cap.release()
    else:
        print(f'Kamera {i}: brak')
"
```

- [ ] Zapisz numery: P1 = `___` , P2 = `___`  
- [ ] Jeśli masz tylko 1 kamerę, obie osoby siedzą razem przed jedną kamerą — użyj `--cam-p1 0 --cam-p2 0`

---

## KROK 1 — Zbierz dataset

> Robi to **Gracz 2** (ten od skoku/bonusu). Kamera skierowana na jego twarz.

```bash
python ml/collect_dataset.py
```

W oknie pojawi się podgląd kamery i liczniki próbek.

**Co robić:**
- Wciśnij `[Space]` → zapisuje próbkę **IDLE** (neutralna twarz, oczy otwarte)
- Wciśnij `[Enter]` → zapisuje próbkę **SPECIAL** (mrugnij lewym okiem — lewe zamknięte, prawe otwarte)
- Wciśnij `[S]` → sprawdź ile masz próbek
- Wciśnij `[Q]` → zakończ i zapisz

**Cel: co najmniej 100 próbek IDLE i 100 próbek SPECIAL.**

Wskazówki:
- Rób próbki IDLE w różnych pozycjach głowy (lekko w lewo, prawo, prosto)
- Rób próbki SPECIAL z wyraźnym mruganiem — wyraźne zamknięcie lewego oka
- Zbieraj próbki w podobnym oświetleniu jak na turnieju

- [ ] Zebrano ≥ 100 próbek IDLE
- [ ] Zebrano ≥ 100 próbek SPECIAL

---

## KROK 2 — Wytrenuj model

```bash
python ml/train_model.py
```

Powinieneś zobaczyć coś takiego:

```
Model: SVM (rbf)  |  StratifiedKFold(k=5)
  Accuracy : 0.9600 ± 0.0200
  Precision: 0.9500 ± 0.0300  (cel: >85%)
  Walidacja precyzji >85%: PASS ✓
Model zapisany → ml/model.joblib
```

- [ ] Precision ≥ 85% — PASS ✓  
- [ ] Plik `ml/model.joblib` istnieje

> Jeśli FAIL: wróć do Kroku 1 i zbierz więcej próbek albo popraw jakość (wyraźniejsze mrugnięcia, lepsze oświetlenie).

---

## KROK 3 — Test bez gry (tryb demo)

Sprawdź, czy kamera i gesty działają zanim zaczniesz grać.

```bash
python main.py --cam-p1 0 --cam-p2 0 --no-keys
```

*(podstaw właściwe numery kamer z Kroku 0)*

W oknie overlay sprawdź:
- [ ] P1: skręt głowy w lewo → pojawia się `<<< LEWO` (zielone)
- [ ] P1: skręt głowy w prawo → pojawia się `PRAWO >>>` (niebieski)
- [ ] P2: uniesienie brwi lub otwarcie ust → pojawia się `SKOK ↑` (żółte)
- [ ] P2: mrugnięcie lewym okiem → pasek ML skacze do góry i pojawia się `★ SPECIAL` (pomarańczowe)
- [ ] FPS ≥ 15 (najlepiej 25+)

---

## KROK 4 — Kalibracja (opcjonalnie, ale zalecana)

Kalibracja dostosowuje progi do konkretnych twarzy graczy. Zajmuje ~2 minuty.

```bash
python main.py --cam-p1 0 --cam-p2 1
```

*(bez `--no-calibrate` — kalibracja uruchamia się automatycznie jeśli nie ma zapisanych progów)*

Lub ręcznie:
```bash
python calibration/calibrate.py
```

Postępuj zgodnie z instrukcjami na ekranie.

- [ ] Kalibracja zakończona — plik `calibration/thresholds.json` istnieje

---

## KROK 5 — Test z grą (bez publiczności)

1. Otwórz [https://www.blobby-online.com](https://www.blobby-online.com) w przeglądarce
2. Dołącz do gry i ustaw **focus na okno przeglądarki** (kliknij w grę raz myszą)
3. Uruchom system:

```bash
python main.py --cam-p1 0 --cam-p2 1 --no-calibrate
```

4. Sprawdź w grze:
   - [ ] Blobby rusza się w lewo gdy P1 skręca głowę w lewo
   - [ ] Blobby rusza się w prawo gdy P1 skręca głowę w prawo
   - [ ] Blobby skacze gdy P2 unosi brwi
   - [ ] Bonus aktywuje się gdy P2 mruga lewym okiem

> **WAŻNE:** Okno przeglądarki musi być aktywne (na pierwszym planie) — `pynput` wysyła klawisze do aktywnego okna. Okno overlay OpenCV może być z boku.

---

## KROK 6 — Gotowi do meczu

Procedura startowa przed każdym meczem:

```bash
# 1. Uruchom system (jeśli kalibracja już jest zapisana)
python main.py --cam-p1 0 --cam-p2 1 --no-calibrate

# 2. Kliknij raz w okno przeglądarki (focus)
# 3. Graj
```

- [ ] System stabilny przez 5 minut bez crashy
- [ ] FPS ≥ 15 przez cały czas
- [ ] Overlay widoczny obok gry

---

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---------|-------------|
| `Nie można otworzyć kamery` | Zmień numer `--cam-p1` lub `--cam-p2` (Krok 0) |
| Blobby nie reaguje na klawisze | Kliknij w okno przeglądarki, żeby dać mu focus |
| SPECIAL nie wykrywa winka | Wróć do Kroku 1, zbierz więcej danych i ponów Krok 2 |
| Dużo fałszywych SPECIAL | Podnieś próg w `ml/classifier.py` → `threshold=0.85` |
| Zbyt wrażliwy skok P2 | Zmień `DEFAULT_BROW_THRESHOLD` w `detection/gestures_p2.py` |
| Zbyt wrażliwy ruch P1 | Zmień `DEFAULT_YAW_THRESHOLD` w `detection/gestures_p1.py` |
| Niska precyzja ML | Zbierz więcej próbek, zwłaszcza IDLE w różnych pozycjach |
| Crash po dłuższym czasie | Sprawdź zużycie RAM — zamknij inne aplikacje |

---

## Skróty klawiszowe w overlay

| Klawisz | Akcja |
|---------|-------|
| `Q` | Zamknij system |
| `S` | (w collect_dataset) Pokaż statystyki |

---

## Czas potrzebny na setup

| Czynność | Czas |
|----------|------|
| Zbieranie datasetu | ~10 min |
| Trening modelu | ~1 min |
| Test demo | ~5 min |
| Kalibracja | ~5 min |
| Test z grą | ~10 min |
| **Łącznie** | **~30 min** |
