# Sprawozdanie z mini-projektu
## Metody Interakcji Człowiek-Maszyna

**Temat:** Interfejs sterowania grą Blobby Volley przy użyciu analizy obrazu w czasie rzeczywistym

**Autorzy:** [...], [...]

---

## 1. Cel projektu

Celem projektu było zaprojektowanie i implementacja systemu sterowania grą Blobby Volley (tryb online, `blobby-online.com`) wyłącznie za pomocą analizy obrazu z kamery — mimiki twarzy oraz ruchu głowy, bez użycia klawiatury czy myszy. Sterowanie podzielone jest między dwóch graczy tworzących jeden zespół: jeden gracz odpowiada za ruch poziomy blobba, drugi za skok oraz aktywację umiejętności specjalnej (bonusu).

Zgodnie z wymaganiami projektu, aktywacja bonusu nie mogła być zrealizowana regułami warunkowymi (if/else), a musiała wynikać z predykcji samodzielnie wytrenowanego modelu klasyfikacyjnego uczenia maszynowego.

---

## 2. Podział ról w zespole

| Gracz | Odpowiada za | Gest |
|-------|-------------|------|
| **P1** | Ruch (lewo / prawo) | Przechylenie głowy w lewo lub prawo |
| **P2** | Skok | Uniesienie obu brwi LUB otwarcie ust |
| **P2** | Bonus (gest specjalny) | Mrugnięcie lewym okiem (wink) — klasyfikator ML |

---

## 3. Architektura systemu

System składa się z dwóch równoległych potoków przetwarzania obrazu (jeden na gracza), których wyniki są łączone i tłumaczone na zdarzenia klawiatury wysyłane do przeglądarki.

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

### 3.1 Komponenty systemu

**Detekcja punktów charakterystycznych twarzy.** Do detekcji landmarków twarzy wykorzystano bibliotekę MediaPipe, a konkretnie nowsze API `FaceLandmarker` (MediaPipe Tasks API, wersja ≥0.10.x), które dostarcza 468 punktów charakterystycznych twarzy w czasie rzeczywistym, działając w trybie `VIDEO` z monotonicznie rosnącym znacznikiem czasu dla każdej z dwóch niezależnych kamer.

**Detekcja gestów P1 (ruch).** Na podstawie pozycji punktów referencyjnych twarzy (m.in. punktu nosa) wyznaczany jest kąt odchylenia głowy względem osi neutralnej (yaw). Przekroczenie ustalonego progu w jedną lub drugą stronę interpretowane jest jako ruch w lewo lub w prawo.

**Detekcja gestów P2 (skok i bonus).** Skok wykrywany jest regułowo na podstawie dwóch niezależnych sygnałów: uniesienia obu brwi powyżej progu lub otwarcia ust powyżej progu (warunek "LUB"). Aktywacja bonusu nie jest wykrywana regułowo — wynika z predykcji wytrenowanego klasyfikatora SVM, opisanego w sekcji 5.

**Symulacja klawiszy.** Moduł `key_controller.py` tłumaczy rozpoznane akcje na zdarzenia klawiatury wysyłane przez bibliotekę `pynput`. Zastosowano maszynę stanów z throttlingiem (50 ms między aktualizacjami) oraz mechanizm edge-detection dla skoku — klawisz skoku jest wciskany jednorazowo (tap) przy wykryciu narastającego zbocza gestu, a następnie automatycznie zwalniany po 80 ms, z cooldownem 500 ms zapobiegającym wielokrotnemu skakaniu przy przeciągniętym geście.

**Wizualizacja (overlay).** Interfejs użytkownika wyświetla podgląd obu kamer, aktualnie rozpoznane akcje obu graczy, poziom pewności klasyfikatora ML dla gestu bonusowego oraz liczbę klatek na sekundę (FPS).

---

## 4. Opis cech wykorzystanych w detekcji

System wykorzystuje trzy podstawowe miary geometryczne wyliczane z landmarków twarzy, znane z literatury dotyczącej analizy mimiki.

### 4.1 Eye Aspect Ratio (EAR)

EAR opisuje stopień otwarcia oka na podstawie sześciu punktów charakterystycznych powieki:

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 · ||p1−p4||)
```

gdzie `p1`–`p6` to punkty rozmieszczone wokół konturu oka — `p1` i `p4` wyznaczają oś horyzontalną (kąciki oka), a pary `p2`-`p6` i `p3`-`p5` wyznaczają odległości wertykalne. Wartość EAR bliska 0 oznacza oko zamknięte, natomiast wartości w okolicy 0.25–0.35 odpowiadają oku otwartemu. EAR liczone jest niezależnie dla oka lewego i prawego, co pozwala wykryć asymetrię potrzebną do detekcji mrugnięcia jednym okiem (wink).

### 4.2 Mouth Aspect Ratio (MAR)

MAR opisuje stopień otwarcia ust jako stosunek odległości wertykalnej (góra-dół wargi) do odległości horyzontalnej (kąciki ust):

```
MAR = ||top−bottom|| / ||left−right||
```

Wyższe wartości MAR odpowiadają szerzej otwartym ustom. MAR wykorzystywane jest do detekcji skoku (P2).

### 4.3 Wysokość brwi (brow height)

Miara wysokości brwi liczona jest jako znormalizowana różnica pomiędzy średnią pozycją wertykalną punktów brwi a średnią pozycją wertykalną punktów oka, znormalizowana przez szerokość oka:

```
brow_height = (eye_center_y − brow_center_y) / eye_width
```

Wyższa wartość oznacza brwi uniesione wyżej względem naturalnej pozycji. Miara liczona jest niezależnie dla strony lewej i prawej twarzy.

### 4.4 Wektor cech dla klasyfikatora ML

Do klasyfikatora gestu bonusowego (SPECIAL) przekazywany jest 6-elementowy wektor cech:

| # | Cecha | Opis |
|---|-------|------|
| 1 | `ear_left` | EAR oka lewego |
| 2 | `ear_right` | EAR oka prawego |
| 3 | `mar` | Otwarcie ust |
| 4 | `brow_left` | Wysokość brwi lewej |
| 5 | `brow_right` | Wysokość brwi prawej |
| 6 | `ear_asymmetry` | Różnica `ear_right − ear_left` |

Cecha `ear_asymmetry` ma szczególne znaczenie dla detekcji winku — przy mrugnięciu lewym okiem (lewe oko zamknięte, prawe otwarte) przyjmuje ona wyraźnie dodatnie wartości, podczas gdy w pozycji neutralnej oscyluje wokół zera.

---

## 5. Model klasyfikacyjny gestu bonusowego

### 5.1 Zbiór danych

Zbiór danych treningowych przygotowano samodzielnie przy użyciu dedykowanego narzędzia (`ml/collect_dataset.py`), pozwalającego na ręczne oznaczanie próbek w czasie rzeczywistym podczas podglądu kamery: naciśnięcie klawisza Space zapisuje aktualny wektor cech jako klasę `idle`, naciśnięcie Enter — jako klasę `special`.

Finalny zbiór danych liczy **800 próbek** (400 klasy `idle`, 400 klasy `special`), znacznie przekraczając minimalne wymaganie projektu (50 próbek na klasę). Dataset zbierany był przez obu graczy, jednak z przeważającą liczbą próbek od gracza P2, na jego własnej twarzy, ponieważ to on wykonuje gest bonusowy podczas rozgrywki — model musiał być dopasowany do indywidualnych cech anatomicznych konkretnej osoby, nie do "podręcznikowego" wzorca gestu.

### 5.2 Proces walidacji jakości danych

W trakcie zbierania danych zaobserwowano, że naturalne wykonanie gestu winku przez daną osobę nie zawsze daje wyraźną, jednoznaczną asymetrię oczu — w praktyce drugie oko często ulega częściowemu, mimowolnemu przymrużeniu. W celu oceny jakości i jednoznaczności etykiet w zbiorze danych przygotowano dedykowane narzędzie diagnostyczne (`tools/check_dataset.py`).

Narzędzie wyznacza próg rozróżnienia klas **empirycznie**, na podstawie rozkładu cechy `ear_asymmetry` w obu klasach, zamiast przyjmować arbitralną wartość z góry:

```
threshold = (mean_idle · std_special + mean_special · std_idle) / (std_idle + std_special)
```

Takie podejście jest odpowiednikiem wyznaczenia granicy decyzyjnej pomiędzy dwoma rozkładami o różnej wariancji, ważonej ich odchyleniami standardowymi — dzięki czemu próg automatycznie dopasowuje się do rzeczywistej, indywidualnej charakterystyki gestu danej osoby, zamiast wymuszać sztywne kryterium.

Wynik analizy na pełnym zbiorze danych:

```
Łącznie próbek: 800  (IDLE=400, SPECIAL=400)

=== Rozkład ear_asymmetry w obu klasach ===
  IDLE:    mean=-0.0125  std=0.0241
  SPECIAL: mean=+0.0929  std=0.0696

  Próg wyznaczony empirycznie: 0.0146
  
  
--- Podsumowanie ---
Próg empiryczny:                     0.0146
Podejrzane SPECIAL (za progiem):      67/400
Podejrzane IDLE (za progiem):         22/400

```
Świadczy to o tym, że próbki poza progiem dla klasy special to 16.75% zbioru, natomiast dla IDLE 5.5%. Jest to akceptowalny wynik.

Średnia wartość asymetrii w klasie `special` (+0.093) jest istotnie wyższa niż w klasie `idle` (−0.013), co potwierdza, że gest jest rozróżnialny na poziomie pojedynczej cechy, choć z naturalną zmiennością — część próbek leży bliżej granicy decyzyjnej. Ponieważ klasyfikator wykorzystuje sześć cech jednocześnie, a nie tylko `ear_asymmetry` w izolacji, próbki graniczne dla tej jednej cechy mogą być poprawnie klasyfikowane na podstawie kombinacji wszystkich wymiarów wektora.

### 5.3 Trening i walidacja modelu

Do klasyfikacji wykorzystano dwa modele: SVM z jądrem RBF oraz wielowarstwowy perceptron (MLP, warstwy 64-32 neuronów). Oba modele poddano walidacji krzyżowej metodą StratifiedKFold (k=5), zapewniającą zachowanie proporcji klas w każdym podziale danych.

```
=== Trening modelu klasyfikatora MICM ===

[Train] Dataset: 800 próbek, 6 cech
  Klasa 0 (idle): 400 próbek
  Klasa 1 (special): 400 próbek

==================================================
  Model: SVM (rbf)  |  StratifiedKFold(k=5)
==================================================
  Accuracy : 0.9537 ± 0.0192
  Precision: 0.9549 ± 0.0181  (cel: >85%)
  Recall   : 0.9537 ± 0.0192
  F1       : 0.9537 ± 0.0193
  Walidacja precyzji >85%: PASS ✓

==================================================
  Model: MLP (64-32)  |  StratifiedKFold(k=5)
==================================================
  Accuracy : 0.9225 ± 0.0252
  Precision: 0.9230 ± 0.0254  (cel: >85%)
  Recall   : 0.9225 ± 0.0252
  F1       : 0.9225 ± 0.0252
  Walidacja precyzji >85%: PASS ✓

[Train] Najlepszy model spełniający kryterium: SVM (rbf) (precision=0.9549)
```

Wybrano model SVM (RBF) jako osiągający wyższą precyzję w walidacji krzyżowej. Raport końcowy na pełnym zbiorze danych:

```
              precision    recall  f1-score   support

        idle       0.98      0.97      0.98       400
     special       0.97      0.98      0.98       400

    accuracy                           0.98       800
   macro avg       0.98      0.98      0.98       800
weighted avg       0.98      0.98      0.98       800

Confusion matrix:
              idle   special   ← przewidywane
  idle         389        11
  special        8       392
```

Model uzyskał precyzję **95.49%** w walidacji krzyżowej (k=5) oraz **98%** na pełnym zbiorze danych, znacznie przekraczając wymagany w instrukcji próg 85%. Macierz błędów pokazuje zrównoważoną liczbę błędnych klasyfikacji w obu kierunkach (11 i 8 błędów), bez wyraźnego skosu w stronę jednej z klas.

### 5.4 Wykryty i naprawiony problem: kolizja gestów JUMP i SPECIAL

W początkowej implementacji logika decyzyjna w `gestures_p2.py` przyznawała absolutny priorytet wynikowi klasyfikatora ML względem reguły detekcji skoku (`if ml_special: return SPECIAL`, niezależnie od stanu detekcji JUMP). W testach na żywo zaobserwowano, że przy szerokim otwarciu ust (gest SKOK) model ML niekiedy błędnie zwracał wysoką pewność klasy `special` (do 86%).

Analiza wykazała, że żadna próbka w zbiorze treningowym klasy `idle` nie zawierała wartości MAR przekraczającej domyślny próg detekcji skoku (`max(MAR_idle) = 0.543` wobec progu `0.55`) — model nigdy nie był uczony na przykładach "szerokie otwarcie ust + symetryczne oczy" jako klasy neutralnej, co przy ekstrapolacji poza znany zakres cech prowadziło do błędnej klasyfikacji.

Problem rozwiązano poprzez zmodyfikowanie logiki decyzyjnej tak, by wynik klasyfikatora ML nie przesłaniał jednoznacznie wykrytego sygnału otwarcia ust przekraczającego próg skoku.

---

## 6. Testy systemowe

### 6.1 Weryfikacja symulacji klawiszy

Z uwagi na niewielką aktywność innych użytkowników na platformie blobby-online.com w trakcie prac deweloperskich, przygotowano niezależny test (tools/test_key_controller.py) weryfikujący, czy moduł key_controller.py faktycznie generuje zdarzenia klawiatury na poziomie systemu operacyjnego, bez konieczności użycia kamery, drugiej osoby czy uruchomionej rozgrywki. Narzędzie pozostaje częścią repozytorium jako pomoc diagnostyczna — pozwala szybko zweryfikować, czy ewentualny brak reakcji gry wynika z błędu w kodzie Pythona, czy z problemu na poziomie samej strony przeglądarkowej.

Test wykorzystuje niezależny obiekt `pynput.keyboard.Listener`, działający jako "świadek" zewnętrzny — nasłuchujący wszystkich zdarzeń klawiatury w systemie, niezależnie od testowanego modułu. Wywołując `KeyController.update()` z symulowanymi sekwencjami akcji (ruch, skok, bonus), zweryfikowano, że:

- Klawisze ruchu (lewo/prawo) są wciskane i zwalniane zgodnie z trwaniem gestu,
- Klawisz skoku generuje jedną parę naciśnięcie/zwolnienie (tap) niezależnie od liczby klatek, w których gest jest utrzymywany — potwierdzając poprawność mechanizmu edge-detection,
- Klawisz bonusu jest wciskany na czas trwania gestu specjalnego.

Na systemie macOS test wymagał uprzedniego przyznania uprawnień dostępu (Accessibility) terminalowi, z którego uruchamiany był skrypt — bez tego uprawnienia biblioteka `pynput` nie generuje rzeczywistych zdarzeń systemowych, mimo braku błędów w kodzie. Po przyznaniu uprawnień test potwierdził prawidłowe działanie całego potoku symulacji klawiszy na poziomie systemowym, niezależnie od ewentualnych problemów z reakcją samej strony przeglądarkowej gry.

### 6.2 Test integracyjny z grą

[...]

---

## 7. Wnioski

Zaprojektowany system spełnia wymagania formalne projektu: wykorzystuje bibliotekę MediaPipe do detekcji w czasie rzeczywistym, rozróżnia podstawowe akcje (ruch, skok, stan neutralny) za pomocą reguł geometrycznych, a aktywacja bonusu wynika z samodzielnie wytrenowanego klasyfikatora ML (SVM) na własnym zbiorze danych liczącym 800 próbek, przekraczając wymagany próg precyzji 85% (uzyskano 95.49% w walidacji krzyżowej).

Istotnym elementem procesu było wykrycie i skorygowanie kolizji między regułową detekcją skoku a predykcją klasyfikatora ML przy ekstremalnych wartościach otwarcia ust. Problem ten ujawnił się dopiero podczas testów na żywo, z rzeczywistą twarzą przed kamerą — co podkreśla znaczenie testowania interaktywnego systemów HCI w warunkach zbliżonych do rzeczywistego użycia, niezależnie od formalnie satysfakcjonujących metryk modelu uzyskanych w walidacji offline.

---

## Dodatki

### Bibliografia / źródła

- Google MediaPipe Documentation — FaceLandmarker (Tasks API)
- Soukupová, T., Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* — źródło definicji Eye Aspect Ratio (EAR)