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

--- Raport końcowy (SVM (rbf)) na pełnym datasecie ---
              precision    recall  f1-score   support

        idle       0.98      0.97      0.98       400
     special       0.97      0.98      0.98       400

    accuracy                           0.98       800
   macro avg       0.98      0.98      0.98       800
weighted avg       0.98      0.98      0.98       800

Confusion matrix:
  idle     special   ← przewidywane
       389       11  ← rzeczywiste: idle
         8      392  ← rzeczywiste: special

[Train] Model zapisany → /Users/annakrasodomska/Documents/studia/MICM/hci-blobby-controller/ml/model.joblib
[Train] Gotowe! Możesz uruchomić main.py
```

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