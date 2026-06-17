"""
train_model.py — trening i walidacja klasyfikatora dla gestu SPECIAL (wink).

Pipeline:
  1. Wczytaj data/dataset.npz
  2. Normalizacja cech (StandardScaler)
  3. StratifiedKFold(5) — cross-validation
  4. Trening SVM (rbf) + MLP jako backup
  5. Raport: accuracy, precision, recall, F1, confusion matrix
  6. Zapis najlepszego modelu → ml/model.joblib

Wymaganie: precision klasy SPECIAL > 85%
"""

import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.npz")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
MIN_SAMPLES = 50
TARGET_PRECISION = 0.85
CLASS_NAMES = ["idle", "special"]


def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset nie znaleziony: {path}\n"
            "Uruchom najpierw: python ml/collect_dataset.py"
        )
    data = np.load(path)
    X, y = data["X"], data["y"]
    print(f"[Train] Dataset: {X.shape[0]} próbek, {X.shape[1]} cech")
    for cls in [0, 1]:
        count = np.sum(y == cls)
        print(f"  Klasa {cls} ({CLASS_NAMES[cls]}): {count} próbek")
        if count < MIN_SAMPLES:
            print(f"  UWAGA: za mało próbek! (min. {MIN_SAMPLES})")
    return X, y


def build_pipelines():
    """Definiuje oba modele jako Pipeline (scaler + classifier)."""
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    class_weight="balanced", probability=True)),
    ])
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )),
    ])
    return {"SVM (rbf)": svm, "MLP (64-32)": mlp}


def cross_validate_model(name: str, pipeline, X, y, n_splits: int = 5):
    """Przeprowadza StratifiedKFold cross-validation i drukuje raport."""
    print(f"\n{'='*50}")
    print(f"  Model: {name}  |  StratifiedKFold(k={n_splits})")
    print(f"{'='*50}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        return_train_score=False,
    )

    acc = scores["test_accuracy"]
    prec = scores["test_precision_macro"]
    rec = scores["test_recall_macro"]
    f1 = scores["test_f1_macro"]

    print(f"  Accuracy : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  Precision: {prec.mean():.4f} ± {prec.std():.4f}  (cel: >{TARGET_PRECISION:.0%})")
    print(f"  Recall   : {rec.mean():.4f} ± {rec.std():.4f}")
    print(f"  F1       : {f1.mean():.4f} ± {f1.std():.4f}")

    passed = prec.mean() >= TARGET_PRECISION
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  Walidacja precyzji >85%: {status}")

    return prec.mean(), passed


def train_final_model(pipeline, X, y):
    """Trenuje model na całym datasecie i zwraca go."""
    pipeline.fit(X, y)
    return pipeline


def full_report(pipeline, X, y, name: str):
    """Pełny raport na całym datasecie (pomocniczy)."""
    y_pred = pipeline.predict(X)
    print(f"\n--- Raport końcowy ({name}) na pełnym datasecie ---")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))
    print("Confusion matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  {CLASS_NAMES[0]:8s} {CLASS_NAMES[1]:8s}  ← przewidywane")
    for i, row in enumerate(cm):
        print(f"  {row[0]:8d} {row[1]:8d}  ← rzeczywiste: {CLASS_NAMES[i]}")


def main():
    print("\n=== Trening modelu klasyfikatora MICM ===\n")
    X, y = load_dataset(DATASET_PATH)

    pipelines = build_pipelines()
    results = {}

    for name, pipeline in pipelines.items():
        prec, passed = cross_validate_model(name, pipeline, X, y)
        results[name] = (prec, passed, pipeline)

    # Wybór najlepszego modelu
    passed_models = {n: r for n, r in results.items() if r[1]}
    if passed_models:
        best_name = max(passed_models, key=lambda n: passed_models[n][0])
        print(f"\n[Train] Najlepszy model spełniający kryterium: {best_name} "
              f"(precision={results[best_name][0]:.4f})")
    else:
        print("\n[Train] UWAGA: żaden model nie osiągnął precision >85%.")
        print("         Zbierz więcej danych lub popraw dataset.")
        best_name = max(results, key=lambda n: results[n][0])
        print(f"         Zapisuję najlepszy dostępny: {best_name}")

    best_pipeline = results[best_name][2]
    final_model = train_final_model(best_pipeline, X, y)
    full_report(final_model, X, y, best_name)

    joblib.dump(final_model, MODEL_PATH)
    print(f"\n[Train] Model zapisany → {MODEL_PATH}")
    print("[Train] Gotowe! Możesz uruchomić main.py")


if __name__ == "__main__":
    main()
