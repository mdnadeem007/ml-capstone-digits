"""Train step: raw CSV -> checked data -> split -> compare models -> final model -> saved files.

Usage:  python train.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # draw plots without needing a screen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from common import (MODEL_PATH, PIXEL_COLS, RANDOM_STATE, REPORTS, ROOT,
                    load_raw, scale_pixels, split_data)


def candidate_models():
    """Every candidate gets the same data.

    Pixels are divided by 16 (fixed scaling). One extra model uses
    StandardScaler so the choice between the two scalings is visible in the results.
    """
    fixed = lambda: FunctionTransformer(scale_pixels)
    return {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(fixed(), LogisticRegression(max_iter=2000)),
        "KNN (k=5)": make_pipeline(fixed(), KNeighborsClassifier()),
        "SVM (RBF)": make_pipeline(
            fixed(), SVC(probability=True, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE),
        "SVM (RBF) with StandardScaler": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # 1. Load and validate the raw data
    df = load_raw()
    print(f"Loaded {len(df)} rows, {len(PIXEL_COLS)} pixel columns, 10 classes")

    # 2. Split first. The test set stays untouched until step 4.
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # 3. Compare candidates with 5-fold cross-validation on the training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = candidate_models()
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X_train, y_train, cv=cv,
                             scoring=["accuracy", "f1_macro"])
        rows.append({
            "model": name,
            "cv_accuracy_mean": res["test_accuracy"].mean(),
            "cv_accuracy_std": res["test_accuracy"].std(),
            "cv_f1_macro_mean": res["test_f1_macro"].mean(),
        })
        print(f"  {name:26s} CV accuracy = {rows[-1]['cv_accuracy_mean']:.3f}"
              f" +/- {rows[-1]['cv_accuracy_std']:.3f}")
    cv_table = pd.DataFrame(rows).round(4)
    cv_table.to_csv(REPORTS / "cv_results.csv", index=False)

    best_name = cv_table.loc[cv_table["cv_accuracy_mean"].idxmax(), "model"]
    print(f"\nChosen by cross-validation: {best_name}")

    # 4. Fit the chosen model on all training data and score the test set once
    final_model = models[best_name].fit(X_train, y_train)
    pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, pred)
    test_f1 = f1_score(y_test, pred, average="macro")
    print(f"Test accuracy: {test_acc:.4f} | Test macro F1: {test_f1:.4f} "
          f"| Wrong: {(pred != y_test).sum()} of {len(y_test)}")

    # 5. Save everything the other scripts and the README need
    joblib.dump(final_model, MODEL_PATH)
    metrics = {
        "chosen_model": best_name,
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "test_rows": int(len(y_test)),
        "test_errors": int((pred != y_test).sum()),
        "baseline_cv_accuracy": float(
            cv_table.loc[cv_table.model == "Baseline (most frequent)",
                         "cv_accuracy_mean"].iloc[0]),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS / "test_classification_report.txt").write_text(
        classification_report(y_test, pred, digits=3))

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    ax.set_title(f"Test set confusion matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(REPORTS / "test_confusion_matrix.png", dpi=120)

    # A small unlabeled sample so predict.py can be tried straight away
    X_test.head(10).to_csv(ROOT / "data" / "sample_input.csv", index=False)
    print("\nSaved model, metrics, plots and data/sample_input.csv")


if __name__ == "__main__":
    main()
