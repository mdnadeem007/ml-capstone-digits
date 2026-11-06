"""Error analysis: where does the chosen model go wrong, and why?

Usage:  python error_analysis.py      (run train.py first)

Cases are studied on the TRAINING data using out-of-fold predictions: every
training image is predicted by a model that never saw it. This gives about 30
mistakes to study instead of 9, and the test set stays untouched.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import (MODEL_PATH, PIXEL_COLS, RANDOM_STATE, REPORTS, load_raw,
                    split_data)


def main():
    df = load_raw()
    X_train, X_test, y_train, y_test = split_data(df)
    model = joblib.load(MODEL_PATH)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- 1. Out-of-fold predictions on the training data ----
    pred = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict")
    proba = cross_val_predict(clone(model), X_train, y_train, cv=cv,
                              method="predict_proba")
    # Confidence = the probability the model gave to the digit it actually predicted.
    # (For an SVM the predicted digit and the highest probability can occasionally differ.)
    confidence = proba[np.arange(len(pred)), pred]
    wrong = pred != y_train.to_numpy()
    print(f"Out-of-fold accuracy: {1 - wrong.mean():.4f} "
          f"({wrong.sum()} mistakes in {len(y_train)} images)")

    errors = pd.DataFrame({
        "row_in_raw_csv": y_train.index[wrong],
        "true_digit": y_train.to_numpy()[wrong],
        "predicted_digit": pred[wrong],
        "confidence": confidence[wrong].round(3),
    }).sort_values("confidence", ascending=False)
    errors.to_csv(REPORTS / "oof_errors.csv", index=False)

    # ---- 2. Which digits get confused? ----
    pairs = (errors.groupby(["true_digit", "predicted_digit"]).size()
             .sort_values(ascending=False).head(6))
    print("\nMost common confusions (true -> predicted):")
    for (t, p), n in pairs.items():
        print(f"  {t} -> {p}: {n} times")

    per_class = pd.DataFrame(classification_report(
        y_train, pred, output_dict=True)).T.iloc[:10][["precision", "recall", "f1-score"]]
    per_class = per_class.round(3)
    per_class.index = [f"digit {i}" for i in range(10)]
    per_class.to_csv(REPORTS / "per_class_scores_oof.csv")
    weakest = per_class["recall"].nsmallest(3)
    print("\nWeakest digits by recall:")
    print(weakest.to_string())

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_train, pred, ax=ax, colorbar=False)
    ax.set_title("Out-of-fold confusion matrix (training data)")
    fig.tight_layout()
    fig.savefig(REPORTS / "oof_confusion_matrix.png", dpi=120)

    # ---- 3. How confident is the model when it is wrong? ----
    right_conf = confidence[~wrong]
    wrong_conf = confidence[wrong]
    print(f"\nMedian confidence when right: {np.median(right_conf):.3f}")
    print(f"Median confidence when wrong: {np.median(wrong_conf):.3f}")
    low = confidence < 0.8
    print(f"Images below 0.8 confidence: {low.sum()} "
          f"(contain {(wrong & low).sum()} of the {wrong.sum()} mistakes)")

    confident = ~low
    print(f"Accuracy on the {confident.sum()} confident images (>= 0.8): "
          f"{1 - (wrong & confident).sum() / confident.sum():.4f}")

    # ---- 4. Do mistakes look different? (ink amount) ----
    ink = X_train.sum(axis=1).to_numpy()
    print(f"\nAverage ink (sum of pixels): mistakes {ink[wrong].mean():.0f} "
          f"vs correct {ink[~wrong].mean():.0f}")

    # ---- 5. Picture the mistakes ----
    worst = errors.head(20)
    n_rows = max(1, int(np.ceil(len(worst) / 5)))
    fig, axes = plt.subplots(n_rows, 5, figsize=(10, 2.2 * n_rows + 0.8), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
        ax.imshow(df.loc[int(r.row_in_raw_csv), PIXEL_COLS].to_numpy(dtype=float)
                  .reshape(8, 8), cmap="gray_r")
        ax.set_title(f"true {int(r.true_digit)}, said {int(r.predicted_digit)}\n"
                     f"conf {r.confidence:.2f}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(worst):]:
        ax.axis("off")
    fig.suptitle("Mistakes, most confident first (out-of-fold, training data)")
    fig.tight_layout()
    fig.savefig(REPORTS / "misclassified_examples.png", dpi=120)

    # ---- 6. Stress test: what if the digit is not perfectly centred? ----
    # Fit on the training data as usual, then score slightly changed test images.
    fitted = joblib.load(MODEL_PATH)

    def score(images):
        return accuracy_score(y_test, fitted.predict(
            pd.DataFrame(images.reshape(len(images), 64), columns=PIXEL_COLS)))

    imgs = X_test.to_numpy(dtype=float).reshape(-1, 8, 8)
    stress = {"original test images": score(imgs)}
    stress["shifted 1 pixel right"] = score(
        np.stack([ndimage.shift(i, (0, 1), order=0) for i in imgs]))
    stress["shifted 1 pixel down"] = score(
        np.stack([ndimage.shift(i, (1, 0), order=0) for i in imgs]))
    stress["rotated 15 degrees"] = score(
        np.stack([ndimage.rotate(i, 15, reshape=False, order=1) for i in imgs]))
    stress["with random pixel noise (sd 2)"] = score(np.clip(
        imgs + np.random.default_rng(RANDOM_STATE).normal(0, 2, imgs.shape), 0, 16))
    stress_df = pd.DataFrame({"test_accuracy": stress}).round(3)
    stress_df.to_csv(REPORTS / "stress_test.csv")
    print("\nStress test (test accuracy):")
    print(stress_df.to_string())

    summary = {
        "oof_accuracy": round(1 - wrong.mean(), 4),
        "oof_mistakes": int(wrong.sum()),
        "median_conf_right": round(float(np.median(right_conf)), 3),
        "median_conf_wrong": round(float(np.median(wrong_conf)), 3),
        "mistakes_below_0.8_conf": int((wrong & low).sum()),
        "images_below_0.8_conf": int(low.sum()),
        "accuracy_on_confident_images": round(
            1 - float((wrong & confident).sum()) / float(confident.sum()), 4),
        "top_confusions": {f"{t}->{p}": int(n) for (t, p), n in pairs.items()},
        "weakest_digits_recall": {k: float(v) for k, v in weakest.items()},
    }
    (REPORTS / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved plots and tables to reports/")


if __name__ == "__main__":
    main()
