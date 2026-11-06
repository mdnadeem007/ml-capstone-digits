"""Predict step: new pixel data in -> predicted digit and confidence out.

Usage:  python predict.py data/sample_input.csv
Input:  a CSV with the 64 columns pixel_0_0 ... pixel_7_7 (a 'target' column is ignored).
Output: reports/predictions.csv
"""
import sys

import joblib
import pandas as pd

from common import MODEL_PATH, PIXEL_COLS, REPORTS, validate_pixels


def main(path):
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run 'python train.py' first.")
    model = joblib.load(MODEL_PATH)

    data = pd.read_csv(path)
    validate_pixels(data)                       # same checks as training
    X = data[PIXEL_COLS]

    digits = model.predict(X)
    proba = model.predict_proba(X)
    # Confidence = probability given to the digit that was actually predicted
    confidence = proba[range(len(digits)), digits]
    out = pd.DataFrame({
        "predicted_digit": digits,
        "confidence": confidence.round(3),
    })
    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / "predictions.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {len(out)} predictions to {REPORTS / 'predictions.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_input.csv")
