"""Shared settings and helpers used by every script in this project."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "digits.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"

RANDOM_STATE = 42
# 8 x 8 image = 64 pixel columns, named pixel_<row>_<col>
PIXEL_COLS = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]


def scale_pixels(X):
    """Divide pixels by 16 so values run from 0 to 1.

    The pixel range (0 to 16) is known in advance, so nothing is learned from
    the data here. See the README for why this replaced StandardScaler.
    """
    return np.asarray(X, dtype=float) / 16.0


def validate_pixels(df):
    """Fail loudly if the pixel data is not what the model expects."""
    missing = [c for c in PIXEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing pixel columns, e.g. {missing[:3]}")
    pixels = df[PIXEL_COLS]
    if pixels.isna().any().any():
        raise ValueError("Pixel data contains missing values")
    if pixels.min().min() < 0 or pixels.max().max() > 16:
        raise ValueError("Pixel values must be between 0 and 16")


def load_raw():
    """Step 1 of the pipeline: read the raw CSV and check it."""
    df = pd.read_csv(RAW_CSV)
    validate_pixels(df)
    if "target" not in df.columns or not df["target"].between(0, 9).all():
        raise ValueError("Column 'target' must hold digits 0 to 9")
    return df


def split_data(df):
    """Step 2: split into train and test. The test set is only used for the final score."""
    X = df[PIXEL_COLS]
    y = df["target"]
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
