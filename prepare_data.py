"""Export the raw digits data to a plain CSV so the pipeline starts from a real file.

Run once. The CSV is already included in the repo, so you only need this
if you want to recreate it.
"""
import pandas as pd
from sklearn.datasets import load_digits

from common import PIXEL_COLS, RAW_CSV

digits = load_digits()
df = pd.DataFrame(digits.data.astype(int), columns=PIXEL_COLS)
df["target"] = digits.target

RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW_CSV, index=False)
print(f"Saved {len(df)} rows to {RAW_CSV}")
