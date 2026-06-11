"""
export_scaler.py  –  run this ONCE from your training notebook environment.

It fits a StandardScaler on the same feature matrix used during model training
and saves it to artifacts/scaler.joblib so the live server can standardise
incoming feature vectors identically to how training data was standardised.

Usage:
    python export_scaler.py \
        --csv /path/to/training_features.csv \
        --out  ./artifacts/scaler.joblib

The CSV must have exactly the 41 feature columns in FEATURE_COLS order
(plus any extra metadata columns like track_id, window_idx which are ignored).
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")          # so we can import constants
from constants import FEATURE_COLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to training feature CSV")
    parser.add_argument("--out", default="./artifacts/scaler.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    X = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    print(f"Fitting StandardScaler on {X.shape[0]} windows × {X.shape[1]} features …")

    scaler = StandardScaler()
    scaler.fit(X)

    joblib.dump(scaler, args.out)
    print(f"Saved scaler → {args.out}")
    print(f"  mean range : [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    print(f"  scale range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")


if __name__ == "__main__":
    main()
