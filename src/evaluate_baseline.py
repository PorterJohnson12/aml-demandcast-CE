"""
evaluate_baseline.py
Calculates all 5 metrics (MAE, RMSE, R², MAPE, MBE) for the baseline model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error
)

from src.train import FEATURE_COLS, VAL_CUTOFF, TEST_CUTOFF, TARGET, DATETIME_COL

def evaluate_baseline():
    print("Loading data...")
    DATA_PATH = Path(__file__).parent.parent / "data" / "features.parquet"
    df = pd.read_parquet(DATA_PATH)
    
    # Sort chronologically
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    
    # Split into Train and Val (just like train.py)
    df_train = df[df[DATETIME_COL] < VAL_CUTOFF]
    df_val = df[(df[DATETIME_COL] >= VAL_CUTOFF) & (df[DATETIME_COL] < TEST_CUTOFF)]
    
    X_train, y_train = df_train[FEATURE_COLS], df_train[TARGET]
    X_val, y_val = df_val[FEATURE_COLS], df_val[TARGET]
    
    print("Training Baseline Random Forest (100 trees, depth 10)...")
    baseline_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    baseline_model.fit(X_train, y_train)
    
    print("Running predictions on Validation set...")
    val_preds = baseline_model.predict(X_val)
    
    print("\n================ BASELINE METRICS ================")
    print(f"Baseline MAE:  {mean_absolute_error(y_val, val_preds):.4f}")
    print(f"Baseline RMSE: {root_mean_squared_error(y_val, val_preds):.4f}")
    print(f"Baseline R²:   {r2_score(y_val, val_preds):.4f}")
    print(f"Baseline MAPE: {mean_absolute_percentage_error(y_val, val_preds):.4f}")
    print(f"Baseline MBE:  {float(np.mean(val_preds - y_val)):.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    evaluate_baseline()