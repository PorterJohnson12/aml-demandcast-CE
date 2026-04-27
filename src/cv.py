"""
cv.py — Time-series cross-validation for DemandCast
=====================================================
This script evaluates a trained model using TimeSeriesSplit cross-validation.
Unlike standard k-fold CV, TimeSeriesSplit always trains on the past and tests
on the future — preserving the temporal ordering of the data.
"""

import sys
from pathlib import Path

# Add the project root to the Python path so we can import from src.train
sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from typing import Any

# Import exactly as defined in our working train.py
from src.train import FEATURE_COLS, VAL_CUTOFF

# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "DemandCast"

DATA_PATH    = Path(__file__).parent.parent / "data" / "features.parquet"
DATETIME_COL = "pickup_hour"
TARGET       = "demand"

# CV runs only on train+val — test set stays sealed
TEST_CUTOFF  = "2025-01-29"   


# ---------------------------------------------------------------------------
# time_series_cv()
# ---------------------------------------------------------------------------

def time_series_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    run_name: str = "cv_run",
) -> pd.DataFrame:
    """Evaluate a model using time-series cross-validation and log results to MLflow."""
    
    # --- 1. Set up MLflow and TimeSeriesSplit ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # --- 2. Open a single MLflow run for the entire CV study ---
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model", type(model).__name__)
        mlflow.log_param("n_splits", n_splits)

        results = []

        # --- 3. Fold loop (inside the with block) ---
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
            y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

            # Clone creates a fresh, unfitted copy — never skip this
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            preds = fold_model.predict(X_fold_test)

            fold_metrics = {
                "fold": fold,
                "mae":  round(mean_absolute_error(y_fold_test, preds), 4),
                "rmse": round(root_mean_squared_error(y_fold_test, preds), 4),
                "r2":   round(r2_score(y_fold_test, preds), 4),
                "mbe":  round(float(np.mean(preds - y_fold_test)), 4),
            }
            results.append(fold_metrics)

            # Log per-fold metrics with the fold number as a step
            mlflow.log_metrics({
                "mae": fold_metrics["mae"],
                "rmse": fold_metrics["rmse"],
                "r2": fold_metrics["r2"],
                "mbe": fold_metrics["mbe"],
            }, step=fold)

            print(f"  Fold {fold}: MAE={fold_metrics['mae']:.2f}  "
                  f"RMSE={fold_metrics['rmse']:.2f}  R²={fold_metrics['r2']:.3f}  MBE={fold_metrics['mbe']:.2f}")

        # --- 4. Log summary metrics and return results ---
        results_df = pd.DataFrame(results)
        mlflow.log_metrics({
            "cv_mae_mean": round(results_df["mae"].mean(), 4),
            "cv_mae_std":  round(results_df["mae"].std(), 4),
            "cv_rmse_mean": round(results_df["rmse"].mean(), 4),
            "cv_r2_mean":  round(results_df["r2"].mean(), 4),
            "cv_mbe_mean": round(results_df["mbe"].mean(), 4),
        })
        
        return results_df

# ---------------------------------------------------------------------------
# Main — Execute Time-Series CV
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    
    print("Loading data for cross-validation...")
    df = pd.read_parquet(DATA_PATH)
    
    # Sort chronologically to preserve temporal order
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    
    # CV operates on train+val, so exclude the sealed test data
    df_cv = df[df[DATETIME_COL] < TEST_CUTOFF].copy()
    
    X_cv = df_cv[FEATURE_COLS]
    y_cv = df_cv[TARGET]
    
    print(f"Running 5-fold TimeSeries CV on {len(df_cv)} rows...")
    
    # We will use Random Forest for evaluating stability since it was one of our best models
    best_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    # --- Evaluate Original Baseline on single Train/Val split to compare ---
    print("\n--- Original Validation Performance (from train.py split) ---")
    df_train_base = df_cv[df_cv[DATETIME_COL] < VAL_CUTOFF]
    df_val_base   = df_cv[df_cv[DATETIME_COL] >= VAL_CUTOFF]
    
    X_train_base, y_train_base = df_train_base[FEATURE_COLS], df_train_base[TARGET]
    X_val_base, y_val_base     = df_val_base[FEATURE_COLS], df_val_base[TARGET]
    
    baseline_model = clone(best_model)
    baseline_model.fit(X_train_base, y_train_base)
    baseline_preds = baseline_model.predict(X_val_base)
    
    print(f"  Baseline MAE:  {mean_absolute_error(y_val_base, baseline_preds):.2f}")
    print(f"  Baseline RMSE: {root_mean_squared_error(y_val_base, baseline_preds):.2f}")
    print(f"  Baseline R²:   {r2_score(y_val_base, baseline_preds):.3f}")
    print(f"  Baseline MBE:  {np.mean(baseline_preds - y_val_base):.2f}\n")
    # -----------------------------------------------------------------------
    
    results_df = time_series_cv(
        model=best_model,
        X=X_cv,
        y=y_cv,
        n_splits=5,
        run_name="cv_random_forest_100est"
    )
    
    mean_mae = results_df['mae'].mean()
    std_mae = results_df['mae'].std()
    
    mean_mbe = results_df['mbe'].mean()
    std_mbe = results_df['mbe'].std()
    
    print("\n==================================")
    print(f"CV MAE: {mean_mae:.2f} ± {std_mae:.2f}")
    print(f"CV MBE: {mean_mbe:.2f} ± {std_mbe:.2f}")
    print("==================================")
    print("Done! You can verify the run in MLflow at http://localhost:5000")

'''
Cross-Validation Interpretation:
The cross-validation resulted in a Mean Absolute Error (MAE) that ranged from a high of 13.42 (Fold 0) down to 
a low of 8.72 (Fold 4).

The standard deviation of the MAE across the 5 time-series folds measures the stability of our model's 
performance over time. In our results, the standard deviation is relatively low (roughly ~1.7), indicating 
that the Random Forest model is reasonably stable and resilient.

Additionally, because we used TimeSeriesSplit, Fold 0 trained on the smallest amount of historical data, 
while Fold 4 trained on the largest amount. The steady decrease in MAE (13.42 → 10.40 → 9.89 → 8.89 → 8.72) 
shows that as the model is given a larger historical context, it becomes significantly more accurate and stable 
at predicting future demand.
'''