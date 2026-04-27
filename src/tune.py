"""
tune.py — Hyperparameter tuning for DemandCast (Optuna + MLflow)
===============================================================
Runs an Optuna study to tune a RandomForestRegressor on the train/val
split. Each trial is logged to MLflow; the best run is registered
to the MLflow Model Registry.

Run from project root with the `.venv` active:
    python src/tune.py
"""

import sys
from pathlib import Path
import datetime
from typing import Tuple, Dict, Any

# Add the project root to the Python path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Import feature configuration exactly as defined in train.py
from src.train import FEATURE_COLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"

# Construct robust absolute paths
DATA_PATH = Path(__file__).parent.parent / "data" / "features.parquet"

# Cutoffs matched to train.py to prevent data leakage
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-01-29"
DATETIME_COL = "pickup_hour"
TARGET = "demand"

N_TRIALS = 15


# ---------------------------------------------------------------------------
# 1. load_splits()
# ---------------------------------------------------------------------------
def load_splits() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load features.parquet, apply temporal splits, and return DataFrames 
    for training and validation.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please run build_features.py.")

    df = pd.read_parquet(DATA_PATH)

    # Ensure temporal ordering
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    # Apply temporal filters
    train = df[df[DATETIME_COL] < VAL_CUTOFF].copy()
    val = df[(df[DATETIME_COL] >= VAL_CUTOFF) & (df[DATETIME_COL] < TEST_CUTOFF)].copy()

    if train.empty or val.empty:
        raise ValueError("Train or Validation split is empty. Check data cutoffs.")

    # Extract strictly defined features and target 
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# 2. objective()
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function. Suggests hyperparameters, runs TimeSeriesSplit CV,
    logs metrics to MLflow, and returns the mean CV MAE to be minimized.
    """
    # --- Part 1: Search Space ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        "random_state": 42,
        "n_jobs": -1 
    }

    # --- Part 2: Setup Data & Validation ---
    X_train, y_train, X_val, y_val = load_splits()
    tscv = TimeSeriesSplit(n_splits=5)

    # --- Part 3: MLflow Logging Context ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"optuna_trial_{trial.number}_{timestamp_str}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("logged_at_utc", timestamp_str)
        mlflow.log_param("objective", "tscv_train")
        mlflow.log_params(params)

        cv_maes = []

        # Perform Time-Series Fold Validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model = RandomForestRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            
            preds = model.predict(X_fold_test)
            fold_mae = mean_absolute_error(y_fold_test, preds)
            cv_maes.append(fold_mae)
            
            mlflow.log_metric(f"fold_{fold}_mae", fold_mae, step=fold)

        # Average Cross-Validation Score
        mean_cv_mae = float(np.mean(cv_maes))
        mlflow.log_metric("mean_cv_mae", mean_cv_mae)

        # Evaluate against the fixed hold-out validation set 
        final_model = RandomForestRegressor(**params)
        final_model.fit(X_train, y_train)
        val_preds = final_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        
        mlflow.log_metric("val_mae", val_mae)
        mlflow.sklearn.log_model(final_model, "model")

    return mean_cv_mae


# ---------------------------------------------------------------------------
# 3. retrain_and_register()
# ---------------------------------------------------------------------------
def retrain_and_register(best_params: Dict[str, Any], stage: str = "Production") -> None:
    """
    Retrains the optimal model on (train + val), tests on the sealed set,
    and registers it directly to the MLflow model registry.
    """
    print("\n--- Retraining Final Model ---")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(DATA_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    # Train+Val Combined vs Sealed Test Set
    trainval = df[df[DATETIME_COL] < TEST_CUTOFF].copy()
    test = df[df[DATETIME_COL] >= TEST_CUTOFF].copy()

    if trainval.empty:
        raise ValueError("Combined Train/Val set is empty.")

    X_trainval, y_trainval = trainval[FEATURE_COLS], trainval[TARGET]
    
    if test.empty:
        print("Warning: Test set is empty. Skipping test evaluation.")
        X_test, y_test = None, None
    else:
        X_test, y_test = test[FEATURE_COLS], test[TARGET]

    # Fit final model
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_trainval, y_trainval)

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"final_retrain_and_register_{timestamp_str}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("logged_at_utc", timestamp_str)
        mlflow.log_params(best_params)

        if X_test is not None and y_test is not None:
            test_preds = best_model.predict(X_test)
            
            test_mae = mean_absolute_error(y_test, test_preds)
            test_rmse = root_mean_squared_error(y_test, test_preds)
            test_r2 = r2_score(y_test, test_preds)
            test_mape = mean_absolute_percentage_error(y_test, test_preds)
            test_mbe = float(np.mean(test_preds - y_test))
            
            mlflow.log_metrics({
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "test_mape": test_mape,
                "test_mbe": test_mbe
            })
            
            print(f"Final Test MAE:  {test_mae:.4f}")
            print(f"Final Test RMSE: {test_rmse:.4f}")
            print(f"Final Test R²:   {test_r2:.4f}")
            print(f"Final Test MAPE: {test_mape:.4f}")
            print(f"Final Test MBE:  {test_mbe:.4f}")

        # Log and Register
        mlflow.sklearn.log_model(best_model, "model")
        
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        print(f"Registering model to MLflow registry under '{MODEL_REGISTRY_NAME}'...")
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

        # Transition stage via MLflowClient
        client = MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered_model.version,
            stage=stage,
            archive_existing_versions=True
        )

        print("\n================ FINAL DEPLOYMENT INFO ================")
        print(f"Model Name:    {registered_model.name}")
        print(f"Model Version: {registered_model.version}")
        print(f"Model Stage:   {stage}")
        print(f"Run ID:        {run_id}")
        print("=======================================================")


# ---------------------------------------------------------------------------
# 4. Execution Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting Optuna hyperparameter optimization ({N_TRIALS} trials)...")
    
    # Minimize the Mean CV MAE
    study = optuna.create_study(direction="minimize", study_name="DemandCast_RF_Tuning")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n================ TUNING COMPLETE ================")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Mean CV MAE: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Automatically retrain the best parameters on the full data and register
    # Note: we merge fixed parameters (n_jobs, random_state) with Optuna's suggestions
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    
    retrain_and_register(best_params=best_params, stage="Production")