"""
tune.py — Hyperparameter tuning for DemandCast (Optuna + MLflow)
===============================================================
Runs an Optuna study to tune a RandomForestRegressor on the train/val
split. Each trial is logged to MLflow; the best run can be registered
to the MLflow Model Registry.

Run from project root with the `.venv` active:
    python tune.py
"""
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import datetime

from src.features_skeleton import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py and cv.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"
TARGET = "demand"

N_TRIALS = 15


def load_splits():
    """Load features.parquet and return train and validation splits.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    # Step 1: Raise a FileNotFoundError if DATA_PATH does not exist.

    # Step 2: Read the parquet file at DATA_PATH into a DataFrame.

    # Step 3: Parse the "hour" column as datetime using pd.to_datetime().

    # Step 4: Filter rows where "hour" < VAL_CUTOFF into a `train` DataFrame.
    #         Filter rows where VAL_CUTOFF <= "hour" < TEST_CUTOFF into a `val` DataFrame.
    #         Use pd.to_datetime() when comparing against the cutoff strings.
    #         Call .copy() on each slice to avoid SettingWithCopyWarning.

    # Step 5: Convert the "hour" column in both `train` and `val` to integer
    #         hour-of-day using the .dt.hour accessor (consistent with train.py preprocessing).

    # Step 6: Return X_train, y_train, X_val, y_val.
    #         Select features with FEATURE_COLS and the target with TARGET.
    pass


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: suggest hyperparams, run TimeSeriesSplit CV on `train`,
    log per-fold metrics to MLflow, and return the mean CV MAE (minimize).
    """
    # --- Part 1: Search space ---
    # Build a `params` dict by sampling hyperparameters from the trial:
    #   - "n_estimators":      int in [lower_limit, higher_limit], step 50
    #   - "max_depth":         int in [lower_limit, higher_limit]
    #   - "min_samples_leaf":  int in [lower_limit, higher_limit]
    #   - "min_samples_split": int in [lower_limit, higher_limit]
    #   - "max_features":      categorical choice among ["sqrt", "log2", 0.5]
    # Also fix "random_state" to 42 and "n_jobs" to -1 (not tuned).
    # Use trial.suggest_int() and trial.suggest_categorical().

    # --- Part 2: Load data and prepare for cross-validation ---
    # Call load_splits() to get X_train, y_train, X_val, y_val.
    # TimeSeriesSplit requires rows to be in chronological order.
    # Sort X_train and y_train by their DatetimeIndex (use .argsort() on the index
    # and .iloc[] to reorder both arrays consistently).

    # Step 3: Create a TimeSeriesSplit object with n_splits=5.

    # Step 4: Configure MLflow — call mlflow.set_tracking_uri() and mlflow.set_experiment().

    # Step 5: Build a unique run name using the trial number and a UTC timestamp string,
    #         e.g. "optuna_trial_<number>_<YYYYMMDDTHHMMSSz>".

    # Step 6: Start an MLflow run using mlflow.start_run(run_name=...).
    #         Inside the run context:

    #   Step 6a: Log a "logged_at_utc" param with the current UTC ISO timestamp.
    #            Log all params from the `params` dict with mlflow.log_params().
    #            Log an "objective" param with value "tscv_train".

    #   Step 6b: Iterate over the folds produced by tscv.split(X_train).
    #            For each fold:
    #              - Slice X_train and y_train with the provided train/test indices.
    #              - Instantiate and fit a RandomForestRegressor using **params.
    #              - Predict on the fold's test slice.
    #              - Compute MAE with mean_absolute_error() and append to a list.
    #              - Log the fold MAE to MLflow as "fold_<n>_mae" at step=fold number.

    #   Step 6c: Compute the mean of all fold MAEs and log it as "mean_cv_mae".

    #   Step 6d: Train a fresh RandomForestRegressor(**params) on the full X_train/y_train.
    #            Predict on X_val, compute val MAE, and log it as "val_mae".

    #   Step 6e: Log the final model artifact with mlflow.sklearn.log_model(model, "model").

    # Primary objective: mean CV MAE on train (minimize)
    # Step 7: Return the mean CV MAE so Optuna can minimize it.
    pass


def retrain_and_register(best_params: dict, stage: str = "Production") -> None:
    """Retrain the chosen hyperparameters on train+val, evaluate on test,
    log test metrics, and register the final model to the Model Registry.
    """
    # Step 1: Configure MLflow — call mlflow.set_tracking_uri() and mlflow.set_experiment().

    # Step 2: Load the full DataFrame from DATA_PATH and parse the "hour" column as datetime.

    # Step 3: Split into two DataFrames:
    #         - `trainval`: rows where "hour" < TEST_CUTOFF  (train + validation combined)
    #         - `test`:     rows where "hour" >= TEST_CUTOFF
    #         Call .copy() on each slice. Raise a ValueError if `trainval` is empty.

    # Step 4: Convert "hour" to integer hour-of-day in `trainval`.
    #         Build X_trainval (FEATURE_COLS) and y_trainval (TARGET).
    #         If `test` is not empty, do the same to get X_test and y_test.
    #         If `test` is empty, print a warning and set X_test and y_test to None.

    # Step 5: Instantiate a RandomForestRegressor(**best_params) and fit it on
    #         X_trainval / y_trainval.

    # Step 6: Build a unique run name using a UTC timestamp,
    #         e.g. "final_retrain_and_register_<YYYYMMDDTHHMMSSz>".
    #         Start an MLflow run with mlflow.start_run(run_name=...).
    #         Inside the run context:

    #   Step 6a: Log a "logged_at_utc" param with the current UTC ISO timestamp.
    #            Log all best_params with mlflow.log_params().

    #   Step 6b: If X_test is not None, predict on X_test, compute test MAE with
    #            mean_absolute_error(), log it as "test_mae", and print it.

    #   Step 6c: Log the final model artifact with mlflow.sklearn.log_model(model, "model").
    #            Register the model with mlflow.register_model() using uri
    #            "runs:/<run_id>/model" and name MODEL_REGISTRY_NAME.

    #   Step 6d: Use mlflow.tracking.MlflowClient() to transition the registered model
    #            version to the given `stage` via client.transition_model_version_stage().

    #   Step 6e: Print the registered model name, version, stage, and run ID.
    pass