"""
model_benchmark.py
------------------
Utility for evaluating multiple sklearn-compatible models on tabular data.
Supports both classification and regression tasks.

Usage:
    from model_benchmark import benchmark_models

    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(),
    }

    results_df = benchmark_models(
        models=models,
        X=X_train,
        y=y_train,
        task="classification",   # or "regression"
        cv=5,
        X_test=X_test,           # optional: score on a held-out test set too
        y_test=y_test,
    )
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    # Classification
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss,
    # Regression
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

CLASSIFICATION_METRICS = {
    "accuracy":  "accuracy",
    "roc_auc":   "roc_auc",
    "f1":        "f1_weighted",
    "precision": "precision_weighted",
    "recall":    "recall_weighted",
    "neg_log_loss": "neg_log_loss",
}

REGRESSION_METRICS = {
    "r2":       "r2",
    "neg_mae":  "neg_mean_absolute_error",
    "neg_rmse": "neg_root_mean_squared_error",
}


def _test_set_scores(model, X_test, y_test, task: str) -> dict:
    """Compute scores on a held-out test set after fitting."""
    scores = {}
    y_pred = model.predict(X_test)

    if task == "classification":
        scores["test_accuracy"]  = accuracy_score(y_test, y_pred)
        scores["test_f1"]        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        scores["test_precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        scores["test_recall"]    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            try:
                scores["test_roc_auc"] = roc_auc_score(
                    y_test, y_prob,
                    multi_class="ovr", average="weighted"
                )
            except ValueError:
                scores["test_roc_auc"] = np.nan
            try:
                scores["test_neg_log_loss"] = -log_loss(y_test, y_prob)
            except ValueError:
                scores["test_neg_log_loss"] = np.nan
    else:
        scores["test_r2"]       = r2_score(y_test, y_pred)
        scores["test_neg_mae"]  = -mean_absolute_error(y_test, y_pred)
        scores["test_neg_rmse"] = -root_mean_squared_error(y_test, y_pred)

    return scores


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def benchmark_models(
    models: dict,
    X,
    y,
    task: str = "classification",
    cv: int = 5,
    X_test=None,
    y_test=None,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train and evaluate multiple models using cross-validation.

    Parameters
    ----------
    models : dict
        Mapping of model name -> instantiated (unfitted) sklearn-compatible estimator.
        Example: {"XGBoost": XGBClassifier(), "LightGBM": LGBMClassifier()}

    X : array-like of shape (n_samples, n_features)
        Training features.

    y : array-like of shape (n_samples,)
        Target variable.

    task : {"classification", "regression"}
        Type of ML task. Determines which metrics are computed.

    cv : int, default=5
        Number of cross-validation folds.

    X_test : array-like, optional
        Held-out test features. If provided (along with y_test), the model is
        refitted on all of X and scored on the test set.

    y_test : array-like, optional
        Held-out test labels / values.

    random_state : int, default=42
        Seed for the CV splitter.

    n_jobs : int, default=-1
        Number of parallel jobs for cross_validate (-1 = all CPUs).

    verbose : bool, default=True
        Print progress and a summary table.

    Returns
    -------
    pd.DataFrame
        One row per model with CV mean/std for every metric, fit time,
        and (optionally) test-set scores.
    """
    if task not in ("classification", "regression"):
        raise ValueError("`task` must be 'classification' or 'regression'.")

    has_test = (X_test is not None) and (y_test is not None)

    # Choose CV strategy and metrics
    if task == "classification":
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scoring = CLASSIFICATION_METRICS
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        scoring = REGRESSION_METRICS

    records = []

    for name, model in models.items():
        if verbose:
            print(f"  ▶ Training {name}...", end=" ", flush=True)

        start = time.perf_counter()

        # Cross-validation
        try:
            cv_results = cross_validate(
                estimator=model,
                X=X,
                y=y,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                return_train_score=False,
                error_score="raise",
            )
        except Exception as exc:
            if verbose:
                print(f"FAILED ({exc})")
            records.append({"model": name, "error": str(exc)})
            continue

        elapsed = time.perf_counter() - start

        # Build record from CV scores
        record = {"model": name}
        for metric_key in scoring:
            col = f"test_{metric_key}"  # cross_validate prefix
            values = cv_results[col]
            record[f"cv_{metric_key}_mean"] = round(float(np.mean(values)), 6)
            record[f"cv_{metric_key}_std"]  = round(float(np.std(values)),  6)

        record["cv_fit_time_mean_s"] = round(float(np.mean(cv_results["fit_time"])), 4)
        record["total_time_s"]       = round(elapsed, 4)

        # Optional: refit on all data and score on held-out test set
        if has_test:
            model.fit(X, y)
            test_scores = _test_set_scores(model, X_test, y_test, task)
            for k, v in test_scores.items():
                record[k] = round(float(v), 6) if not np.isnan(v) else np.nan

        records.append(record)

        if verbose:
            key_metric = "cv_roc_auc_mean" if task == "classification" else "cv_r2_mean"
            val = record.get(key_metric, "n/a")
            print(f"done  ({key_metric}={val}, {elapsed:.1f}s)")

    results_df = pd.DataFrame(records).set_index("model")

    if verbose:
        _print_summary(results_df, task)

    return results_df


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, task: str) -> None:
    """Print a ranked summary table to stdout."""
    sort_col = "cv_roc_auc_mean" if task == "classification" else "cv_r2_mean"
    sort_col = sort_col if sort_col in df.columns else df.columns[0]

    display_cols = [c for c in df.columns if c.endswith("_mean") or c in ("total_time_s",)]
    summary = df[display_cols].sort_values(sort_col, ascending=False)

    sep = "-" * 80
    print(f"\n{sep}")
    print(f"  BENCHMARK RESULTS  ({task.upper()})")
    print(sep)
    print(summary.to_string())
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.dummy import DummyClassifier

    print("=" * 60)
    print("CLASSIFICATION DEMO")
    print("=" * 60)

    X_c, y_c = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train_c, X_test_c = X_c[:800], X_c[800:]
    y_train_c, y_test_c = y_c[:800], y_c[800:]

    clf_models = {
        "Dummy (baseline)":       DummyClassifier(strategy="most_frequent"),
        "LogisticRegression":     LogisticRegression(max_iter=1000),
        "RandomForest":           RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting":       GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    clf_results = benchmark_models(
        models=clf_models,
        X=X_train_c,
        y=y_train_c,
        task="classification",
        cv=5,
        X_test=X_test_c,
        y_test=y_test_c,
    )

    print("\n" + "=" * 60)
    print("REGRESSION DEMO")
    print("=" * 60)

    X_r, y_r = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train_r, X_test_r = X_r[:800], X_r[800:]
    y_train_r, y_test_r = y_r[:800], y_r[800:]

    reg_models = {
        "Ridge":         Ridge(),
        "RandomForest":  RandomForestRegressor(n_estimators=100, random_state=42),
    }

    reg_results = benchmark_models(
        models=reg_models,
        X=X_train_r,
        y=y_train_r,
        task="regression",
        cv=5,
        X_test=X_test_r,
        y_test=y_test_r,
    )