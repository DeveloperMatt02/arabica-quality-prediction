"""Model interpretation: encoded design matrices, XGBoost gain/weight
importance, SHAP values and out-of-fold permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from . import config as C
from .evaluation import make_cv
from .preprocessing import clean_name


def _estimator(model):
    return getattr(model, "best_estimator_", model)


def transform_to_matrix(model, X: pd.DataFrame) -> pd.DataFrame:
    """Push raw predictors through every non-sampler step but the classifier."""
    est = _estimator(model)
    out = X
    for name, step in est.steps[:-1]:
        if hasattr(step, "fit_resample") and not hasattr(step, "transform"):
            continue  # samplers (SMOTE-NC) are only active at fit time
        out = step.transform(out)
    return pd.DataFrame(out)  # column names are the raw encoder names (sklearn checks them at predict time)


def classifier_of(model):
    return _estimator(model).named_steps["classifier"]


# --------------------------------------------------------------------------- #
# XGBoost native importances
# --------------------------------------------------------------------------- #
def xgb_importance(model) -> pd.DataFrame:
    """Gain and weight (split count) importance of a fitted XGBoost pipeline."""
    booster = classifier_of(model).get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    feats = sorted(set(gain) | set(weight))
    table = pd.DataFrame(
        {"feature": [clean_name(f) for f in feats],
         "gain": [gain.get(f, 0.0) for f in feats],
         "weight": [weight.get(f, 0.0) for f in feats]}
    )
    return table.sort_values("gain", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def shap_values(model, X: pd.DataFrame):
    """TreeExplainer SHAP values of a fitted tree-based pipeline on ``X``."""
    import shap

    matrix = transform_to_matrix(model, X)
    explainer = shap.TreeExplainer(classifier_of(model))
    values = explainer(matrix)
    if values.values.ndim == 3:  # some versions return (n, p, 2) for binary
        values = values[:, :, 1]
    values.feature_names = [clean_name(c) for c in matrix.columns]
    matrix = matrix.rename(columns=clean_name)
    return values, matrix


def shap_summary_table(values) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted."""
    mean_abs = np.abs(values.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": values.feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Permutation importance, out-of-fold
# --------------------------------------------------------------------------- #
def cv_permutation_importance(model, X_train, y_train, *, cv=None, n_repeats: int = 20,
                              scoring: str = "average_precision", seed: int = C.SEED) -> pd.DataFrame:
    """Permutation importance of the *encoded* columns, averaged over CV folds.

    For each fold the pipeline is fitted on the training part and the
    importance is measured on the held-out part (never on the test split, as
    in the original notebook). Columns are permuted after encoding so that
    every dummy gets its own score; the ``raw_feature`` column aggregates
    them back to the original variable.
    """
    cv = cv or make_cv()
    scores: dict[str, list[float]] = {}
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        fitted = clone(model).fit(X_tr, y_tr)
        matrix = transform_to_matrix(fitted, X_va)
        result = permutation_importance(
            classifier_of(fitted), matrix, y_va, scoring=scoring, n_repeats=n_repeats,
            random_state=seed, n_jobs=1,
        )
        for name, mean in zip(matrix.columns, result.importances_mean):
            scores.setdefault(clean_name(name), []).append(float(mean))
    table = pd.DataFrame(
        {"feature": list(scores), "importance_mean": [np.mean(v) for v in scores.values()],
         "importance_std": [np.std(v) for v in scores.values()], "n_folds": [len(v) for v in scores.values()]}
    )
    table["raw_feature"] = table["feature"].map(_raw_feature)
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _raw_feature(encoded: str) -> str:
    for cat in C.CATEGORICAL_FEATURES:
        if encoded.startswith(cat + "_"):
            return cat
    return encoded
