"""Gradient boosting (XGBoost) models and the SMOTE-NC resampling pipeline.

* the number of trees is *frozen* with early stopping inside the
  cross-validation folds (median best iteration x 1.1), so that the final
  model never peeks at the evaluation fold;
* class imbalance is handled either with ``scale_pos_weight`` or with
  SMOTE-NC oversampling — the latter placed inside an ``imblearn`` pipeline,
  hence applied to the training folds only;
* a deliberately *wrong* SMOTE pipeline is kept as a teaching example of
  validation leakage (:func:`naive_smote_demo`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .. import config as C
from ..evaluation import compute_metrics, find_optimal_threshold, make_cv
from ..features import AgronomicFeatureEngineer, MacroAreaMapper, MoistureImputer, RegionAltitudeImputer
from ..preprocessing import build_encoder, build_feature_pipeline, build_model_pipeline

# Conservative configuration for ~80 positives: shallow trees, small learning
# rate, row/column subsampling.
XGB_PARAMS = dict(
    max_depth=3,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=C.SEED,
    n_jobs=1,
)


def imbalance_ratio(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def freeze_n_estimators(X_train, y_train, *, engineer: bool = False, max_rounds: int = 4000,
                        early_stopping_rounds: int = 50, cv=None, params: dict | None = None) -> tuple[int, list[int]]:
    """Choose ``n_estimators`` with early stopping inside the CV folds.

    In each outer fold the training part is split again into a fit set and an
    early-stopping set; the outer validation fold is never touched. The final
    number of trees is the median of the per-fold best iterations, inflated
    by 10% because the final model sees more data.
    """
    cv = cv or make_cv()
    params = {**XGB_PARAMS, **(params or {})}
    best_iterations = []
    for train_idx, _ in cv.split(X_train, y_train):
        X_fold, y_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fit, X_es, y_fit, y_es = train_test_split(X_fold, y_fold, test_size=0.2, random_state=C.SEED, stratify=y_fold)
        features = build_feature_pipeline(engineer=engineer, scale=False).fit(X_fit, y_fit)
        model = XGBClassifier(
            n_estimators=max_rounds,
            scale_pos_weight=imbalance_ratio(y_fit),
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )
        model.fit(features.transform(X_fit), y_fit, eval_set=[(features.transform(X_es), y_es)], verbose=False)
        best_iterations.append(int(model.best_iteration) + 1)
    n_trees = int(np.median(best_iterations) * 1.1)
    return n_trees, best_iterations


def build_xgboost(n_estimators: int, scale_pos_weight: float, *, engineer: bool = False, **overrides):
    params = {**XGB_PARAMS, **overrides}
    clf = XGBClassifier(n_estimators=n_estimators, scale_pos_weight=scale_pos_weight, **params)
    return build_model_pipeline(clf, engineer=engineer, scale=False)


def build_xgboost_grid(scale_pos_weight: float, *, engineer: bool = False) -> GridSearchCV:
    """Small grid over depth, learning rate, number of trees and class weight."""
    clf = XGBClassifier(**{**XGB_PARAMS, "n_estimators": 300, "scale_pos_weight": scale_pos_weight})
    pipe = build_model_pipeline(clf, engineer=engineer, scale=False)
    grid = {
        "classifier__max_depth": [3, 6],
        "classifier__learning_rate": [0.02, 0.05],
        "classifier__n_estimators": [150, 300],
        "classifier__scale_pos_weight": [0.5 * scale_pos_weight, scale_pos_weight, 2.0 * scale_pos_weight],
    }
    inner = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED + 81)
    return GridSearchCV(pipe, grid, cv=inner, scoring="average_precision", n_jobs=1)


def build_smotenc_xgboost(n_estimators: int, *, engineer: bool = False, **overrides) -> ImbPipeline:
    """XGBoost trained on SMOTE-NC-balanced folds.

    SMOTE-NC interpolates the numeric columns and copies the majority category
    of the neighbours for categorical ones, so it must run *before* one-hot
    encoding and *after* the imputation / macro-area steps. Being a sampler
    inside an ``imblearn`` pipeline it is applied only when ``fit`` is called,
    i.e. on training folds, never on validation or test data.
    """
    numeric = AgronomicFeatureEngineer.output_numeric_features() if engineer else list(C.NUMERIC_MODEL_FEATURES)
    steps = [("altitude", RegionAltitudeImputer()), ("moisture", MoistureImputer()), ("macro_area", MacroAreaMapper())]
    if engineer:
        steps.append(("engineer", AgronomicFeatureEngineer()))
    steps += [
        ("smote", SMOTENC(categorical_features=list(C.CATEGORICAL_FEATURES), random_state=C.SEED)),
        ("encode", build_encoder(numeric, list(C.CATEGORICAL_FEATURES), scale=False)),
        ("classifier", XGBClassifier(n_estimators=n_estimators, scale_pos_weight=1.0, **{**XGB_PARAMS, **overrides})),
    ]
    return ImbPipeline(steps)


def naive_smote_demo(X_train, y_train, X_test, y_test, n_estimators: int) -> tuple[pd.DataFrame, dict]:
    """The *wrong* way, reproduced on purpose.

    1. one-hot encode, then oversample the **whole** training split with
       plain SMOTE (synthetic decimals on dummy columns);
    2. pick the threshold on the **resampled training** predictions.

    Training-set metrics look spectacular and the test threshold is
    mis-calibrated. Compare with the SMOTE-NC-in-CV pipeline.
    """
    features = build_feature_pipeline(scale=False).fit(X_train, y_train)
    X_tr, X_te = features.transform(X_train), features.transform(X_test)
    X_res, y_res = SMOTE(random_state=C.SEED).fit_resample(X_tr, y_train)
    model = XGBClassifier(n_estimators=n_estimators, scale_pos_weight=1.0, **XGB_PARAMS).fit(X_res, y_res)
    train_proba = model.predict_proba(X_res)[:, 1]
    threshold, _ = find_optimal_threshold(y_res, train_proba)
    train_metrics = compute_metrics(y_res, train_proba, threshold)
    test_proba = model.predict_proba(X_te)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold)
    table = pd.DataFrame(
        [train_metrics, test_metrics],
        index=["resampled training set (what the notebook looked at)", "untouched test set (reality)"],
    )
    return table, {"threshold": threshold, "test_proba": test_proba}
