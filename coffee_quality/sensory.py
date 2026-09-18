"""The sensory track: which cupping dimensions separate excellent coffees?

These scores are available only *after* tasting and are the very components
of ``Total.Cup.Points``, so a classifier built on them is not a predictor of
quality but a (partial) reconstruction of the target — the "sensory
paradox" discussed in docs/LESSONS_LEARNED.md. The track is kept because the
*ranking* of the dimensions is still informative for cuppers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .evaluation import make_cv


def sensory_matrix(df: pd.DataFrame, cols=None) -> tuple[pd.DataFrame, pd.Series]:
    cols = list(cols or C.SENSORY_MODEL_COLS)
    data = df[cols + [C.TARGET]].dropna()
    return data[cols], data[C.TARGET]


def build_logistic() -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000, random_state=C.SEED))])


def build_random_forest_grid() -> GridSearchCV:
    rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=C.SEED, n_jobs=1)
    grid = {"n_estimators": [300], "max_depth": [5, None], "min_samples_leaf": [2, 5]}
    return GridSearchCV(rf, grid, cv=make_cv(), scoring="average_precision", n_jobs=-1)


def cross_validated_scores(models: dict, X, y) -> pd.DataFrame:
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1", "balanced_accuracy": "balanced_accuracy"}
    rows = []
    for name, model in models.items():
        s = cross_validate(clone(model), X, y, cv=make_cv(), scoring=scoring, n_jobs=-1)
        for metric in scoring:
            rows.append({"model": name, "metric": metric, "mean": s[f"test_{metric}"].mean(), "std": s[f"test_{metric}"].std()})
    return pd.DataFrame(rows).pivot(index="metric", columns="model", values=["mean", "std"])


def fit_logit_statsmodels(X_train: pd.DataFrame, y_train: pd.Series):
    """Standardised logit (statsmodels) for p-values and odds ratios per SD."""
    scaler = StandardScaler().fit(X_train)
    Z = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    result = sm.Logit(y_train, sm.add_constant(Z)).fit(disp=0, maxiter=200)
    table = pd.DataFrame({"coef": result.params, "std_err": result.bse, "p_value": result.pvalues, "odds_ratio_per_sd": np.exp(result.params)})
    return result, table.drop(index="const")


def lasso_path_coefficients(X, y) -> pd.DataFrame:
    model = Pipeline([("scale", StandardScaler()),
                      ("model", LogisticRegressionCV(Cs=20, penalty="l1", solver="liblinear", class_weight="balanced", cv=make_cv(), scoring="average_precision", max_iter=5000, random_state=C.SEED))])
    model.fit(X, y)
    coef = model.named_steps["model"].coef_.ravel()
    return pd.DataFrame({"feature": X.columns, "lasso_coef": coef, "abs_lasso_coef": np.abs(coef)}).sort_values("abs_lasso_coef", ascending=False).reset_index(drop=True)


def oof_permutation_importance(model, X, y, n_repeats: int = 20) -> pd.DataFrame:
    scores: dict[str, list[float]] = {c: [] for c in X.columns}
    for tr, va in make_cv().split(X, y):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        est = getattr(fitted, "best_estimator_", fitted)
        r = permutation_importance(est, X.iloc[va], y.iloc[va], scoring="average_precision", n_repeats=n_repeats, random_state=C.SEED)
        for c, v in zip(X.columns, r.importances_mean):
            scores[c].append(v)
    return (pd.DataFrame({"feature": list(scores), "permutation_importance": [np.mean(v) for v in scores.values()],
                          "permutation_std": [np.std(v) for v in scores.values()]})
            .sort_values("permutation_importance", ascending=False).reset_index(drop=True))


def final_ranking(logit_table: pd.DataFrame, lasso: pd.DataFrame, gini: pd.Series, perm: pd.DataFrame,
                  mean_diff: pd.Series) -> pd.DataFrame:
    """Rank the sensory dimensions by averaging four min-max-scaled scores."""
    t = pd.DataFrame({"feature": logit_table.index})
    t["logit_coef"] = logit_table["coef"].to_numpy()
    t = t.merge(lasso[["feature", "lasso_coef"]], on="feature").merge(perm[["feature", "permutation_importance"]], on="feature")
    t["rf_gini_importance"] = t["feature"].map(gini)
    t["mean_difference"] = t["feature"].map(mean_diff)
    parts = []
    for col in ["logit_coef", "lasso_coef", "rf_gini_importance", "permutation_importance"]:
        v = t[col].abs()
        parts.append((v - v.min()) / (v.max() - v.min() + 1e-12))
    t["overall_score"] = np.mean(parts, axis=0)
    return t.sort_values("overall_score", ascending=False).reset_index(drop=True)
