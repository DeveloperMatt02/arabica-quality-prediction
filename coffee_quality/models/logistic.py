"""Logistic regression models (the interpretable baseline).

Four variants, all with ``class_weight="balanced"``:

* **unpenalised** — the textbook GLM; one dummy per categorical variable is
  dropped so that the design matrix has full rank. Coefficients are
  interpretable and a ``statsmodels`` refit gives standard errors, p-values
  and residual / influence diagnostics;
* **unpenalised + RFECV** — recursive feature elimination with
  cross-validation on top of the same model;
* **L2 (ridge)** and **L1 (lasso)** — penalty strength ``C`` tuned by an
  inner grid search on average precision; all dummies are kept and the
  penalty resolves the collinearity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .. import config as C
from ..preprocessing import build_feature_pipeline, build_model_pipeline, clean_name

C_GRID = np.logspace(-3, 3, 30)


def _inner_cv() -> StratifiedKFold:
    # different seed from the outer CV so the nested folds are not aligned
    return StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED + 81)


def build_unpenalized() -> Pipeline:
    clf = LogisticRegression(penalty=None, class_weight="balanced", solver="lbfgs", max_iter=5000)
    return build_model_pipeline(clf, drop_reference=True)


def build_unpenalized_rfecv() -> Pipeline:
    base = LogisticRegression(penalty=None, class_weight="balanced", solver="lbfgs", max_iter=5000)
    selector = RFECV(base, step=1, cv=_inner_cv(), scoring="average_precision", min_features_to_select=5, n_jobs=1)
    return Pipeline(
        [
            ("features", build_feature_pipeline(drop_reference=True)),
            ("feature_selection", selector),
            ("classifier", LogisticRegression(penalty=None, class_weight="balanced", solver="lbfgs", max_iter=5000)),
        ]
    )


def build_penalized(penalty: str = "l2") -> GridSearchCV:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(penalty=penalty, class_weight="balanced", solver=solver, max_iter=5000)
    pipe = build_model_pipeline(clf, drop_reference=False)
    return GridSearchCV(pipe, {"classifier__C": C_GRID}, cv=_inner_cv(), scoring="average_precision", n_jobs=1)


def all_models() -> dict[str, object]:
    return {
        "LogReg (unpenalised)": build_unpenalized(),
        "LogReg (unpenalised + RFECV)": build_unpenalized_rfecv(),
        "LogReg L2": build_penalized("l2"),
        "LogReg L1": build_penalized("l1"),
    }


# --------------------------------------------------------------------------- #
# statsmodels refit for inference & diagnostics
# --------------------------------------------------------------------------- #
def fit_glm(X_train: pd.DataFrame, y_train: pd.Series):
    """Refit the unpenalised, class-weighted logistic regression with statsmodels.

    Returns ``(result, design)`` where ``design`` is the weighted design matrix
    (with intercept) used for residual and influence plots.
    """
    features = build_feature_pipeline(drop_reference=True).fit(X_train, y_train)
    design = features.transform(X_train)
    design.columns = [clean_name(c) for c in design.columns]
    design.insert(0, "const", 1.0)
    design.index = X_train.index
    # class_weight="balanced" equivalent: w_i = n / (2 * n_class_i)
    counts = np.bincount(y_train)
    weights = len(y_train) / (2.0 * counts[y_train.to_numpy()])
    model = sm.GLM(y_train, design, family=sm.families.Binomial(), var_weights=weights)
    return model.fit(), design


def coefficient_table(results: dict) -> pd.DataFrame:
    """Side-by-side coefficients of the fitted logistic models (outer join)."""
    from ..preprocessing import feature_names

    series = {}
    for name, res in results.items():
        est = getattr(res.estimator, "best_estimator_", res.estimator)
        coef = est.named_steps["classifier"].coef_.ravel()
        series[name] = pd.Series(coef, index=feature_names(res.estimator))
    table = pd.DataFrame(series)
    table["max_abs"] = table.abs().max(axis=1)
    return table.sort_values("max_abs", ascending=False).drop(columns="max_abs")
