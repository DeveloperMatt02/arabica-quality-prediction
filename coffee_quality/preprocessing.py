"""The single preprocessing pipeline shared by every supervised model.

Raw predictors  ->  altitude imputation  ->  moisture imputation (+ indicator)
                ->  macro-area mapping
                ->  (optional feature engineering)
                ->  scaling of numeric columns + one-hot encoding of categorical
                    columns (rare varieties pooled into an "infrequent" level)

The original notebooks used three slightly different encoders (one per model
family), which made model comparisons apples-to-oranges. Here the encoder is
identical; only two switches differ per model:

* ``drop_reference``: drop one dummy per categorical variable. Required by the
  unpenalised logistic regression (perfect collinearity otherwise) and
  harmless for trees, but *not* used with L1/L2 penalties, where keeping every
  level lets the penalty decide.
* ``scale``: standardise numeric columns. Needed by the logistic models, a
  no-op for trees (kept for uniformity).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import AgronomicFeatureEngineer, MacroAreaMapper, MoistureImputer, RegionAltitudeImputer

# Reference levels dropped by the unpenalised logistic regression, chosen as
# the "market standard" of each variable (Caturra variety, washed process,
# green beans, Central/North America = most frequent macro-area).
REFERENCE_LEVELS = {
    "MacroArea": "Central/North America",
    "Color": "green",
    "Processing.Method": "Washed",
    "Variety": "Caturra",
}


def build_encoder(numeric_features: list[str], categorical_features: list[str], *,
                  drop_reference: bool = False, scale: bool = True,
                  max_categories: int = C.MAX_CATEGORIES) -> ColumnTransformer:
    """Column transformer: scaler on numeric, one-hot on categorical."""
    drop = [REFERENCE_LEVELS[c] for c in categorical_features] if drop_reference else None
    ohe = OneHotEncoder(
        drop=drop,
        max_categories=max_categories,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    encoder = ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", numeric_features),
            ("cat", ohe, categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    # pandas output keeps column names all the way to the classifier, which
    # makes feature importances / SHAP plots readable without bookkeeping.
    encoder.set_output(transform="pandas")
    return encoder


def build_feature_pipeline(*, engineer: bool = False, drop_reference: bool = False,
                           scale: bool = True) -> Pipeline:
    """Everything between the raw predictors and the model matrix."""
    numeric = (
        AgronomicFeatureEngineer.output_numeric_features() if engineer else list(C.NUMERIC_MODEL_FEATURES)
    )
    steps = [
        ("altitude", RegionAltitudeImputer()),
        ("moisture", MoistureImputer()),
        ("macro_area", MacroAreaMapper()),
    ]
    if engineer:
        steps.append(("engineer", AgronomicFeatureEngineer()))
    steps.append(
        ("encode", build_encoder(numeric, list(C.CATEGORICAL_FEATURES),
                                 drop_reference=drop_reference, scale=scale))
    )
    return Pipeline(steps)


def build_model_pipeline(classifier, *, engineer: bool = False, drop_reference: bool = False,
                         scale: bool = True) -> Pipeline:
    """Feature pipeline + classifier, as a single sklearn ``Pipeline``.

    The classifier step is always called ``"classifier"`` so that grid-search
    parameter names are uniform (``classifier__C``, ``classifier__max_depth``).
    """
    features = build_feature_pipeline(engineer=engineer, drop_reference=drop_reference, scale=scale)
    return Pipeline([("features", features), ("classifier", classifier)])


def clean_name(name: str) -> str:
    return name.replace("infrequent_sklearn", "Rare")


def feature_names(fitted_pipeline) -> np.ndarray:
    """Human-readable names of the columns fed to the classifier."""
    est = getattr(fitted_pipeline, "best_estimator_", fitted_pipeline)
    encoder = est.named_steps["features"].named_steps["encode"] if "features" in est.named_steps else est.named_steps["encode"]
    names = encoder.get_feature_names_out()
    if "feature_selection" in est.named_steps:
        names = names[est.named_steps["feature_selection"].get_support()]
    if "select" in est.named_steps:
        names = np.asarray(est.named_steps["select"].selected_)
    return np.asarray([clean_name(n) for n in names], dtype=object)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep only the encoded columns listed in ``names`` (missing ones are ignored)."""

    def __init__(self, names: list[str]):
        self.names = names

    def fit(self, X, y=None):
        wanted = {clean_name(n) for n in self.names}
        self.selected_ = [c for c in X.columns if clean_name(c) in wanted]
        return self

    def transform(self, X):
        return X[self.selected_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)
