"""Decision tree and random forest models.

Both use ``class_weight="balanced"`` and are tuned by grid search on average
precision. The *reduced* random forest keeps only the encoded columns whose
out-of-fold permutation importance exceeds a threshold
(:func:`coffee_quality.interpret.cv_permutation_importance`) — the
selection is made on training folds, never on the test split.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .. import config as C
from ..preprocessing import ColumnSelector, build_feature_pipeline, build_model_pipeline


def _inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED + 81)


def build_decision_tree() -> GridSearchCV:
    clf = DecisionTreeClassifier(class_weight="balanced", random_state=C.SEED)
    pipe = build_model_pipeline(clf, scale=False)
    grid = {
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [2, 4, 8, 16],
    }
    return GridSearchCV(pipe, grid, cv=_inner_cv(), scoring="average_precision", n_jobs=1)


RF_GRID = {
    "classifier__n_estimators": [300],
    "classifier__max_depth": [3, 5, 7, None],
    "classifier__max_features": ["sqrt", None],
}


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", random_state=C.SEED, n_jobs=1)


def build_random_forest(*, engineer: bool = False) -> GridSearchCV:
    pipe = build_model_pipeline(_rf(), engineer=engineer, scale=False)
    return GridSearchCV(pipe, RF_GRID, cv=_inner_cv(), scoring="average_precision", n_jobs=1)


def build_reduced_random_forest(selected_columns: list[str], *, engineer: bool = False) -> GridSearchCV:
    """Random forest restricted to ``selected_columns`` of the encoded matrix."""
    pipe = Pipeline(
        [
            ("features", build_feature_pipeline(engineer=engineer, scale=False)),
            ("select", ColumnSelector(list(selected_columns))),
            ("classifier", _rf()),
        ]
    )
    return GridSearchCV(pipe, RF_GRID, cv=_inner_cv(), scoring="average_precision", n_jobs=1)
