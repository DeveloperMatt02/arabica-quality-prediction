"""One evaluation protocol for every model.

    1. out-of-fold (OOF) probabilities on the training split via stratified
       k-fold cross-validation (nested when the estimator is a ``GridSearchCV``);
    2. decision threshold = the one maximising F1 on the OOF probabilities
       — the test labels are never consulted;
    3. refit on the whole training split, predict the hold-out test split;
    4. threshold-free metrics (AUPRC, ROC AUC) and threshold-dependent ones
       (precision, recall, F1) on the test split, with bootstrap confidence
       intervals because the test split only holds ~20 positives.

AUPRC (average precision) is the headline metric: with 8% positives a
classifier that flags nothing is 92% accurate, and ROC AUC is dominated by the
easy negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C

METRIC_ORDER = ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "auprc"]


def make_cv(n_splits: int = C.N_FOLDS, seed: int = C.SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# --------------------------------------------------------------------------- #
# Thresholds and metrics
# --------------------------------------------------------------------------- #
def find_optimal_threshold(y_true, y_proba, beta: float = 1.0) -> tuple[float, float]:
    """Threshold maximising the F-beta score on ``(y_true, y_proba)``.

    Returns ``(threshold, best_fbeta)``. Must be called on out-of-fold or
    validation predictions, never on the test split.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    best = int(np.argmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
    }


def bootstrap_ci(y_true, y_proba, threshold: float, metrics=("auprc", "f1", "roc_auc"),
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = C.SEED) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals of test metrics.

    Resamples the test rows with replacement (stratification is not enforced;
    resamples without positives are skipped).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        m = compute_metrics(y_true[idx], y_proba[idx], threshold)
        for k in metrics:
            draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in draws.items()}


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    name: str
    estimator: object
    threshold: float
    oof_proba: np.ndarray
    test_proba: np.ndarray
    y_test: np.ndarray
    metrics: dict[str, float]
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    oof_metrics: dict[str, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def y_pred(self) -> np.ndarray:
        return (self.test_proba >= self.threshold).astype(int)

    @property
    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_test, self.y_pred)

    def summary_row(self) -> dict:
        row = {"model": self.name, "threshold": self.threshold}
        row.update({k: self.metrics[k] for k in METRIC_ORDER})
        for k, (lo, hi) in self.ci.items():
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = lo, hi
        row["oof_auprc"] = self.oof_metrics.get("auprc", np.nan)
        row["oof_roc_auc"] = self.oof_metrics.get("roc_auc", np.nan)
        return row


def oof_probabilities(estimator, X, y, cv=None, n_jobs: int = -1) -> np.ndarray:
    """Out-of-fold positive-class probabilities.

    If ``estimator`` is a ``GridSearchCV`` this is a *nested* cross-validation:
    hyper-parameters are re-tuned inside each outer training fold.
    """
    cv = cv or make_cv()
    return cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]


def evaluate(name: str, estimator, X_train, y_train, X_test, y_test, *, cv=None,
             beta: float = 1.0, n_boot: int = 1000, n_jobs: int = -1) -> ModelResult:
    """Run the full protocol for one estimator and return a :class:`ModelResult`."""
    cv = cv or make_cv()
    oof = oof_probabilities(estimator, X_train, y_train, cv=cv, n_jobs=n_jobs)
    threshold, _ = find_optimal_threshold(y_train, oof, beta=beta)
    oof_metrics = compute_metrics(y_train, oof, threshold)

    fitted = clone(estimator).fit(X_train, y_train)
    test_proba = fitted.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_proba, threshold)
    ci = bootstrap_ci(y_test, test_proba, threshold, n_boot=n_boot) if n_boot else {}
    return ModelResult(
        name=name,
        estimator=fitted,
        threshold=threshold,
        oof_proba=oof,
        test_proba=test_proba,
        y_test=np.asarray(y_test),
        metrics=metrics,
        ci=ci,
        oof_metrics=oof_metrics,
    )


def summary_table(results: list[ModelResult]) -> pd.DataFrame:
    """Tidy comparison table (one row per model)."""
    return pd.DataFrame([r.summary_row() for r in results]).set_index("model")


def format_summary(results: list[ModelResult]) -> pd.DataFrame:
    """Compact, human-readable table with CIs, for the README / notebooks."""
    rows = []
    for r in results:
        m, ci = r.metrics, r.ci
        rows.append(
            {
                "Model": r.name,
                "Threshold": f"{r.threshold:.2f}",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1": f"{m['f1']:.3f}" + (f" [{ci['f1'][0]:.2f}, {ci['f1'][1]:.2f}]" if "f1" in ci else ""),
                "ROC AUC": f"{m['roc_auc']:.3f}",
                "AUPRC": f"{m['auprc']:.3f}" + (f" [{ci['auprc'][0]:.2f}, {ci['auprc'][1]:.2f}]" if "auprc" in ci else ""),
                "OOF AUPRC": f"{r.oof_metrics.get('auprc', float('nan')):.3f}",
            }
        )
    return pd.DataFrame(rows).set_index("Model")


# --------------------------------------------------------------------------- #
# Leakage demonstration
# --------------------------------------------------------------------------- #
def threshold_leakage_demo(result: ModelResult) -> pd.DataFrame:
    """Show how much F1 is inflated by tuning the threshold on the test labels.

    This is exactly the mistake found in one of the original notebooks
    (``precision_recall_curve(y_test, ...)`` used to pick the cut-off). AUPRC
    is unaffected because it does not depend on the threshold.
    """
    honest = result.metrics
    leaked_thr, _ = find_optimal_threshold(result.y_test, result.test_proba)
    leaked = compute_metrics(result.y_test, result.test_proba, leaked_thr)
    return pd.DataFrame(
        {
            "threshold": [result.threshold, leaked_thr],
            "precision": [honest["precision"], leaked["precision"]],
            "recall": [honest["recall"], leaked["recall"]],
            "f1": [honest["f1"], leaked["f1"]],
            "auprc": [honest["auprc"], leaked["auprc"]],
        },
        index=["threshold chosen on OOF train predictions (correct)",
               "threshold chosen on the test labels (leakage)"],
    )
