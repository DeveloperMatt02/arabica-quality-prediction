import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from coffee_quality import data, evaluation as E
from coffee_quality.preprocessing import build_model_pipeline


def test_optimal_threshold_recovers_separable_case():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    thr, f1 = E.find_optimal_threshold(y, p)
    assert 0.4 < thr <= 0.8
    assert f1 == pytest.approx(1.0)


def test_compute_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    m = E.compute_metrics(y, p, 0.5)
    assert set(m) == set(E.METRIC_ORDER)
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.1).astype(int)
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.2, 300), 0, 1)
    point = E.compute_metrics(y, p, 0.5)
    ci = E.bootstrap_ci(y, p, 0.5, n_boot=200)
    for k, (lo, hi) in ci.items():
        assert lo <= point[k] <= hi


def test_evaluate_protocol_end_to_end():
    df = data.load_clean()
    X, y = data.make_model_matrix(df)
    X_tr, X_te, y_tr, y_te = data.stratified_split(X, y)
    est = build_model_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000), drop_reference=True)
    res = E.evaluate("lr", est, X_tr, y_tr, X_te, y_te, n_boot=20, n_jobs=1)
    assert len(res.oof_proba) == len(y_tr)
    assert len(res.test_proba) == len(y_te)
    assert 0 < res.threshold < 1
    assert res.metrics["auprc"] > y_te.mean()  # better than random
    row = res.summary_row()
    assert row["model"] == "lr" and "auprc_ci_low" in row
    table = E.summary_table([res])
    assert isinstance(table, pd.DataFrame) and table.index.tolist() == ["lr"]


def test_threshold_leakage_demo_never_lowers_f1():
    rng = np.random.default_rng(2)
    y = (rng.random(200) < 0.1).astype(int)
    p = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.25, 200), 0, 1)
    res = E.ModelResult("m", None, 0.5, p, p, y, E.compute_metrics(y, p, 0.5))
    table = E.threshold_leakage_demo(res)
    assert table.iloc[1]["f1"] >= table.iloc[0]["f1"]
    assert table.iloc[1]["auprc"] == table.iloc[0]["auprc"]
