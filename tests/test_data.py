import numpy as np
import pandas as pd
import pytest

from coffee_quality import config as C
from coffee_quality import data


@pytest.fixture(scope="module")
def clean():
    df, report = data.clean_data(data.load_raw(), data.load_coordinates())
    return data.add_target(df), report


def test_raw_shape():
    assert data.load_raw().shape == (1311, 44)


def test_cleaning_removes_placeholder_and_outliers(clean):
    df, report = clean
    assert (df[C.TARGET_SCORE] > 0).all()
    alt = df["altitude_mean_meters"].dropna()
    assert alt.between(C.ALTITUDE_MIN, C.ALTITUDE_MAX).all()
    assert "Other" not in set(df["Processing.Method"])
    assert report.steps[0] == ("raw", 1311)
    assert report.steps[-1][1] == len(df)


def test_admin_columns_dropped(clean):
    df, _ = clean
    assert not set(C.ADMIN_COLS) & set(df.columns)


def test_labels_harmonised(clean):
    df, _ = clean
    assert set(df["Processing.Method"]) <= {"Washed", "Natural", "Semi-washed", "Honey", "Unknown"}
    assert set(df["Color"]) <= {"green", "blue-green", "Unknown"}


def test_moisture_zero_recoded(clean):
    df, _ = clean
    assert not (df["Moisture"] == 0).any()
    assert df["Moisture"].isna().sum() > 0


def test_target_definition(clean):
    df, _ = clean
    assert df[C.TARGET].isin([0, 1]).all()
    assert (df[C.TARGET] == (df[C.TARGET_SCORE] >= C.EXCELLENT_THRESHOLD).astype(int)).all()
    assert 0.05 < df[C.TARGET].mean() < 0.12  # heavily imbalanced


def test_model_matrix_has_no_sensory_columns(clean):
    df, _ = clean
    X, y = data.make_model_matrix(df)
    assert not set(C.SENSORY_COLS) & set(X.columns)
    assert C.TARGET_SCORE not in X.columns
    assert len(X) == len(y)


def test_split_is_stratified(clean):
    df, _ = clean
    X, y = data.make_model_matrix(df)
    X_tr, X_te, y_tr, y_te = data.stratified_split(X, y)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert len(X_te) == pytest.approx(len(X) * C.TEST_SIZE, abs=1)
    assert not set(X_tr.index) & set(X_te.index)


def test_eda_imputation_fills_everything(clean):
    df, _ = clean
    out = data.impute_for_eda(df)
    assert out["altitude_mean_meters"].notna().all()
    assert out["Moisture"].notna().all()
    assert np.isclose(out["Moisture"].median(), df["Moisture"].median())
    assert isinstance(out, pd.DataFrame)
