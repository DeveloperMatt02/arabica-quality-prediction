import numpy as np
import pandas as pd
import pytest

from coffee_quality import config as C
from coffee_quality.features import AgronomicFeatureEngineer, MacroAreaMapper, MoistureImputer, RegionAltitudeImputer
from coffee_quality.preprocessing import ColumnSelector, build_feature_pipeline, feature_names


@pytest.fixture
def toy():
    return pd.DataFrame(
        {
            "Country.of.Origin": ["Brazil", "Brazil", "Brazil", "Kenya", "Kenya", "Japan", "Mexico"],
            "Region": ["minas", "minas", "sul", "nyeri", None, None, "chiapas"],
            "altitude_mean_meters": [1000.0, 1200.0, np.nan, 1800.0, np.nan, np.nan, 1400.0],
            "Moisture": [0.11, np.nan, 0.12, 0.10, 0.13, np.nan, 0.11],
            "Category.One.Defects": [0, 1, 0, 0, 3, 0, 0],
            "Category.Two.Defects": [2, 0, 5, 1, 0, 8, 1],
            "Variety": ["Bourbon", "Bourbon", "Typica", "SL28", "SL28", "Typica", "Caturra"],
            "Processing.Method": ["Natural", "Natural", "Washed", "Washed", "Washed", "Unknown", "Washed"],
            "Color": ["green", "green", "blue-green", "green", "Unknown", "green", "green"],
        }
    )


def test_altitude_imputer_hierarchy(toy):
    imp = RegionAltitudeImputer().fit(toy)
    out = imp.transform(toy)
    assert out["altitude_mean_meters"].notna().all()
    # Brazil/sul has no altitude -> country mean of Brazil (1100)
    assert out.loc[2, "altitude_mean_meters"] == 1100
    # Kenya with missing region -> country mean of Kenya (1800)
    assert out.loc[4, "altitude_mean_meters"] == 1800
    # Japan unknown everywhere -> global median of the fitted data
    assert out.loc[5, "altitude_mean_meters"] == np.median([1000, 1200, 1800, 1400])


def test_altitude_imputer_learns_only_from_fit_data(toy):
    train = toy.iloc[:3]  # Brazil only
    imp = RegionAltitudeImputer().fit(train)
    out = imp.transform(toy)
    # Kenya is unseen at fit time -> falls back to the *training* median, not Kenya's own values
    assert out.loc[4, "altitude_mean_meters"] == 1100
    assert "Kenya" not in imp.country_means_


def test_moisture_imputer_adds_flag(toy):
    out = MoistureImputer().fit(toy).transform(toy)
    assert out[C.MOISTURE_MISSING_FLAG].tolist() == [0, 1, 0, 0, 0, 1, 0]
    assert out["Moisture"].notna().all()
    assert out.loc[1, "Moisture"] == pytest.approx(np.nanmedian(toy["Moisture"]))


def test_macro_area_mapping(toy):
    out = MacroAreaMapper().fit(toy).transform(toy)
    assert out["MacroArea"].tolist() == ["South America"] * 3 + ["Africa"] * 2 + ["Maritime Asia/Pacific", "Central/North America"]
    assert "Country.of.Origin" not in out.columns and "Region" not in out.columns
    unknown = MacroAreaMapper().fit(toy).transform(toy.assign(**{"Country.of.Origin": "Atlantis"}))
    assert (unknown["MacroArea"] == "Unknown").all()


def test_feature_engineering(toy):
    filled = MoistureImputer().fit(toy).transform(RegionAltitudeImputer().fit(toy).transform(toy))
    out = AgronomicFeatureEngineer().fit(filled).transform(filled)
    assert out["Has_Cat_One_Defect"].tolist() == [0, 1, 0, 0, 1, 0, 0]
    assert np.allclose(out["Log_C2_Defects"], np.log1p(toy["Category.Two.Defects"]))
    assert "Category.One.Defects" not in out.columns
    assert set(AgronomicFeatureEngineer.output_numeric_features()) <= set(out.columns)


def test_feature_pipeline_output_is_dense_named_and_complete(toy):
    pipe = build_feature_pipeline()
    M = pipe.fit_transform(toy)
    assert isinstance(M, pd.DataFrame)
    assert M.isna().sum().sum() == 0
    assert "MacroArea_Africa" in M.columns and "altitude_mean_meters" in M.columns
    # reference-level drop removes exactly one dummy per categorical variable
    M_ref = build_feature_pipeline(drop_reference=True).fit_transform(toy)
    assert M.shape[1] - M_ref.shape[1] == len(C.CATEGORICAL_FEATURES)


def test_column_selector_keeps_only_requested_columns(toy):
    pipe = build_feature_pipeline()
    M = pipe.fit_transform(toy)
    sel = ColumnSelector(["altitude_mean_meters", "MacroArea_Africa", "not_a_column"]).fit(M)
    assert sel.transform(M).columns.tolist() == ["altitude_mean_meters", "MacroArea_Africa"]


def test_feature_names_helper_matches_matrix_width(toy):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("features", build_feature_pipeline()), ("classifier", LogisticRegression(max_iter=200))])
    pipe.fit(toy, [0, 1, 0, 1, 0, 1, 0])
    assert len(feature_names(pipe)) == pipe.named_steps["classifier"].coef_.shape[1]
