"""Learned feature transformations, written as scikit-learn transformers.

Because they are transformers, they can be dropped into a ``Pipeline`` and are
therefore fitted on the training folds only — inside cross-validation as well
as for the final model. This is what closes the pre-processing leakage of the
original notebooks (altitude means and rare-variety pooling computed on the
whole dataset before the split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C


class RegionAltitudeImputer(BaseEstimator, TransformerMixin):
    """Impute missing ``altitude_mean_meters`` hierarchically.

    Missing altitudes are replaced, in order of preference, by the mean
    altitude of the same *(country, region)*, of the same *country*, and
    finally by the global median. All three lookups are estimated on the data
    passed to :meth:`fit` (i.e. the training fold), so nothing is learned from
    the evaluation data.
    """

    def __init__(self, country_col: str = "Country.of.Origin", region_col: str = "Region",
                 alt_col: str = "altitude_mean_meters"):
        self.country_col = country_col
        self.region_col = region_col
        self.alt_col = alt_col

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        region = X[self.region_col].fillna("__missing__")
        self.region_means_ = (
            X.assign(_region=region)
            .groupby([self.country_col, "_region"])[self.alt_col]
            .mean()
            .dropna()
            .to_dict()
        )
        self.country_means_ = X.groupby(self.country_col)[self.alt_col].mean().dropna().to_dict()
        self.global_median_ = float(X[self.alt_col].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        missing = X[self.alt_col].isna()
        if missing.any():
            region = X[self.region_col].fillna("__missing__")
            keys = list(zip(X[self.country_col], region))
            by_region = pd.Series([self.region_means_.get(k, np.nan) for k in keys], index=X.index)
            by_country = X[self.country_col].map(self.country_means_)
            filled = by_region.fillna(by_country).fillna(self.global_median_)
            X.loc[missing, self.alt_col] = filled[missing]
        X[self.alt_col] = pd.to_numeric(X[self.alt_col], errors="coerce").round(0)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class MoistureImputer(BaseEstimator, TransformerMixin):
    """Median-impute ``Moisture`` and add a ``Moisture_missing`` indicator.

    The raw export uses ``0`` for "not measured"; :func:`data.clean_data`
    turns it into NaN and this step (fitted on the training fold) fills it.
    The indicator is kept as a feature because missingness is not random: it
    is concentrated in a few certifying laboratories / origins.
    """

    def __init__(self, col: str = "Moisture", flag: str = C.MOISTURE_MISSING_FLAG):
        self.col = col
        self.flag = flag

    def fit(self, X, y=None):
        self.median_ = float(pd.DataFrame(X)[self.col].median())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.flag] = X[self.col].isna().astype(int)
        X[self.col] = X[self.col].fillna(self.median_)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + [self.flag], dtype=object)


class MacroAreaMapper(BaseEstimator, TransformerMixin):
    """Replace ``Country.of.Origin`` / ``Region`` with a coarse ``MacroArea``.

    Country (36 levels) and Region (300+ levels) would let a tree model
    memorise individual farms; five macro-areas keep the geographic signal
    at a resolution that can generalise. Countries absent from the mapping
    become ``"Unknown"``.
    """

    def __init__(self, country_col: str = "Country.of.Origin", region_col: str = "Region",
                 mapping: dict | None = None, drop: bool = True):
        self.country_col = country_col
        self.region_col = region_col
        self.mapping = mapping
        self.drop = drop

    def fit(self, X, y=None):
        self.mapping_ = dict(C.MACRO_AREA_MAP if self.mapping is None else self.mapping)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        X["MacroArea"] = X[self.country_col].map(self.mapping_).fillna("Unknown")
        if self.drop:
            X = X.drop(columns=[c for c in (self.country_col, self.region_col) if c in X.columns])
        return X

    def get_feature_names_out(self, input_features=None):
        cols = [c for c in input_features if not (self.drop and c in (self.country_col, self.region_col))]
        return np.asarray(cols + ["MacroArea"], dtype=object)


class AgronomicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Hand-crafted features for the tree ensembles.

    * ``Log_C1_Defects`` / ``Log_C2_Defects``: ``log1p`` of the long-tailed
      defect counts;
    * ``Has_Cat_One_Defect``: binary flag, primary defects are rare but
      disqualifying for specialty grading;
    * ``Log_Altitude_Moisture_Ratio``: ``log1p(altitude / moisture)``, an
      interaction between the two strongest numeric predictors.

    The raw defect counts are dropped to avoid duplicating information.
    """

    def __init__(self, drop_raw_defects: bool = True):
        self.drop_raw_defects = drop_raw_defects

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        X["Log_C1_Defects"] = np.log1p(X["Category.One.Defects"])
        X["Log_C2_Defects"] = np.log1p(X["Category.Two.Defects"])
        X["Has_Cat_One_Defect"] = (X["Category.One.Defects"] > 0).astype(int)
        X["Log_Altitude_Moisture_Ratio"] = np.log1p(X["altitude_mean_meters"] / (X["Moisture"] + 1e-4))
        if self.drop_raw_defects:
            X = X.drop(columns=["Category.One.Defects", "Category.Two.Defects"])
        return X

    @staticmethod
    def output_numeric_features(drop_raw_defects: bool = True) -> list[str]:
        base = ["Moisture", "altitude_mean_meters", C.MOISTURE_MISSING_FLAG]
        if not drop_raw_defects:
            base += ["Category.One.Defects", "Category.Two.Defects"]
        return base + ["Log_C1_Defects", "Log_C2_Defects", "Has_Cat_One_Defect", "Log_Altitude_Moisture_Ratio"]
