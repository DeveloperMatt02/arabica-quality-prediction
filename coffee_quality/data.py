"""Loading, cleaning and splitting of the CQI Arabica dataset.

The cleaning steps here are *target-agnostic and row-wise*: they only remove
administrative columns, fix obviously corrupted values and harmonise labels.
Anything that has to be *learned* from the data (altitude imputation, rare
category pooling, scaling, one-hot encoding) lives in
:mod:`coffee_quality.features` and :mod:`coffee_quality.preprocessing` so that
it is fitted on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path=C.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CQI export (1,311 rows x 44 columns)."""
    return pd.read_csv(path)


def load_coordinates(path=C.COORDS_PATH) -> pd.DataFrame:
    """Read the (Country, Region) -> (Latitude, Longitude) lookup table.

    The table was produced once with ``scripts/geocode_regions.py``
    (OpenStreetMap / OpenCage geocoding) and is versioned so that the
    pipeline never needs network access.
    """
    coords = pd.read_csv(path)
    for col in C.GEO_COLS:
        coords[col] = coords[col].str.strip()  # keeps NaN as NaN
    return coords[C.GEO_COLS + ["Latitude", "Longitude"]].drop_duplicates(subset=C.GEO_COLS)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    """Row counts after each cleaning step, for the data-cleaning notebook."""

    steps: list[tuple[str, int]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "rows"])


def clean_data(raw: pd.DataFrame, coords: pd.DataFrame | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the deterministic cleaning steps and return the tidy dataset.

    Steps (in order):

    1. drop the single record with ``Total.Cup.Points == 0`` (all sensory
       scores are zero too: a placeholder, not a coffee);
    2. drop administrative / identifier columns;
    3. strip whitespace in text columns, turn empty strings into NaN;
    4. harmonise ``Processing.Method`` and ``Color`` labels;
    5. attach region-level coordinates (used for EDA only);
    6. ``Quakers``: the single NaN becomes 0 (93% of the values are 0);
    7. drop rows whose altitude is outside the plausible range
       (:data:`config.ALTITUDE_MIN` - :data:`config.ALTITUDE_MAX`); missing
       altitudes are *kept* and imputed later inside the modelling pipeline;
    8. missing ``Processing.Method`` / ``Color`` / ``Variety`` become the
       explicit level ``"Unknown"`` (missingness is informative: it is
       concentrated in a few origins);
    9. drop ``Processing.Method == "Other"``: 26 rows, no excellent coffee,
       and a source of perfect separation in the logistic regression;
    10. ``Moisture == 0`` (physically impossible for green coffee, 18% of the
        rows) is recoded as missing; the modelling pipeline imputes it and
        keeps an indicator column.
    """
    df = raw.copy()
    steps: list[tuple[str, int]] = [("raw", len(df))]

    # 1. placeholder record
    df = df[df[C.TARGET_SCORE] > 0].copy()
    steps.append(("drop Total.Cup.Points == 0", len(df)))

    # 2. administrative columns
    df = df.drop(columns=[c for c in C.ADMIN_COLS if c in df.columns])

    # 3. text hygiene
    text_cols = df.select_dtypes(include=["object", "string", "str"]).columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip()).replace("", np.nan)
    df = df.dropna(subset=C.GEO_COLS, how="all")
    df["Country.of.Origin"] = df["Country.of.Origin"].astype(str).str.strip()
    steps.append(("drop rows without country and region", len(df)))

    # 4. label harmonisation
    df["Processing.Method"] = df["Processing.Method"].replace(C.PROCESSING_MAP)
    df["Color"] = df["Color"].str.lower().replace({"bluish-green": "blue-green"})

    # 5. coordinates (EDA only)
    if coords is not None:
        df["Region"] = df["Region"].astype(str).str.strip().replace({"nan": np.nan})
        # merge on a stringified region so that NaN regions match the "NaN" rows of the lookup
        key = df["Region"].fillna("NaN")
        lookup = coords.copy()
        lookup["Region"] = lookup["Region"].fillna("NaN")
        merged = pd.DataFrame({"Country.of.Origin": df["Country.of.Origin"], "Region": key}).merge(
            lookup, on=C.GEO_COLS, how="left"
        )
        df["Latitude"] = merged["Latitude"].to_numpy()
        df["Longitude"] = merged["Longitude"].to_numpy()

    # 6. quakers
    df["Quakers"] = df["Quakers"].fillna(0)

    # 7. implausible altitudes
    alt = df["altitude_mean_meters"]
    implausible = alt.notna() & ((alt < C.ALTITUDE_MIN) | (alt > C.ALTITUDE_MAX))
    df = df[~implausible].copy()
    steps.append(("drop implausible altitudes", len(df)))

    # 8. explicit "Unknown" level
    for col in ["Processing.Method", "Color", "Variety"]:
        df[col] = df[col].fillna("Unknown")

    # 9. "Other" processing method
    df = df[df["Processing.Method"] != "Other"].copy()
    steps.append(("drop Processing.Method == Other", len(df)))

    # 10. moisture "not measured" code
    n_zero = int((df["Moisture"] == 0).sum())
    df.loc[df["Moisture"] == 0, "Moisture"] = np.nan
    steps.append((f"recode Moisture == 0 as missing ({n_zero} rows, no drop)", len(df)))

    df = df.reset_index(drop=True)
    return df, CleaningReport(steps)


def add_target(df: pd.DataFrame, threshold: float = C.EXCELLENT_THRESHOLD) -> pd.DataFrame:
    """Add the binary ``Excellent`` column (1 if Total.Cup.Points >= threshold)."""
    out = df.copy()
    out[C.TARGET] = (out[C.TARGET_SCORE] >= threshold).astype(int)
    return out


def impute_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing altitudes (country/region mean) and moisture (median) on *all* rows.

    Only for exploratory analysis and unsupervised learning, where there is no
    train/test boundary to respect. The supervised pipelines use the
    transformers in :mod:`coffee_quality.features` instead.
    """
    out = df.copy()
    alt = "altitude_mean_meters"
    region_mean = out.groupby(C.GEO_COLS, dropna=False)[alt].transform("mean")
    country_mean = out.groupby("Country.of.Origin")[alt].transform("mean")
    out[alt] = out[alt].fillna(region_mean).fillna(country_mean).fillna(out[alt].median())
    out["Moisture"] = out["Moisture"].fillna(out["Moisture"].median())
    return out


# --------------------------------------------------------------------------- #
# Model matrix and split
# --------------------------------------------------------------------------- #
def make_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with the raw agronomic predictors and the binary target.

    ``X`` still contains ``Country.of.Origin`` / ``Region`` (needed by the
    altitude imputer and the macro-area mapper) and possibly missing
    altitudes: everything else is handled inside the fitted pipeline.
    """
    if C.TARGET not in df.columns:
        df = add_target(df)
    X = df[C.MODEL_INPUT_COLS].copy()
    y = df[C.TARGET].astype(int).copy()
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = C.TEST_SIZE, seed: int = C.SEED):
    """Single stratified hold-out split used by every supervised model."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def load_clean(with_target: bool = True) -> pd.DataFrame:
    """Convenience: raw -> clean (-> target) in one call."""
    df, _ = clean_data(load_raw(), load_coordinates())
    return add_target(df) if with_target else df
