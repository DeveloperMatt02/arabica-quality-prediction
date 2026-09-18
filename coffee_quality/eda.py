"""Small exploratory-analysis helpers (tables; figures live in plotting.py)."""

from __future__ import annotations

import pandas as pd
from scipy.stats import mannwhitneyu

from . import config as C


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_pct": (df.isna().mean() * 100).round(1), "n_unique": df.nunique()})
        .sort_values("missing_pct", ascending=False)
    )


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[C.TARGET].value_counts().rename(index={0: "Not excellent", 1: "Excellent"})
    return pd.DataFrame({"count": counts, "percentage": (counts / counts.sum() * 100).round(2)})


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Excellence rate and mean score per level of a categorical column."""
    g = df.groupby(col)
    out = pd.DataFrame(
        {"n": g.size(), "n_excellent": g[C.TARGET].sum(), "excellence_rate": g[C.TARGET].mean(), "mean_score": g[C.TARGET_SCORE].mean()}
    )
    return out.sort_values("excellence_rate", ascending=False)


def mann_whitney_by_class(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Two-sided Mann–Whitney U test of each numeric column between classes."""
    rows = []
    for col in cols:
        a = df.loc[df[C.TARGET] == 0, col].dropna()
        b = df.loc[df[C.TARGET] == 1, col].dropna()
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {"feature": col, "median_not_excellent": a.median(), "median_excellent": b.median(),
             "mean_difference": b.mean() - a.mean(), "p_value": p}
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
