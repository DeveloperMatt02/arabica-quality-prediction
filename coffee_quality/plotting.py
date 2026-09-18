"""Figures. One consistent style (colour-blind-safe categorical palette,
single-hue sequential ramp, blue–red diverging ramp) for notebooks and
reports. Every function returns the ``Figure`` and, when ``path`` is given,
saves it."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}  # not excellent / excellent
CLASS_LABELS = {0: "Not excellent", 1: "Excellent"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
SEQUENTIAL = LinearSegmentedColormap.from_list("blues_seq", ["#f4f8fd", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1d1d"])


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelcolor": TEXT,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def _finish(fig, path: str | Path | None):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def plot_score_distribution(df, threshold: float, path=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df["Total.Cup.Points"], bins=40, color=SERIES[0], alpha=0.85, edgecolor="white")
    ax.axvline(threshold, color=SERIES[1], lw=2, ls="--")
    share = (df["Total.Cup.Points"] >= threshold).mean() * 100
    ax.text(threshold + 0.2, ax.get_ylim()[1] * 0.9, f"≥ {threshold:g}: {share:.1f}% of samples", color=SERIES[1])
    ax.set_xlabel("Total Cup Points")
    ax.set_ylabel("Samples")
    ax.set_title("Cup score distribution and the excellence cut-off")
    return _finish(fig, path)


def plot_altitude_vs_score(df, path=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cls in (0, 1):
        sub = df[df["Excellent"] == cls]
        ax.scatter(sub["altitude_mean_meters"], sub["Total.Cup.Points"], s=14, alpha=0.55,
                   color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], edgecolor="none")
    ax.set_xlabel("Mean altitude (m)")
    ax.set_ylabel("Total Cup Points")
    ax.set_title("Altitude vs cup score")
    ax.legend(loc="lower right")
    return _finish(fig, path)


def plot_geo_scatter(df, path=None):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cls in (0, 1):
        sub = df[df["Excellent"] == cls]
        ax.scatter(sub["Longitude"], sub["Latitude"], s=16 if cls else 10, alpha=0.7 if cls else 0.35,
                   color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], edgecolor="none")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where the samples (and the excellent ones) come from")
    ax.legend(loc="lower left")
    return _finish(fig, path)


def plot_rate_by_category(table: pd.DataFrame, title: str, path=None, min_n: int = 10):
    """Horizontal bars of the excellence rate per level (table from eda.rate_by)."""
    t = table[table["n"] >= min_n].sort_values("excellence_rate")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(t) + 1.4))
    ax.barh(t.index.astype(str), t["excellence_rate"] * 100, color=SERIES[0], height=0.6)
    for i, (rate, n) in enumerate(zip(t["excellence_rate"], t["n"])):
        ax.text(rate * 100 + 0.5, i, f"{rate*100:.1f}%  (n={n})", va="center", color=MUTED, fontsize=8)
    ax.set_xlabel("Excellent samples (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(t["excellence_rate"].max() * 100 + 15, 20))
    return _finish(fig, path)


def plot_numeric_by_class(df, cols, path=None):
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.4))
    for ax, col in zip(np.atleast_1d(axes), cols):
        data = [df.loc[df["Excellent"] == c, col].dropna() for c in (0, 1)]
        parts = ax.boxplot(data, tick_labels=[CLASS_LABELS[0], CLASS_LABELS[1]], widths=0.55, patch_artist=True,
                           medianprops={"color": TEXT}, flierprops={"markersize": 3, "alpha": 0.4})
        for patch, c in zip(parts["boxes"], (CLASS_COLORS[0], CLASS_COLORS[1])):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(col)
        ax.tick_params(axis="x", labelsize=8)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Unsupervised
# --------------------------------------------------------------------------- #
def plot_pca_variance(pca, path=None):
    ratio = pca.explained_variance_ratio_
    k = np.arange(1, len(ratio) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(k, ratio * 100, color=SERIES[0])
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Scree plot")
    axes[1].plot(k, np.cumsum(ratio) * 100, marker="o", color=SERIES[0])
    axes[1].axhline(80, color=MUTED, ls="--", lw=1)
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative explained variance (%)")
    axes[1].set_title("Cumulative variance")
    return _finish(fig, path)


def plot_pca_loadings(loadings: pd.DataFrame, n_components: int = 3, path=None):
    fig, axes = plt.subplots(1, n_components, figsize=(3.4 * n_components, 3.6), sharey=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        vals = loadings.iloc[:, i]
        ax.barh(loadings.index, vals, color=[SERIES[0] if v >= 0 else SERIES[7] for v in vals])
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlim(-1, 1)
        ax.set_title(loadings.columns[i])
    return _finish(fig, path)


def plot_pca_biplot(scores: np.ndarray, loadings: pd.DataFrame, labels, path=None, scale: float = 3.0):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    labels = np.asarray(labels)
    for cls in (0, 1):
        m = labels == cls
        ax.scatter(scores[m, 0], scores[m, 1], s=16 if cls else 10, alpha=0.8 if cls else 0.3,
                   color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], edgecolor="none", zorder=2 + cls)
    for feat, (lx, ly) in loadings.iloc[:, :2].iterrows():
        ax.arrow(0, 0, lx * scale, ly * scale, color=TEXT, head_width=0.12, length_includes_head=True, zorder=5)
        ax.text(lx * scale * 1.12, ly * scale * 1.12, feat, color=TEXT, ha="center", va="center", fontsize=8, zorder=6)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA biplot of the agronomic variables")
    ax.legend(loc="upper right")
    return _finish(fig, path)


def plot_embedding(coords: np.ndarray, groups, group_names, title: str, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    groups = np.asarray(groups)
    for i, name in enumerate(group_names):
        m = groups == i
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.7, color=SERIES[i % len(SERIES)], label=name, edgecolor="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(markerscale=2)
    return _finish(fig, path)


def plot_silhouette_scores(table: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, col in enumerate(table.columns):
        ax.plot(table.index, table[col], marker="o", label=col, color=SERIES[i])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette")
    ax.set_title("K-means: silhouette across k")
    ax.legend()
    return _finish(fig, path)


def plot_dbscan_grid(scores: np.ndarray, labellings: dict, path=None):
    n = len(labellings)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.3))
    for ax, (title, labels) in zip(np.atleast_1d(axes), labellings.items()):
        labels = np.asarray(labels)
        noise = labels == -1
        ax.scatter(scores[noise, 0], scores[noise, 1], s=8, color="#c3c2b7", alpha=0.6, edgecolor="none")
        for i, lab in enumerate(sorted(set(labels) - {-1})):
            m = labels == lab
            ax.scatter(scores[m, 0], scores[m, 1], s=8, color=SERIES[i % len(SERIES)], alpha=0.8, edgecolor="none")
        n_clusters = len(set(labels) - {-1})
        ax.set_title(f"{title}\nclusters={n_clusters}, noise={noise.sum()}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    return _finish(fig, path)


def plot_k_distance(distances: np.ndarray, k: int, path=None):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(np.sort(distances), color=SERIES[0], lw=2)
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"Distance to {k}-th neighbour")
    ax.set_title("DBSCAN k-distance plot")
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Supervised
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(results, path=None, ncols: int = 3):
    n = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        ConfusionMatrixDisplay(r.confusion, display_labels=["Not exc.", "Excellent"]).plot(
            ax=ax, cmap=SEQUENTIAL, colorbar=False, values_format="d")
        ax.set_title(f"{r.name}\nthreshold = {r.threshold:.2f}", fontsize=9)
        ax.grid(False)
    for ax in axes[n:]:
        ax.axis("off")
    return _finish(fig, path)


def plot_pr_curves(results, path=None, title: str = "Precision–recall curves (test split)"):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    baseline = results[0].y_test.mean()
    for i, r in enumerate(results):
        p, rec, _ = precision_recall_curve(r.y_test, r.test_proba)
        ap = average_precision_score(r.y_test, r.test_proba)
        ax.plot(rec, p, lw=2, color=SERIES[i % len(SERIES)], label=f"{r.name} (AP={ap:.2f})")
    ax.axhline(baseline, color=MUTED, ls="--", lw=1, label=f"random (AP={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    return _finish(fig, path)


def plot_roc_curves(results, path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r.y_test, r.test_proba)
        ax.plot(fpr, tpr, lw=2, color=SERIES[i % len(SERIES)], label=f"{r.name} (AUC={roc_auc_score(r.y_test, r.test_proba):.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test split)")
    ax.legend(fontsize=8, loc="lower right")
    return _finish(fig, path)


def plot_metric_comparison(summary: pd.DataFrame, metric: str = "auprc", path=None):
    """Dot-and-whisker chart of one test metric with its bootstrap CI."""
    t = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1.3))
    y = np.arange(len(t))
    lo, hi = t.get(f"{metric}_ci_low"), t.get(f"{metric}_ci_high")
    if lo is not None:
        ax.hlines(y, lo, hi, color=SERIES[0], lw=2, alpha=0.6)
    ax.scatter(t[metric], y, color=SERIES[0], s=45, zorder=3)
    if f"oof_{metric}" in t:
        ax.scatter(t[f"oof_{metric}"], y, color=SERIES[1], s=45, marker="D", zorder=3, label="out-of-fold (train CV)")
        ax.scatter([], [], color=SERIES[0], s=45, label="test split (95% bootstrap CI)")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.set_xlabel(metric.upper().replace("_", " "))
    ax.set_xlim(0, 1)
    ax.set_title(f"{metric.upper()} by model")
    return _finish(fig, path)


def plot_coefficients(table: pd.DataFrame, top: int = 20, path=None):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.2))
    width = 0.8 / len(t.columns)
    y = np.arange(len(t))
    for i, col in enumerate(t.columns):
        ax.barh(y + i * width, t[col].fillna(0), height=width, color=SERIES[i], label=col)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y + width * (len(t.columns) - 1) / 2)
    ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("Coefficient (log-odds per SD / vs reference level)")
    ax.set_title("Logistic regression coefficients")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_glm_residuals(glm_result, design: pd.DataFrame, num_vars: list[str], path=None, bins: int = 8):
    """Deviance residuals vs each numeric predictor with a binned mean curve."""
    resid = np.asarray(glm_result.resid_deviance)
    fig, axes = plt.subplots(1, len(num_vars), figsize=(3.6 * len(num_vars), 3.4), sharey=True)
    for ax, var in zip(np.atleast_1d(axes), num_vars):
        x = design[var].to_numpy()
        ax.scatter(x, resid, s=10, alpha=0.3, color=SERIES[0], edgecolor="none")
        tmp = pd.DataFrame({"x": x, "r": resid})
        tmp["bin"] = pd.qcut(tmp["x"].rank(method="first"), bins, labels=False)
        mean = tmp.groupby("bin").agg(x=("x", "mean"), r=("r", "mean"))
        ax.plot(mean["x"], mean["r"], color=SERIES[1], marker="o", lw=2)
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_title(var, fontsize=9)
        ax.set_xlabel("standardised value")
    np.atleast_1d(axes)[0].set_ylabel("Deviance residual")
    return _finish(fig, path)


def plot_influence(glm_result, top_k: int = 5, path=None):
    infl = glm_result.get_influence()
    leverage = np.asarray(infl.hat_matrix_diag)
    cooks = np.asarray(infl.cooks_distance[0])
    std_resid = np.asarray(glm_result.resid_pearson) / np.sqrt(1 - leverage)
    labels = np.asarray(glm_result.model.data.row_labels)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(leverage, std_resid, s=800 * cooks + 12, alpha=0.35, color=SERIES[0], edgecolor="none")
    for pos in np.argsort(cooks)[-top_k:]:
        ax.annotate(str(labels[pos]), (leverage[pos], std_resid[pos]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardised Pearson residual")
    ax.set_title("Influence plot (bubble size ∝ Cook's distance)")
    return _finish(fig, path), labels[np.argsort(cooks)[-top_k:]]


def plot_importance(table: pd.DataFrame, value: str, label: str, title: str, error: str | None = None,
                    top: int = 20, path=None, color=None):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(t) + 1.2))
    ax.barh(t["feature"], t[value], color=color or SERIES[0], xerr=t[error] if error else None,
            error_kw={"ecolor": MUTED, "capsize": 2})
    ax.set_xlabel(label)
    ax.set_title(title)
    return _finish(fig, path)


def plot_importance_comparison(table: pd.DataFrame, cols: list[str], title: str, top: int = 15, path=None):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(t) + 1.2))
    width = 0.8 / len(cols)
    y = np.arange(len(t))
    for i, col in enumerate(cols):
        ax.barh(y + i * width, t[col], height=width, color=SERIES[i], label=col)
    ax.set_yticks(y + width * (len(cols) - 1) / 2)
    ax.set_yticklabels(t["feature"], fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Gain")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_shap_beeswarm(values, path=None, max_display: int = 15):
    import shap

    fig = plt.figure(figsize=(8, 0.4 * max_display + 1.5))
    shap.plots.beeswarm(values, max_display=max_display, show=False, color_bar=True)
    ax = fig.axes[0]
    ax.set_title("SHAP values — contribution to the log-odds of 'excellent' (test split)", loc="left")
    ax.grid(False)
    return _finish(fig, path)


def plot_corr_heatmap(corr: pd.DataFrame, title: str, path=None):
    fig, ax = plt.subplots(figsize=(0.7 * len(corr) + 2, 0.6 * len(corr) + 1.5))
    positive_only = (corr.to_numpy() >= 0).all()
    im = ax.imshow(corr, cmap=SEQUENTIAL if positive_only else DIVERGING, vmin=0 if positive_only else -1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else TEXT)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    return _finish(fig, path)


def plot_leakage_bars(table: pd.DataFrame, metrics=("f1", "auprc"), title: str = "", path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(metrics))
    width = 0.8 / len(table)
    for i, (name, row) in enumerate(table.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics], width=width, color=SERIES[i], label=name)
    ax.set_xticks(x + width * (len(table) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _finish(fig, path)
