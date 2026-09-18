"""Unsupervised exploration of the agronomic variables: PCA, t-SNE,
k-means, hierarchical clustering and DBSCAN.

The question here is whether the coffees fall into natural groups (by
origin, altitude band, processing…) that could be exploited later. Spoiler:
they do not — every method points at one continuous cloud stretched along
the altitude axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config as C

UNSUPERVISED_COLS = ["Moisture", "Category.One.Defects", "Category.Two.Defects", "altitude_mean_meters", "Longitude", "LatitudeAbs"]


def numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Standardisable numeric block used by every unsupervised method.

    Latitude enters as its absolute value: what matters agronomically is the
    distance from the equator, not the hemisphere.
    """
    out = df[["Moisture", "Category.One.Defects", "Category.Two.Defects", "altitude_mean_meters", "Longitude"]].copy()
    out["LatitudeAbs"] = df["Latitude"].abs()
    return out.dropna()


def altitude_band(altitude: pd.Series) -> tuple[np.ndarray, list[str]]:
    bands = ["Low (<1200 m)", "Medium (1200–1800 m)", "High (≥1800 m)"]
    codes = np.select([altitude < 1200, altitude < 1800], [0, 1], default=2)
    return codes, bands


def fit_pca(X: pd.DataFrame):
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(random_state=C.SEED).fit(Z)
    scores = pca.transform(Z)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    return pca, Z, scores, loadings


def fit_tsne(Z: np.ndarray, perplexity: int = 50, n_pca: int = 4) -> np.ndarray:
    reduced = PCA(n_components=n_pca, random_state=C.SEED).fit_transform(Z)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=C.SEED).fit_transform(reduced)


def kmeans_silhouettes(Z: np.ndarray, scores: np.ndarray, ks=range(2, 7), n_pcs: int = 3) -> pd.DataFrame:
    rows = {}
    for k in ks:
        lab_raw = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit_predict(Z)
        lab_pca = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit_predict(scores[:, :n_pcs])
        rows[k] = {"standardised variables": silhouette_score(Z, lab_raw), f"first {n_pcs} PCs": silhouette_score(scores[:, :n_pcs], lab_pca)}
    return pd.DataFrame(rows).T.rename_axis("k")


def hierarchical_linkages(Z: np.ndarray) -> dict[str, np.ndarray]:
    return {m: linkage(Z, method=m, metric="euclidean") for m in ("single", "average", "complete", "ward")}


def k_distances(scores: np.ndarray, k: int = 7) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k).fit(scores)
    return nn.kneighbors(scores)[0][:, -1]


def dbscan_sweep(scores: np.ndarray, eps_values=(0.75, 1.0, 1.25, 1.5), min_samples: int = 7) -> dict[str, np.ndarray]:
    return {f"eps={e}": DBSCAN(eps=e, min_samples=min_samples).fit_predict(scores) for e in eps_values}


def cluster_vs_class(labels, y: pd.Series) -> pd.DataFrame:
    """Cross-tab of cluster membership against the excellence label."""
    tab = pd.crosstab(pd.Series(labels, name="cluster"), y.rename("Excellent"))
    tab["excellence_rate"] = tab[1] / tab.sum(axis=1)
    return tab
