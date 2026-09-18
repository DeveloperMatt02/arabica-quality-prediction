#!/usr/bin/env python
"""End-to-end reproduction of every number and figure in the repository.

    python scripts/run_pipeline.py            # full run (~10 min on a laptop)
    python scripts/run_pipeline.py --quick    # fewer bootstrap draws, coarser grids
    python scripts/run_pipeline.py --stage models --stage interpret

Outputs land in ``reports/metrics/*.csv`` and ``reports/figures/*.png``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coffee_quality import config as C  # noqa: E402
from coffee_quality import data, eda  # noqa: E402
from coffee_quality import evaluation as E  # noqa: E402
from coffee_quality import interpret as I  # noqa: E402
from coffee_quality import plotting as P  # noqa: E402
from coffee_quality import sensory as S  # noqa: E402
from coffee_quality import unsupervised as U  # noqa: E402
from coffee_quality.models import boosting, forest, logistic  # noqa: E402

warnings.filterwarnings("ignore")
FIG, MET = C.FIGURES_DIR, C.METRICS_DIR


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
def stage_eda(df: pd.DataFrame, report) -> None:
    log("EDA")
    report.to_frame().to_csv(MET / "cleaning_steps.csv", index=False)
    eda.class_balance(df).to_csv(MET / "class_balance.csv")
    eda.mann_whitney_by_class(df, C.NUMERIC_FEATURES).to_csv(MET / "eda_mann_whitney.csv", index=False)
    for col in ["Processing.Method", "Color", "Variety", "Country.of.Origin"]:
        eda.rate_by(df, col).to_csv(MET / f"eda_rate_by_{col.replace('.', '_').lower()}.csv")
    P.plot_score_distribution(df, C.EXCELLENT_THRESHOLD, FIG / "eda_score_distribution.png")
    P.plot_altitude_vs_score(df, FIG / "eda_altitude_vs_score.png")
    P.plot_geo_scatter(df, FIG / "eda_geography.png")
    P.plot_numeric_by_class(df, C.NUMERIC_FEATURES, FIG / "eda_numeric_by_class.png")
    P.plot_rate_by_category(eda.rate_by(df, "Country.of.Origin"), "Excellence rate by country (n ≥ 10)", FIG / "eda_rate_by_country.png")
    P.plot_rate_by_category(eda.rate_by(df, "Processing.Method"), "Excellence rate by processing method", FIG / "eda_rate_by_processing.png", min_n=1)
    plt.close("all")


def stage_unsupervised(df: pd.DataFrame) -> None:
    log("Unsupervised")
    df_u = data.impute_for_eda(df)
    X = U.numeric_matrix(df_u)
    y = df_u.loc[X.index, C.TARGET]
    pca, Z, scores, loadings = U.fit_pca(X)
    pd.DataFrame({"component": loadings.columns, "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(MET / "pca_variance.csv", index=False)
    loadings.round(3).to_csv(MET / "pca_loadings.csv")
    P.plot_pca_variance(pca, FIG / "pca_variance.png")
    P.plot_pca_loadings(loadings, 3, FIG / "pca_loadings.png")
    P.plot_pca_biplot(scores, loadings, y, FIG / "pca_biplot.png")
    bands, band_names = U.altitude_band(X["altitude_mean_meters"])
    emb = U.fit_tsne(Z)
    P.plot_embedding(emb, bands, band_names, "t-SNE of the agronomic variables, coloured by altitude band", FIG / "tsne_altitude.png")
    sil = U.kmeans_silhouettes(Z, scores)
    sil.round(3).to_csv(MET / "kmeans_silhouette.csv")
    P.plot_silhouette_scores(sil, FIG / "kmeans_silhouette.png")
    P.plot_k_distance(U.k_distances(scores[:, :3]), 7, FIG / "dbscan_kdistance.png")
    P.plot_dbscan_grid(scores, U.dbscan_sweep(scores[:, :3]), FIG / "dbscan_eps_sweep.png")
    plt.close("all")


def stage_models(X_train, y_train, X_test, y_test, quick: bool) -> dict[str, E.ModelResult]:
    log("Supervised models")
    n_boot = 200 if quick else 1000
    results: dict[str, E.ModelResult] = {}

    def run(name, est):
        t = time.time()
        results[name] = E.evaluate(name, est, X_train, y_train, X_test, y_test, n_boot=n_boot)
        m = results[name].metrics
        log(f"  {name:<32} AUPRC={m['auprc']:.3f}  F1={m['f1']:.3f}  ROC={m['roc_auc']:.3f}  ({time.time()-t:.0f}s)")

    for name, est in logistic.all_models().items():
        run(name, est)

    spw = boosting.imbalance_ratio(y_train)
    n_trees, best_its = boosting.freeze_n_estimators(X_train, y_train)
    n_trees_fe, best_its_fe = boosting.freeze_n_estimators(X_train, y_train, engineer=True)
    json.dump({"vanilla": {"n_estimators": n_trees, "best_iteration_per_fold": best_its},
               "feature_engineering": {"n_estimators": n_trees_fe, "best_iteration_per_fold": best_its_fe}},
              open(MET / "xgb_n_estimators.json", "w"), indent=2)
    run("XGBoost", boosting.build_xgboost(n_trees, spw))
    run("XGBoost + feature eng.", boosting.build_xgboost(n_trees_fe, spw, engineer=True))
    if not quick:
        run("XGBoost (grid search)", boosting.build_xgboost_grid(spw))
    run("XGBoost + SMOTE-NC", boosting.build_smotenc_xgboost(n_trees))

    run("Decision tree", forest.build_decision_tree())
    run("Random forest", forest.build_random_forest())

    # reduced random forest: columns chosen by *out-of-fold* permutation importance
    rf_best = results["Random forest"].estimator.best_estimator_
    perm = I.cv_permutation_importance(rf_best, X_train, y_train, n_repeats=10 if quick else 20)
    perm.to_csv(MET / "rf_permutation_importance_oof.csv", index=False)
    selected = perm.loc[perm["importance_mean"] > 0.005, "feature"].tolist()
    json.dump(selected, open(MET / "rf_reduced_selected_columns.json", "w"), indent=2)
    run("Random forest (reduced)", forest.build_reduced_random_forest(selected))
    P.plot_importance(perm, "importance_mean", "Mean AUPRC drop when permuted (out-of-fold)",
                      "Random forest — permutation importance", error="importance_std", path=FIG / "rf_permutation_importance.png")

    # tables & figures
    summary = E.summary_table(list(results.values()))
    summary.to_csv(MET / "model_comparison.csv")
    E.format_summary(list(results.values())).to_csv(MET / "model_comparison_pretty.csv")
    P.plot_metric_comparison(summary, "auprc", FIG / "model_comparison_auprc.png")
    P.plot_metric_comparison(summary, "f1", FIG / "model_comparison_f1.png")
    P.plot_confusion_matrices(list(results.values()), FIG / "confusion_matrices.png")
    headline = [results[k] for k in ["LogReg (unpenalised)", "XGBoost", "XGBoost + SMOTE-NC", "Random forest"] if k in results]
    P.plot_pr_curves(headline, FIG / "pr_curves.png")
    P.plot_roc_curves(headline, FIG / "roc_curves.png")

    # logistic regression: coefficients and GLM diagnostics
    log_results = {k: v for k, v in results.items() if k.startswith("LogReg")}
    coef = logistic.coefficient_table(log_results)
    coef.round(4).to_csv(MET / "logreg_coefficients.csv")
    P.plot_coefficients(coef, path=FIG / "logreg_coefficients.png")
    glm, design = logistic.fit_glm(X_train, y_train)
    with open(MET / "logreg_glm_summary.txt", "w") as fh:
        fh.write(str(glm.summary()))
    P.plot_glm_residuals(glm, design, C.NUMERIC_FEATURES, FIG / "logreg_deviance_residuals.png")
    P.plot_influence(glm, path=FIG / "logreg_influence.png")

    # leakage demonstrations
    E.threshold_leakage_demo(results["Random forest"]).to_csv(MET / "leakage_threshold_on_test.csv")
    naive, _ = boosting.naive_smote_demo(X_train, y_train, X_test, y_test, n_trees)
    naive.to_csv(MET / "leakage_naive_smote.csv")
    P.plot_leakage_bars(naive, title="Naive SMOTE: resampled-train vs test metrics", path=FIG / "leakage_naive_smote.png")
    plt.close("all")
    return results


def stage_interpret(results: dict[str, E.ModelResult], X_test) -> None:
    log("Interpretation")
    xgb, smote = results["XGBoost"], results["XGBoost + SMOTE-NC"]
    gain = I.xgb_importance(xgb.estimator)
    gain.to_csv(MET / "xgb_gain_importance.csv", index=False)
    gain_smote = I.xgb_importance(smote.estimator).rename(columns={"gain": "gain_smotenc"})
    comp = gain.merge(gain_smote[["feature", "gain_smotenc"]], on="feature", how="outer").fillna(0).rename(columns={"gain": "gain_vanilla"})
    comp = comp.sort_values("gain_vanilla", ascending=False)
    comp.to_csv(MET / "xgb_gain_vanilla_vs_smotenc.csv", index=False)
    P.plot_importance(gain, "gain", "Gain", "XGBoost — gain importance", path=FIG / "xgb_gain.png")
    P.plot_importance(gain.sort_values("weight", ascending=False), "weight", "Number of splits", "XGBoost — split count (weight)", path=FIG / "xgb_weight.png", color=P.SERIES[1])
    P.plot_importance_comparison(comp, ["gain_vanilla", "gain_smotenc"], "Gain importance: XGBoost vs XGBoost + SMOTE-NC", path=FIG / "xgb_gain_vanilla_vs_smotenc.png")
    values, _ = I.shap_values(xgb.estimator, X_test)
    I.shap_summary_table(values).to_csv(MET / "shap_mean_abs.csv", index=False)
    P.plot_shap_beeswarm(values, FIG / "shap_beeswarm_xgb.png")
    plt.close("all")
    values_fe, _ = I.shap_values(results["XGBoost + feature eng."].estimator, X_test)
    P.plot_shap_beeswarm(values_fe, FIG / "shap_beeswarm_xgb_fe.png")
    plt.close("all")


def stage_sensory(df: pd.DataFrame, quick: bool) -> None:
    log("Sensory track")
    mw = eda.mann_whitney_by_class(df, C.SENSORY_COLS[:-1])
    mw.to_csv(MET / "sensory_mann_whitney.csv", index=False)
    X, y = S.sensory_matrix(df)
    P.plot_corr_heatmap(X.corr(), "Correlation of sensory scores", FIG / "sensory_correlation.png")
    X_train, X_test, y_train, y_test = data.stratified_split(X, y, test_size=0.25)
    models = {"Logistic regression": S.build_logistic(), "Random forest": S.build_random_forest_grid()}
    S.cross_validated_scores(models, X_train, y_train).round(3).to_csv(MET / "sensory_cv_scores.csv")
    rows = []
    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
        proba = fitted[name].predict_proba(X_test)[:, 1]
        rows.append({"model": name, **E.compute_metrics(y_test, proba, 0.5)})
    pd.DataFrame(rows).set_index("model").round(3).to_csv(MET / "sensory_test_metrics.csv")
    logit, logit_table = S.fit_logit_statsmodels(X_train, y_train)
    logit_table.round(4).to_csv(MET / "sensory_logit_statsmodels.csv")
    lasso = S.lasso_path_coefficients(X_train, y_train)
    rf = getattr(fitted["Random forest"], "best_estimator_", fitted["Random forest"])
    gini = pd.Series(rf.feature_importances_, index=X.columns)
    perm = S.oof_permutation_importance(models["Random forest"], X_train, y_train, n_repeats=10 if quick else 20)
    mean_diff = mw.set_index("feature")["mean_difference"]
    ranking = S.final_ranking(logit_table, lasso, gini, perm, mean_diff)
    ranking.round(4).to_csv(MET / "sensory_feature_ranking.csv", index=False)
    P.plot_importance(ranking.rename(columns={"overall_score": "score"}), "score", "Combined importance (0–1)",
                      "Sensory dimensions ranked by association with excellence", path=FIG / "sensory_ranking.png")
    plt.close("all")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="fewer bootstrap draws / repeats, skip the XGBoost grid search")
    parser.add_argument("--stage", action="append", choices=["eda", "unsupervised", "models", "interpret", "sensory"],
                        help="run only the given stage(s); default: all")
    args = parser.parse_args()
    stages = set(args.stage or ["eda", "unsupervised", "models", "interpret", "sensory"])
    if "interpret" in stages:
        stages.add("models")

    FIG.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    P.set_style()
    np.random.seed(C.SEED)

    df, report = data.clean_data(data.load_raw(), data.load_coordinates())
    df = data.add_target(df)
    X, y = data.make_model_matrix(df)
    X_train, X_test, y_train, y_test = data.stratified_split(X, y)
    json.dump({"n_rows": int(len(df)), "n_excellent": int(y.sum()), "n_train": int(len(y_train)), "n_test": int(len(y_test)),
               "positives_train": int(y_train.sum()), "positives_test": int(y_test.sum())}, open(MET / "dataset_summary.json", "w"), indent=2)

    if "eda" in stages:
        stage_eda(df, report)
    if "unsupervised" in stages:
        stage_unsupervised(df)
    results = None
    if "models" in stages:
        results = stage_models(X_train, y_train, X_test, y_test, args.quick)
    if "interpret" in stages and results is not None:
        stage_interpret(results, X_test)
    if "sensory" in stages:
        stage_sensory(df, args.quick)
    log("done")


if __name__ == "__main__":
    main()
