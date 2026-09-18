# Methodology

**Project:** Arabica Quality Prediction - Applied Statistics, Politecnico di Milano
**Companion documents:** [RESULTS.md](RESULTS.md) · [LESSONS_LEARNED.md](LESSONS_LEARNED.md) · [data/README.md](../data/README.md)

---

## Table of contents

1. [Question and target](#1-question-and-target)
2. [Data cleaning](#2-data-cleaning)
3. [Predictors and the feature pipeline](#3-predictors-and-the-feature-pipeline)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [Models](#5-models)
6. [Interpretation](#6-interpretation)
7. [Unsupervised analysis](#7-unsupervised-analysis)
8. [Sensory track](#8-sensory-track)
9. [Reproducibility](#9-reproducibility)

---

## 1. Question and target

The Coffee Quality Institute (CQI) grades Arabica lots on a 100-point scale summed from ten cupping attributes. Lots scoring **≥ 85** are marketed as *excellent* / *outstanding* specialty coffee and command a premium. The project asks whether that label can be anticipated from information that exists **before the cupping session**:

```
Excellent = 1  if  Total.Cup.Points >= 85  else 0
```

This is a binary classification with **8 % positives** (106 of 1,311 raw lots; 103 of 1,240 after cleaning). Two consequences drive every choice below:

* accuracy is meaningless (a constant "no" scores 92 %) - the headline metric is the **area under the precision–recall curve (AUPRC / average precision)**, whose random baseline equals the positive rate (0.08);
* with ~20 positives in any 20 % hold-out, single-split metrics are noisy - every test metric is reported with a **bootstrap confidence interval** and next to its **out-of-fold** counterpart on the training split.

## 2. Data cleaning

Implemented in `coffee_quality/data.py::clean_data`; every step is deterministic, row-wise and target-agnostic (nothing is *learned*). Row counts:

| Step | Rows |
|---|---|
| raw export | 1,311 |
| drop the placeholder record with `Total.Cup.Points == 0` (all sensory scores are 0) | 1,310 |
| drop the row with neither country nor region | 1,309 |
| drop implausible altitudes (< 200 m or > 3,300 m; the file contains values up to 190,164 m) | 1,266 |
| drop `Processing.Method == "Other"` (26 lots, none excellent, perfect separation in the logit) | 1,240 |

Further transformations that do not remove rows:

* 23 administrative / identifier columns are dropped (owner, mill, ICO number, certification body, dates, bag counts…);
* text columns are stripped; `Processing.Method` labels are shortened (`Washed / Wet → Washed` …); `Color` is lower-cased and `bluish-green` merged into `blue-green`;
* missing `Processing.Method`, `Color` and `Variety` become the explicit level **`Unknown`**. Missingness is informative (it is concentrated in Ethiopian lots, which are often excellent), so encoding it as a level is more honest than imputing the mode;
* the single missing `Quakers` value becomes 0; the column is later ignored (93 % zeros);
* **`Moisture == 0` is recoded as missing.** Green coffee is 10–12 % water; a zero is the export's "not measured" code. It affects 229 rows (18 %), mostly from a few certifying labs (Colombia, Hawaii, Brazil) whose lots have a 12.7 % excellence rate against 7.3 % elsewhere. Left as-is it acts as a lab-signature proxy; see [LESSONS_LEARNED.md](LESSONS_LEARNED.md#the-moisture--0-artefact);
* missing altitudes (225 rows, 18 %) are **kept**: their imputation is a learned step (§3).

Region-level coordinates (`data/raw/extracted_coords.csv`) are merged for the exploratory map and the unsupervised analysis only.

## 3. Predictors and the feature pipeline

### 3.1 What goes in

| Group | Columns |
|---|---|
| numeric | `Moisture`, `Category.One.Defects`, `Category.Two.Defects`, `altitude_mean_meters` |
| categorical | `MacroArea` (derived from country), `Color`, `Processing.Method`, `Variety` |

Deliberately **excluded**: the ten sensory scores and `Total.Cup.Points` (they define the target - see §8), `Country.of.Origin` and `Region` as such (36 and 300+ levels: a tree can memorise farms; they are collapsed into five macro-areas), latitude/longitude (region centroids, same memorisation risk, no agronomic content beyond altitude and macro-area), `Harvest.Year` (dirty free text, no expected causal role), identifiers.

### 3.2 The pipeline (`coffee_quality/preprocessing.py`)

Every model receives the same raw predictors and applies the same `sklearn.Pipeline`:

```
RegionAltitudeImputer  →  MoistureImputer  →  MacroAreaMapper  →  [AgronomicFeatureEngineer]  →  ColumnTransformer
```

| Step | What it learns at `fit` | What it does at `transform` |
|---|---|---|
| `RegionAltitudeImputer` | mean altitude per (country, region), per country, global median | fills missing altitude hierarchically: region → country → global |
| `MoistureImputer` | median moisture | fills missing moisture; adds the `Moisture_missing` indicator |
| `MacroAreaMapper` | - | `Country.of.Origin` → one of *Central/North America, South America, Africa, Mainland Asia, Maritime Asia/Pacific*; drops country and region |
| `AgronomicFeatureEngineer` (XGBoost + FE only) | - | `log1p` of both defect counts, `Has_Cat_One_Defect` flag, `log1p(altitude / moisture)`; drops raw defect counts |
| `ColumnTransformer` | scaler statistics; category levels | `StandardScaler` on numeric columns (identity for trees); `OneHotEncoder` with `max_categories=10` (rare varieties → *Rare*) and `handle_unknown="infrequent_if_exist"`; one reference level dropped for the unpenalised logit only (Caturra, Washed, green, Central/North America) |

Because all of this is inside the pipeline, it is re-fitted on the training part of every cross-validation fold and on the full training split for the final model. **No statistic of the validation or test rows ever enters a transformation.** The one-hot encoder emits a pandas DataFrame (`set_output(transform="pandas")`) so that column names survive to the classifier, which keeps SHAP and importance plots readable.

### 3.3 Split

One stratified 80/20 hold-out (`random_state=42`): 992 training lots (82 excellent) and 248 test lots (21 excellent). The test split is touched exactly once per model, at the very end.

## 4. Evaluation protocol

`coffee_quality/evaluation.py::evaluate` runs the same four steps for every estimator:

1. **Out-of-fold probabilities** on the training split with 5-fold stratified CV (`random_state=42`). When the estimator is a `GridSearchCV`, this is a *nested* CV: hyper-parameters are re-tuned on the inner folds (`random_state=123`) of every outer fold.
2. **Decision threshold** = the value maximising **F1** on the out-of-fold probabilities. The original notebooks used F2 for the logit and F1 elsewhere; one rule for all keeps the comparison fair.
3. Refit on the whole training split; predict the test split.
4. Metrics on the test split at that threshold - precision, recall, F1, balanced accuracy - plus the threshold-free ROC AUC and **AUPRC**; **95 % percentile-bootstrap intervals** (1,000 resamples of the test rows) for AUPRC, F1 and ROC AUC; the same metrics on the out-of-fold predictions.

Why this matters: choosing the threshold, the features or the resampling on data that is later used for scoring produces optimistic numbers. [LESSONS_LEARNED.md](LESSONS_LEARNED.md) quantifies each of those shortcuts on this dataset.

## 5. Models

All classifiers use class weighting (`class_weight="balanced"` or `scale_pos_weight`) unless the training folds are rebalanced by SMOTE-NC.

### 5.1 Logistic regression (`models/logistic.py`)

| Variant | Design matrix | Tuning |
|---|---|---|
| unpenalised | reference levels dropped | - (`lbfgs`, 5,000 iterations) |
| unpenalised + RFECV | reference levels dropped | recursive feature elimination, step 1, scored on average precision, ≥ 5 features |
| L2 (ridge) | all dummies | `C ∈ logspace(-3, 3, 30)`, inner 5-fold CV on average precision |
| L1 (lasso) | all dummies | same grid, `liblinear` |

The unpenalised model is refitted with `statsmodels.GLM(Binomial, var_weights = balanced class weights)` for standard errors, p-values, deviance-residual plots against each numeric predictor (binned mean curve) and an influence plot (leverage × standardised Pearson residual, bubble ∝ Cook's distance).

### 5.2 XGBoost (`models/boosting.py`)

Base configuration, chosen conservatively for ~80 positives: `max_depth=3`, `learning_rate=0.01`, `subsample=0.9`, `colsample_bytree=0.9`, `min_child_weight=2`, `tree_method="hist"`, `scale_pos_weight = n_neg / n_pos`.

* **Number of trees** is frozen by early stopping *inside* the CV folds (`freeze_n_estimators`): each outer training fold is split 80/20 into a fit set and an early-stopping set (patience 50 rounds, cap 4,000); the final `n_estimators` is the median best iteration × 1.1. The outer validation fold is never used for stopping.
* **Vanilla**, **+ feature engineering** (§3.2), and **grid search** (depth {3, 6} × learning rate {0.02, 0.05} × trees {150, 300} × class weight {½, 1, 2} × base ratio; nested CV on average precision).
* **+ SMOTE-NC**: an `imblearn.Pipeline` - altitude/moisture imputation → macro-area → `SMOTENC` on the four categorical columns → one-hot → XGBoost with `scale_pos_weight=1`. Samplers in an imblearn pipeline run only at `fit`, i.e. on training folds; validation and test rows are never resampled. SMOTE-NC (rather than SMOTE) is used because it interpolates numeric columns only and votes on categories, so no synthetic lot has a variety that is 0.37 Bourbon.

### 5.3 Decision tree and random forest (`models/forest.py`)

* Decision tree: grid over `max_depth ∈ {3, 5, 8, None}`, `min_samples_leaf ∈ {2, 4, 8, 16}`.
* Random forest: 300 trees, grid over `max_depth ∈ {3, 5, 7, None}` × `max_features ∈ {sqrt, None}`.
* **Reduced random forest**: same grid on the encoded columns whose **out-of-fold permutation importance** (§6) exceeds 0.005 AUPRC. The selection uses training folds only - the original notebook computed it on the test split.

## 6. Interpretation (`coffee_quality/interpret.py`)

* **XGBoost gain vs weight**: gain (mean loss reduction per split) against split count; their disagreement diagnoses over-splitting on continuous variables.
* **Gain under SMOTE-NC**: the same table for the resampled model, to see which variables the oversampling promotes or demotes.
* **SHAP** (`TreeExplainer`) on the test split: mean |SHAP| ranking and beeswarm plot for the vanilla and feature-engineered XGBoost.
* **Permutation importance, out-of-fold**: for each CV fold the pipeline is fitted on the training part and each *encoded* column of the validation part is permuted 20 times; the AUPRC drop is averaged over folds. Dummies are then aggregated back to their raw variable.

## 7. Unsupervised analysis (`coffee_quality/unsupervised.py`)

On the six standardised numeric variables (moisture, two defect counts, altitude, longitude, |latitude|), with missing values filled on the whole dataset (no train/test boundary exists here): PCA (variance, loadings, biplot coloured by class), t-SNE (perplexity 50 on the first four PCs, coloured by altitude band), k-means silhouettes for k = 2…6 (raw and PCA space), single/average/Ward dendrograms, DBSCAN with the k-distance heuristic and an ε sweep.

## 8. Sensory track (`coffee_quality/sensory.py`)

The cupping attributes are the components of `Total.Cup.Points`, hence any classifier built on them reconstructs the target rather than predicting it. The track is nevertheless kept to answer a different question, *which attributes weigh most in the verdict*, with a stratified 75/25 split, a standardised logistic regression (sklearn + statsmodels for p-values and odds ratios per SD), an L1-penalised path, a random forest (grid on depth and leaf size), out-of-fold permutation importance, and a **combined ranking** that averages the four min–max-scaled importance measures. Near-constant attributes (`Uniformity`, `Clean.Cup`, `Sweetness`: median 10/10 in both classes) and the global `Cupper.Points` are excluded.

## 9. Reproducibility

* `scripts/run_pipeline.py` regenerates every table (`reports/metrics/`) and figure (`reports/figures/`); `--quick` reduces bootstrap draws and skips the XGBoost grid; `--stage` selects stages.
* Seeds: `SEED = 42` for the split, the outer CV, the models and the bootstrap; `SEED + 81` for inner CVs.
* `tests/` (pytest, 22 tests) covers the cleaning invariants, each transformer (including that imputers learn only from the data they are fitted on), the column selector, the threshold rule, the bootstrap and the end-to-end protocol. GitHub Actions runs them on Python 3.10 and 3.12.
* Library versions used for the committed numbers: scikit-learn 1.8, XGBoost 3.2, imbalanced-learn 0.14, SHAP 0.51, statsmodels 0.15, pandas 3.0. Other versions reproduce the qualitative results; the last digits may differ.
