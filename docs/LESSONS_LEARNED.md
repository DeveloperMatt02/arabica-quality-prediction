# Lessons learned: three kinds of leakage (and one artefact)

The first version of this project reported a random forest with F1 = 0.62 and AUPRC = 0.61, an XGBoost + SMOTE with F1 = 0.59, and a sensory model with ROC AUC = 0.98. None of those numbers survived a careful look at *where the information came from*. This document records what went wrong, how it was found, how big each effect was on this dataset, and what the refactored code does instead. It is the part of the project we are most proud of.

---

## 1. Pre-processing leakage

**What happened.** In the early notebooks the missing altitudes were filled with the mean altitude of the same country/region, and the rare varieties were pooled into the ten most frequent ones, *before* the train/test split - both statistics were computed on all 1,311 lots, test rows included.

**Why it matters.** A test lot with an unknown altitude received the mean of its region computed partly from other test lots; a variety's "frequency" was counted on data the model would later be scored on. Individually small, these steps make the evaluation slightly optimistic and, more importantly, make it impossible to say what the model would do on a genuinely new lot.

**How it was found.** Code review while merging the team notebooks (commit *"fixed data leakage (variety, altitude)"*): the imputation function was called on the full DataFrame long before `train_test_split`.

**What the repository does.** Every learned transformation is a scikit-learn transformer inside the model pipeline - `RegionAltitudeImputer`, `MoistureImputer`, the one-hot encoder with `max_categories` - so it is fitted on the training part of each cross-validation fold and on the training split for the final model (`coffee_quality/preprocessing.py`). `tests/test_features.py::test_altitude_imputer_learns_only_from_fit_data` guards the property.

**Size of the effect here.** Small (the region means barely change), but it is the cheapest of the three to get right and the one that reviewers check first.

## 2. Validation leakage

Three variants of the same mistake - letting evaluation data influence a decision that is then evaluated on the same data.

### 2a. Threshold chosen on the test labels

**What happened.** The decision-tree / random-forest notebook computed `precision_recall_curve(y_test, y_proba)` and picked the threshold maximising precision × recall - on the test split - then reported F1 at that threshold.

**Size of the effect.** On the refactored random forest, tuning the threshold on the test labels gives F1 = 0.44; the honest threshold (chosen on out-of-fold training predictions) gives F1 = 0.33. That is **+0.10 F1 from a single line of code**, and the original notebook reported a jump of the same order (0.52 → 0.59 → 0.65). AUPRC is unchanged (0.44 in both cases) because it does not depend on the threshold - which is why it is the headline metric of the project. `evaluation.threshold_leakage_demo` reproduces this on any `ModelResult`.

### 2b. Feature selection on the test split

**What happened.** The "reduced" random forest kept the columns whose permutation importance - computed on the test split - exceeded a threshold, was retrained, and was scored on the same test split.

**What the repository does.** `interpret.cv_permutation_importance` fits the pipeline on each training fold and permutes columns of the held-out fold; the reduced model is built from that ranking (`models/forest.py::build_reduced_random_forest`). The test split plays no part in the selection.

### 2c. Resampling before cross-validation

**What happened.** SMOTE was applied to the whole one-hot-encoded training set; the model was trained on the balanced set and the threshold picked on its predictions **for the resampled training rows**. Reported F1 ≈ 0.6 (and, internally, near-perfect training metrics).

**Size of the effect.** `boosting.naive_smote_demo` repeats the procedure: F1 = 0.97 and AUPRC = 0.997 on the resampled training set, **F1 = 0.38 and AUPRC = 0.40 on the untouched test split** - no better than the model without resampling. Two things go wrong at once: synthetic points interpolated between minority lots are trivially easy to classify, and plain SMOTE produces synthetic dummies such as "0.37 Bourbon".

**What the repository does.** `build_smotenc_xgboost` puts `SMOTENC` inside an `imblearn.Pipeline`, after the imputation/macro-area steps and before one-hot encoding. Samplers in an imblearn pipeline run only at `fit`, so every validation fold and the test split stay at their natural 8 % positive rate; SMOTE-NC copies categories instead of interpolating them.

**What we learned about oversampling.** Done correctly, SMOTE-NC changes *what the model looks at* (the gain of the lab-signature dummies collapses, origin gains weight) but not *how well it does* (test AUPRC 0.41 vs 0.41; out-of-fold 0.36 vs 0.34). On this dataset class imbalance is not the limiting factor.

## 3. Target leakage: the sensory paradox

**What happened.** The second track modelled excellence from the ten cupping scores and obtained ROC AUC ≈ 0.99, AUPRC ≈ 0.93 - numbers that in a presentation look like the "good" model next to the "bad" agronomic one.

**Why it is leakage.** `Total.Cup.Points` is (essentially) the *sum* of those scores, and `Excellent` is a threshold on it. A classifier on the sensory scores is not predicting quality; it is recovering the arithmetic that defines it. It also arrives too late: the scores exist only after the cupping session that the project is trying to anticipate.

**What the repository does.** The sensory scores and the total are excluded from the agronomic predictors by construction (`config.MODEL_INPUT_COLS`, checked by `tests/test_data.py::test_model_matrix_has_no_sensory_columns`). The sensory track is kept, clearly labelled, for the one question it can answer legitimately: the relative weight of the attributes (flavour > aftertaste ≈ balance > acidity > body > aroma).

## 4. The `Moisture == 0` artefact

Not a leakage in the textbook sense, but the same lesson: *know where a signal comes from before trusting it*.

**What happened.** SHAP and gain importance in the original XGBoost analysis singled out moisture as a top predictor, with "high moisture → not excellent" as the interpretation. True as far as it goes - but 229 lots (18 %) have `Moisture == 0.00`, which is physically impossible for green coffee. The zeros come from a few certifying laboratories (Colombia, Hawaii, Brazil, Taiwan) whose lots are excellent 12.7 % of the time against 7.3 % elsewhere. Part of the "moisture effect" was the model recognising the lab.

**How it was found.** Boxplots by class in the refactored EDA (`eda_numeric_by_class.png`): the excellent class has a mass of exactly-zero moisture values that the other class lacks.

**What the repository does.** `clean_data` recodes `0` as missing; `MoistureImputer` fills it with the training median and adds an explicit `Moisture_missing` indicator. The indicator stays in the model - it *is* information about the lot's provenance - but it is now visible, named, and reported (SHAP rank 11, GLM coefficient +0.33, p < 0.001) instead of hiding inside a physical variable.

## 5. Working rules we adopted

1. **The test split is read once**, at the very end, by `evaluation.evaluate`. No threshold, feature set, hyper-parameter, imputation constant or resampling decision may depend on it.
2. **If it is learned, it is a pipeline step.** Anything with a `fit` (imputers, encoders, scalers, samplers, selectors) lives inside the estimator that cross-validation clones.
3. **AUPRC first, F1 second, accuracy never** for an 8 %-positive problem; and every test metric comes with a bootstrap interval and its out-of-fold twin, because 21 positives cannot separate models that differ by 0.05.
4. **Reproduce the mistake.** The notebooks keep the wrong procedures as one-cell demonstrations next to the right ones. A number is easier to distrust when you can see how it was inflated.
5. **Look at the raw values of the top features.** A predictor that a physical variable cannot take (zero moisture, 190 km altitude) is a code, not a measurement.
