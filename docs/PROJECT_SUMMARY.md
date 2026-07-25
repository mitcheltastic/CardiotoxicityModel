> [WARNING] STALE - OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

> ⚠️ STALE — OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

# Project Summary: In-Silico Cardiotoxicity Risk Prediction Ensembles

This document serves as a comprehensive handoff and record of the cardiotoxicity (Torsade de Pointes - TdP) risk prediction project. It outlines the dataset properties, methodology, critical data leakage discovery, performance results, and explainability evidence.

---

## 1. Project Overview

This project aims to develop a robust, explainable, in-silico cardiotoxicity (Torsade de Pointes) risk prediction model. Using 11 cardiac biomarkers derived from cellular action potential simulations, the model classifies drugs into a three-tier clinical risk category: **high**, **intermediate**, or **low** risk. The predictive framework is built on stacking and voting ensembles that combine machine learning (ML) base estimators (Random Forest, XGBoost, Logistic Regression, Support Vector Machines) and deep learning (DL) artificial neural networks (ANNs). The ultimate goal of the project is to persist a single, high-performance, deployable model ready for integration into a web-based clinical screening tool.

---

## 2. Dataset Properties

The analysis was performed across three datasets representing different experimental/simulation sources:
1. **`chantest`**: The primary reference dataset and target deployment split.
2. **`nanion`**: A dataset representing high-throughput automated patch-clamp recordings.
3. **`manual`**: A dataset compiled from manual patch-clamp measurements.

### Data Splits and Drug Disjointness
For each dataset, the data is split into training and testing sets with a **24k training / 32k testing** row configuration:
* **Training set:** 12 unique drugs, 2,000 simulated samples per drug (totaling 24,000 rows prior to missing value cleaning).
* **Testing set:** 16 unique drugs, 2,000 simulated samples per drug (totaling 32,000 rows prior to missing value cleaning).
* **Drug Disjointness:** The training and testing drug sets are completely disjoint (zero overlap). This ensures that performance estimates reflect the model's generalizability to completely unseen compounds.

### The 11 CiPA Biomarker Features
The model operates on the 11 biomarkers proposed by the Comprehensive In Vitro Proarrhythmia Assay (CiPA) initiative:
* `qNet` (Net charge)
* `dvdtmax` (Max rate of rise of action potential upstroke)
* `vmax` (Maximum action potential voltage)
* `vrest` (Resting membrane potential)
* `APD50` (Action potential duration at 50% repolarization)
* `APD90` (Action potential duration at 90% repolarization)
* `max_dv` (Max upstroke velocity/slope)
* `camax` (Peak calcium transient)
* `carest` (Diastolic calcium)
* `CaTD50` (Calcium transient duration at 50%)
* `CaTD90` (Calcium transient duration at 90%)

### Missing Value (NaN) Handling
Manual NaN handling was applied to clean the datasets before modeling. Only a small fraction of simulated rows contained incomplete outputs:
* **Dropped in Train split:** 25 rows (all from `chantest`, remaining train shape: 23,975 $\times$ 11).
* **Dropped in Test split:** 1 row (from `chantest`, remaining test shape: 31,999 $\times$ 11).

---

## 3. Methodology by Phase

### Phase 0/1: Data Validation & Preprocessing
* Checked dataset shapes, biomarker columns, disjoint drug sets, class balance, and NaNs.
* Fits a `StandardScaler` on the training split only and transforms both training and test splits separately per dataset to prevent test-set information leakage.

### Phase 2 + 2.5: Base Tuning and the Critical Leakage Discovery
* **Leakage Discovery:** In Phase 2, initial cross-validation using standard row-level `StratifiedKFold` yielded near-perfect validation Macro-F1 scores ($\approx 0.99$). However, Phase 2.5 diagnostic testing revealed that the models had memorized drug-level features. Evaluating a fresh RandomForestClassifier under a whole-drug holdout scheme caused the cross-validation Macro-F1 score to collapse:
  * **Row-level CV (StratifiedKFold):** **0.999542** Macro-F1
  * **Drug-level CV (GroupKFold):** **0.460310** Macro-F1
* **Mitigation:** The leakage was confirmed as drug-level memorization (since the 2,000 samples per drug share identical drug-level parameters). To fix this, all base models were re-tuned in Phase 2 using a leakage-free **`StratifiedGroupKFold(n_splits=4)`** cross-validation scheme, grouping by `drug_name` so that whole drugs are held out in each fold. This represents the key methodological contribution of the project.

### Phase 3: Combinatorial Ensemble Sweep
* Conducted a massive **612-configuration sweep** (3 datasets $\times$ 2 scenarios $\times$ 17 base estimator combinations $\times$ 6 ensemble configurations).
* Base learners evaluated: Random Forest (`RF`), XGBoost (`XGB`), Logistic Regression (`LR`), Support Vector Classifier (`SVM`), and three multi-layer perceptrons (`ANN1`, `ANN2`, `ANN3`).
* Every stacking ensemble that required internal cross-validation was strictly routed using `StratifiedGroupKFold(4)` grouped by `drug_name` using scikit-learn's metadata routing system (`enable_metadata_routing=True`).

### Phase 4: CiPA-Style Bootstrap Validation
* Performed a **10,000-iteration bootstrap validation** with drug-level resampling (randomly drawing one sample per drug out of its 2,000 rows, resulting in exactly 16 samples per iteration) on the test split.
* Pre-computed all test predictions to optimize loop runtime (reducing evaluation time from hours to under 3 minutes total).
* Computed medians and 95% confidence intervals (2.5th and 97.5th percentiles) for F1-macro, class recalls, and one-vs-rest class AUCs.

### Phase 5a/b/c: Persistence, SHAP, and Figures
* Serialized the winning `chantest` model, its aligned scaler, and metadata into a joblib bundle.
* Executed black-box SHAP explainability using `shap.KernelExplainer` (100 background, 500 evaluation samples).
* Generated publication-grade (300 DPI) thesis figures.

---

## 4. Key Results

### The Deployable Winner Model
The final deployable model selected for the web tool is the **`chantest` 3-Class Stacking Ensemble** with base learners **`RF` and `XGB`** and a **`RandomForestClassifier`** meta-learner.
* **Test Macro-F1 Point Estimate:** **0.828004** (Accuracy: **0.8241875**)
* **Bootstrap Test Macro-F1 Median [95% CI]:** **0.819373 [0.694017, 0.878307]**

### Deployable Model Class Breakdown (Chantest 3-Class)
Extracting the class-specific metrics for the deployable model from the Phase 3 sweep:

| Class | F1-Score | Recall |
| :--- | :--- | :--- |
| **High** (Class 0) | **0.795013** | **0.992375** |
| **Intermediate** (Class 1) | **0.767932** | **0.659571** |
| **Low** (Class 2) | **0.921067** | **0.920100** |

### Cross-Dataset Best Configurations (Scenario == 3class)
Performance comparison of the single best configuration for each dataset split:

| Dataset | Best Combo | Detail | Test Macro-F1 | Recall (High) | F1 (Intermediate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **chantest** | RF+XGB | Stacking (`meta=RF`) | **0.828004** | **0.992375** | **0.767932** |
| **nanion** | RF+LR | Stacking (`meta=RF`) | **0.685028** | **1.000000** | **0.668055** |
| **manual** | XGB+LR+SVM | Stacking (`meta=RF`) | **0.727566** | **0.980373** | **0.704773** |

### Bootstrap Summary Table (Median [95% CI])
Median and 95% confidence intervals from the 10,000-iteration bootstrap run:

| Metric | Chantest | Nanion | Manual |
| :--- | :--- | :--- | :--- |
| **Macro-F1** | 0.819373 [0.694017, 0.878307] | 0.679365 [0.623377, 0.754762] | 0.750000 [0.599868, 0.822222] |
| **AUC (High)** | 0.916667 [0.875000, 0.958333] | 0.916667 [0.875000, 0.958333] | 0.947917 [0.875000, 1.000000] |
| **AUC (Intermediate)** | 0.825397 [0.714286, 0.912698] | 0.658730 [0.571429, 0.746032] | 0.809524 [0.698413, 0.936508] |
| **AUC (Low)** | 1.000000 [0.836364, 1.000000] | 0.700000 [0.645455, 0.800000] | 0.872727 [0.745455, 0.963636] |
| **Recall (High)** | 1.000000 [0.750000, 1.000000] | 1.000000 [1.000000, 1.000000] | 1.000000 [0.750000, 1.000000] |
| **Recall (Intermediate)**| 0.714286 [0.428571, 0.857143] | 0.714286 [0.571429, 0.857143] | 0.714286 [0.571429, 1.000000] |
| **Recall (Low)** | 1.000000 [0.600000, 1.000000] | 0.400000 [0.400000, 0.600000] | 0.400000 [0.200000, 0.600000] |

### Key Findings
1. **Stacking vs. Voting:** Stacking ensembles consistently outperformed soft/hard voting ensembles (best stacking F1 was **0.828004** vs. best voting F1 of **0.821999**).
2. **Architecture Invariance:** While the optimal base learner combinations differed slightly across datasets, the winning meta-learning architecture remained invariant: **Stacking with a RandomForestClassifier meta-learner (`meta=RF`)** won Rank 1 on all three datasets.
3. **Task Difficulty:** Classifying the intermediate risk category remains the hardest prediction task across all datasets, with intermediate AUCs and F1s tracking significantly lower than the other classes.
4. **Safety & Zero-Critical-Misses:** For the target `chantest` dataset, the model achieved a **median recall on the HIGH class of 1.0000** (with a point estimate of **0.992375**). Most importantly, the model achieved **zero critical misses**—not a single high-risk drug sample was predicted as low-risk (verified by the confusion matrix).

---

## 5. Honest Caveats

* **Mixed-Paradigm Trade-off:** The best mixed-paradigm model (`mixed-all-7` Stacking `meta=RF`) achieved a test Macro-F1 of **0.816995** under our final leakage-free protocol. Although some mixed networks might show marginally higher scores under leaky scenarios, the pure-ML `RF+XGB` stacking model was chosen as the deployable model. It offers superior explainability, lower latency, and structural simplicity (no deep learning wrappers required).
* **Statistical Significance:** The 95% confidence interval for `chantest` Macro-F1 `[0.694017, 0.878307]` overlaps with `nanion`'s interval `[0.623377, 0.754762]`. Thus, while `chantest` shows the highest median performance, the difference is not strictly statistically significant.
* **Resting Potential Importance:** SHAP analysis ranked resting membrane potential (`vrest`) as the #2 driver of predictions. While classic cardiac safety literature focuses heavily on repolarization markers like `qNet` and `APD90`, `vrest` represents depolarization/excitability and warrants further clinical interpretation in the thesis.
* **Manual Dataset Exoneration:** The `manual` dataset showed exceptionally high training CV scores ($\approx 0.9788$ for XGBoost) under Grouped CV. While this could raise suspicions of lingering leakage, its test split performance dropped to a normal level (**0.727566** Macro-F1), validating that the grouped test set remains disjoint and leakage-free.

---

## 6. Explainability (SHAP Analysis)

We explained the reloaded deployable model using `shap.KernelExplainer` wrapping the full pipeline's `predict_proba` (operating on scaled inputs, 100-row background, 500-row evaluation sample).

### Global Feature Importance Rankings (mean(|SHAP|) from `shap_values.joblib`)
1. **`qNet`**: **0.161796** (Aggregated Mean Absolute SHAP)
2. **`vrest`**: **0.083903**
3. **`APD50`**: **0.071386**
4. **`max_dv`**: **0.062798**
5. **`APD90`**: **0.058283**
6. **`vmax`**: 0.030772
7. **`camax`**: 0.017091
8. **`CaTD50`**: 0.011256
9. **`CaTD90`**: 0.010575
10. **`dvdtmax`**: 0.007295
11. **`carest`**: 0.005708

### Alignment with CiPA Physiological Literature
* **`qNet` (Rank 1):** FDA/CiPA steering committee publications identify `qNet` (net charge) as the primary proarrhythmic biomarker because it integrates multi-channel blockade (IKr, ICaL, and INaL currents) into a single representation of repolarization reserve. Its high SHAP rank confirms the model uses the most physiologically sound metric.
* **`APD90` & `APD50` (Rank 3 & 5):** Action Potential Duration metrics directly represent prolongation of repolarization, which is the cellular driver of early afterdepolarizations (EADs) and QT prolongation that trigger clinical TdP events.

---

## 7. Artifacts & Deployment Pathway

All project artifacts are persisted in the `./artifacts/` directory:

* `./artifacts/deployable_chantest_3class.joblib`: The primary final deployable bundle containing the fitted StackingClassifier, the StandardScaler, class maps, and performance metadata.
* `./artifacts/phase3_protocolA_results.csv`: Complete tabular results (612 configs) of the Protocol A combinatorial ensemble sweep.
* `./artifacts/phase4_bootstrap.joblib`: Raw per-iteration array results of the 10,000-iteration bootstrap run.
* `./artifacts/phase4_bootstrap_summary.csv`: CSV table storing medians and 95% CIs of the bootstrap metrics.
* `./artifacts/shap_values.joblib`: Serialized raw SHAP values and ranked global/class importance tables.
* `./artifacts/base_models.joblib`: Base estimators tuned on row-level StratifiedKFold (kept for baseline backup).
* `./artifacts/base_models_grouped.joblib`: Leakage-free base estimators tuned on StratifiedGroupKFold.
* `./artifacts/keras/`: Folder containing ANN models tuned on row-level StratifiedKFold (backup).
* `./artifacts/keras_grouped/`: Folder containing 18 ANN models trained on StratifiedGroupKFold.
* `./artifacts/shap/`: Folder containing the three individual SHAP summary plots generated during Phase 5b.
* `./artifacts/figures/`: Folder containing the 8 publication-grade thesis figures.
  * `confusion_matrix_chantest.png`
  * `bootstrap_macrof1_distributions.png`
  * `per_class_auc_comparison.png`
  * `shap_global_importance.png`
  * `ensemble_family_comparison.png`
  * `shap_summary_high.png`
  * `shap_summary_intermediate.png`
  * `shap_summary_low.png`

> [!CAUTION]
> **Deployment Reload Requirement:**
> The serialized StackingClassifier utilizes custom wrapper classes (`PreFittedEstimatorWrapper`, `FittedHolder`, and `FinalEstimatorWrapper`). When unpickling the model in the web tool, the helper script `deploy_utils.py` **MUST** be present in the Python import path. Attempting to run `joblib.load()` without importing or having `deploy_utils` available will trigger a `ModuleNotFoundError`.

---

## 8. Status & Next Steps

* **Model Training & Sweeps:** Completed. All base models, deep networks, and ensembles have been fit and validated.
* **Explainability & CIs:** Completed. All statistics, confidence intervals, and SHAP values are persisted.
* **Remaining Work:**
  1. Thesis writeup (incorporating the suggested captions and results tables from this document).
  2. Construction of the web interface tool (which will load `deployable_chantest_3class.joblib` and `deploy_utils.py` to classify raw user-input biomarkers).
