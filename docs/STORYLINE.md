# PROJECT STORYLINE & TECHNICAL NARRATIVE
**Repository Context**: Cardiotoxicity Risk Assessment via Ensemble Machine Learning and Conformal Prediction  
**Document Purpose**: A self-contained, plain-language record of technical development, validation, data integrity, and methodological audit.

---

## 1. THE PROBLEM

### Context: Cardiotoxicity Risk Screening
Torsade de Pointes (TdP) is a rare but potentially fatal cardiac arrhythmia (heart rhythm disorder). Certain pharmaceutical compounds induce cardiotoxicity by altering electrical activity in cardiac cells, delaying cardiac repolarization and prolonging the QT interval. Because TdP can lead to sudden cardiac death, evaluating cardiotoxicity risk is a fundamental safety requirement in pharmaceutical development.

### Computational Biomarker Pipeline
To evaluate cardiotoxicity risk *in silico* (via computer simulation), experimental electrophysiology measurements of drug-channel interactions are processed through biophysical cell models to generate simulated cellular action potentials and intracellular calcium transients. From these dynamic simulations, 11 summary features—referred to in dataset headers as **biomarkers**—are extracted:
1. `qNet`
2. `dvdtmax`
3. `vmax`
4. `vrest`
5. `APD50`
6. `APD90`
7. `max_dv` *(Note: The physical definition of `max_dv` is not documented in this repository. Per `DATA_AUDIT_REV.md`, it spans negative to positive float values, e.g. -1.024670 to +3.607143).*
8. `camax`
9. `carest`
10. `CaTD50`
11. `CaTD90`

### 3-Tier Risk Classification and Clinical Safety Goals
The clinical objective is to classify drugs into one of three ordinal risk categories:
- **High Risk (Class 0 in repository)**: High propensity for TdP; dangerous cardiotoxin requiring strict restriction.
- **Intermediate Risk (Class 1 in repository)**: Moderate TdP risk; requires clinical monitoring.
- **Low Risk (Class 2 in repository)**: Low/negligible TdP risk; safe from a repolarization standpoint.

From a clinical standpoint, a model must not only maintain overall accuracy but strictly avoid **under-triage** (predicting a High-risk drug as Intermediate or Low risk). Under-triage is an unsafe failure mode that could allow a lethal cardiotoxin to be administered unmonitored. Conversely, **over-call** (predicting a Low-risk drug as Intermediate or High risk) is a conservative failure mode that recommends extra safety monitoring, preserving patient safety.

---

## 2. THE DATA

### Benchmark Library, Splits, and Cell Simulations
The project evaluates machine learning models on a benchmark library of **28 drugs**:
- **12 Training Drugs**: *bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, sotalol, terfenadine, verapamil*. (Used for base/meta-learner training and conformal calibration).
- **16 Held-Out Test Drugs**: *astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib*. (Used strictly for cross-platform validation and operating-point evaluation).

For each drug, stochastic cell simulations generate **2,000 simulated cells** to capture variability across individual cell responses. Thus, each drug is represented by 2,000 rows of 11 biomarker measurements.

### Electrophysiology Platforms
To test whether models generalize across experimental measurement techniques, datasets are provided from three patch-clamp electrophysiology platforms:
1. **Chantest**: High-throughput electrophysiology platform (Primary baseline for model training and calibration).
2. **Nanion**: Secondary electrophysiology platform.
3. **Manual**: Manual patch-clamp electrophysiology platform.

### Pre-Normalized `_rev` Datasets
All experiments in this repository utilize the revised (`_rev`) dataset files. These datasets are pre-normalized against control simulations. Consequently, **no additional feature scaling** (such as `StandardScaler` or `MinMaxScaler`) was applied in this project.  
*Note*: The exact mathematical normalization formula applied upstream to generate `_rev` features from raw biophysical outputs is not documented in this repository.

---

## 3. DATA INTEGRITY WORK

### Data Audit & Sentinel Cell Discovery
Before model training, a systematic data audit (`DATA_AUDIT_REV.md`) verified dataset array shapes, data types, missing values, and sentinel error strings across all three platform files.

The forensic audit (`DATA_AUDIT_REV_PART2.md`) identified **exactly 26 corrupted rows** containing the Excel error string `'#VALUE!'` across 75 corrupted cells:
- **25 rows (75 cells)** in `manual_training_scaled_rev.csv`: All belonging to *mexiletine* simulations, spanning columns `APD50`, `APD90`, and `max_dv`.
- **1 row (1 cell)** in `manual_testing_scaled_rev.csv`: Belonging to an *ibutilide* simulation, in column `CaTD90`.
- **0 sentinel rows** were found in any Chantest or Nanion dataset files.

### Drop-Not-Impute Decision
A strict decision was made to **drop rather than impute** these 26 rows. Imputing values in biophysically simulated cell outputs could introduce artificial correlations or unphysical state variables. Because 25 rows represent a tiny fraction (1.25%) of mexiletine's 2,000 rows (leaving 1,975 rows) and 1 row represents 0.05% of ibutilide's rows, dropping them preserves absolute data purity without impairing statistical power.

### Hardened Loader & Audit Trail
The canonical data loader (`src/data_loader_rev.py`) was hardened with strict assertion guards (`LOADER_VERIFICATION_REV.md`). It automatically detects sentinel strings, verifies feature order, drops invalid rows, and logs every dropped row to the audit trail file `logs/DROPPED_ROWS_REV.csv`.

---

## 4. THE MODEL

### Model Architecture: Stacking Classifier
To capture non-linear interactions between biomarkers, an ensemble **Stacking Classifier** (`StackingClassifier`) architecture was implemented:
- **Level 1 Base Learners**: 
  1. **Random Forest (RF)**: An ensemble of decision trees trained on bootstrap sub-samples that averages individual tree probability predictions.
  2. **XGBoost (XGB)**: A gradient-boosted decision tree framework that iteratively trains trees to minimize residual classification errors.
- **Level 2 Meta-Learner**: A secondary Random Forest that takes the out-of-fold class probability predictions ($p_{	ext{high}}, p_{	ext{int}}, p_{	ext{low}}$) of the Level 1 base learners as inputs and outputs final ensembled class probabilities.

### Grouped Cross-Validation
Because 2,000 simulated rows represent the same physical drug, naive row-level cross-validation would split rows of the same drug across training and validation folds. This causes severe **data leakage**, as the model would memorize drug-specific biomarker signatures.

To prevent leakage, Level 1 out-of-fold probability generation strictly enforced **Grouped Cross-Validation** via `StratifiedGroupKFold` ($k=4$, grouped by `drug_name`). This guarantees that every out-of-fold prediction for the meta-learner is generated by base models trained on entirely unseen drugs.

### Tuned Hyperparameters & Model Provenance
Model hyperparameters were tuned on the Chantest training set and frozen in `models/chantest_stacking_metadata_rev.json`:
- **Random Forest Base Learner**: `n_estimators=400`, `criterion='entropy'`, `max_depth=None`.
- **XGBoost Base Learner**: `n_estimators=300`, `learning_rate=0.1`, `max_depth=10`.
- **Meta-Learner**: `RandomForestClassifier` with default parameters.
- **Stacking Configuration**: `stack_method='predict_proba'`, `passthrough=False`, `seed=42`.
- **Model Artifact Hash**: SHA-256 `A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2` (`models/chantest_stacking_model_rev.joblib`).

---

## 5. RESULTS AS THEY CAME IN

### Performance Metrics & Definitions
- **Macro-F1**: The unweighted arithmetic mean of F1 scores across all three risk classes ($rac{F1_{	ext{high}} + F1_{	ext{int}} + F1_{	ext{low}}}{3}$), giving equal weight to rare and frequent classes.
- **Row-level vs. Drug-level Evaluation**:
  - *Row-level*: Evaluates predictions on every individual cell simulation row (~32,000 rows per test platform).
  - *Drug-level*: Averages the 2,000 predicted row probabilities for a drug to form a single drug-level mean probability vector before taking the `argmax` or evaluating conformal sets.

### Empirical Performance Summary (16 Held-Out Test Drugs)
Per `docs/PAPER_NUMBERS.md`, point estimates and drug-level bootstrap confidence intervals (10,000 iterations, seed 42) were evaluated across platforms:

1. **CHANTEST (Primary Baseline)**:
   - **Row-level Macro-F1**: `0.688289` (Source: `docs/STEP3_STACKING_CHANTEST.md`)
   - **Drug-level Macro-F1**: Point Estimate = `0.694444`, Bootstrap Median = `0.679739` [95% CI: `0.391534`, `0.896825`] (Source: `TrainAI/results/bootstrap_distributions_rev.csv`)
   - **Per-Class F1 (Drug-Level)**: High = `0.666667`, Int = `0.666667`, Low = `0.750000`
   - **Confusion Matrix (High/Int/Low)**: High `[3, 1, 0]`, Int `[2, 5, 0]`, Low `[0, 2, 3]`

2. **NANION (Cross-Platform)**:
   - **Row-level Macro-F1**: `0.632400` (Source: `docs/STEP4_CROSS_PLATFORM.md`)
   - **Drug-level Macro-F1**: Point Estimate = `0.621032`, Bootstrap Median = `0.595238` [95% CI: `0.311111`, `0.851852`] (Source: `TrainAI/results/bootstrap_distributions_rev.csv`)
   - **Per-Class F1 (Drug-Level)**: High = `0.666667`, Int = `0.625000`, Low = `0.571429`
   - **Confusion Matrix (High/Int/Low)**: High `[3, 1, 0]`, Int `[2, 5, 0]`, Low `[0, 3, 2]`

3. **MANUAL**:
   - **Row-level Macro-F1**: `0.692524` (Source: `docs/STEP4_CROSS_PLATFORM.md`)
   - **Drug-level Macro-F1**: Point Estimate = `0.754762`, Bootstrap Median = `0.738889` [95% CI: `0.479347`, `0.944056`] (Source: `TrainAI/results/bootstrap_distributions_rev.csv`)
   - **Per-Class F1 (Drug-Level)**: High = `0.800000`, Int = `0.714286`, Low = `0.750000`
   - **Confusion Matrix (High/Int/Low)**: High `[4, 0, 0]`, Int `[2, 5, 0]`, Low `[0, 2, 3]`

*Note on Bootstrap Widths*: The 95% bootstrap confidence intervals are relatively wide (e.g., `[0.39, 0.90]` on Chantest). This width reflects the fundamental sample size limit of $n=16$ test drugs: resampling 16 drugs with replacement frequently omits difficult edge-case drugs, causing large shifts in macro-averaged metrics.

### Baseline Base Learner Reference Performance
Per `docs/STEP11_BASE_LEARNER_REFERENCE.md` & `docs/PAPER_NUMBERS.md` (evaluated on frozen base estimators without retraining):
- **Random Forest (RF) Base Learner**:
  - Chantest: Drug-level Macro-F1 = `0.877778` (14/16 correct, **0 under-triage**)
  - Nanion: Drug-level Macro-F1 = `0.819373` (13/16 correct, **0 under-triage**)
  - Manual: Drug-level Macro-F1 = `0.755556` (12/16 correct, **1 under-triage** [`clarithromycin`]) (Source: `docs/STEP11_BASE_LEARNER_REFERENCE.md` & `docs/STEP13C_RF_CONFORMAL_FIXED.md`)
- **XGBoost (XGB) Base Learner**:
  - Chantest: Drug-level Macro-F1 = `0.819373` (13/16 correct, **0 under-triage**)
  - Nanion: Drug-level Macro-F1 = `0.754762` (12/16 correct, **0 under-triage**)
  - Manual: Drug-level Macro-F1 = `0.819373` (13/16 correct, **0 under-triage**)

**Comparative Finding**: On point estimates, both RF alone and XGB alone outperform the full STACK model across all three platforms. However, 10,000-iteration paired bootstrap tests (`docs/STEP11_BASE_LEARNER_REFERENCE.md` & `docs/STEP17_CI_RECONCILIATION.md`) show that all six confidence intervals for the difference (STACK minus base learner) contain zero (e.g. STACK - RF on Chantest: median `-0.187302`, 95% CI `[-0.448155, +0.000000]`).

**Directional Test (STEP 17)**: Evaluating the number of drugs where STACK predicts correctly while RF predicts incorrectly yields **0 drugs on Chantest**, **0 drugs on Nanion**, and **1 drug on Manual** (*clarithromycin*). Because STACK is correct and RF wrong on 0 drugs for Chantest and Nanion, the upper bound of STACK - RF difference is mathematically bounded by $\le 0.000000$ on those platforms.

---

## 6. META-LEARNER DIAGNOSTICS & OVERRIDING BEHAVIOR

### Diagnostic Analysis (16 Test Drugs)
A detailed evaluation (`docs/STEP12_STACK_VS_BASE_DIAGNOSTIC.md`) investigated why STACK performs worse than its individual base learners. Exactly 3 drugs on Chantest, 3 drugs on Nanion, and 1 drug on Manual were correctly predicted by RF but incorrectly predicted by STACK:
- **Chantest Platform (3 drugs)**:
  1. *disopyramide* (True HIGH): RF and XGB both correctly predict HIGH, but STACK overrides both to predict Intermediate (**Category A: Overrides both correct base learners**, causing under-triage).
  2. *loratadine* (True LOW): RF and XGB both correctly predict LOW, but STACK overrides both to predict Intermediate (**Category A**).
  3. *tamoxifen* (True LOW): RF predicts LOW (correct), XGB predicts Intermediate (wrong); STACK follows XGB to predict Intermediate (**Category B: Follows wrong base learner**).
- **Nanion Platform (3 drugs)**:
  1. *disopyramide* (True HIGH): RF and XGB both correctly predict HIGH; STACK overrides both to Intermediate (**Category A**).
  2. *loratadine* (True LOW): RF and XGB both correctly predict LOW; STACK overrides both to Intermediate (**Category B**).
  3. *tamoxifen* (True LOW): RF predicts LOW; STACK predicts Intermediate (**Category B**).
- **Manual Platform (1 drug)**:
  1. *tamoxifen* (True LOW): RF and XGB both correctly predict LOW; STACK overrides both to Intermediate (**Category A**).

### Characteristic Pathology
All characteristic errors made by STACK compress predictions toward the **Intermediate** class. Notably, *disopyramide* (True HIGH) is correctly classified as HIGH by both base learners across all three platforms, but STACK misclassifies it as Intermediate on Chantest and Nanion, introducing unsafe under-triage.

### Methodological Explanation
Neutrally considered, the meta-learner was trained on Level 1 out-of-fold predictions when base learners were fitted on only 9 drugs per fold ($k=4$ grouped fold), whereas during test inference, base learners were fitted on all 12 training drugs. A causal relationship between this cross-validation fold mismatch and meta-learner probability distortion has not been experimentally proven in this repository.

---

## 7. THE CONFORMAL LAYER

### Prediction Sets and Safety Principles
Standard machine learning models output a single hard prediction (`argmax`), which forces a forced choice even when the model is highly uncertain. **Conformal Prediction** replaces hard labels with **prediction sets** $\mathcal{C}(X) \subseteq \{	ext{High}, 	ext{Intermediate}, 	ext{Low}\}$. A prediction set contains all classes whose predicted probability clears an inclusion threshold $1 - \hat{q}$. If a drug's classification is ambiguous, the set expands to include multiple classes (e.g., `['High', 'Intermediate']`).

### Severity Map & Outcome Directionality
Repository label integers (`high=0`, `intermediate=1`, `low=2`) are inverted relative to clinical risk. To prevent logic errors, an explicit severity map was defined:
$$	ext{SEVERITY} = \{0: 2 	ext{ (High)}, 1: 1 	ext{ (Intermediate)}, 2: 0 	ext{ (Low)}\}$$

The **Clinical Action** of a prediction set is defined as the most severe class present in the set:
$$	ext{Action} = \min_{c \in \mathcal{C}(X)} c \quad (	ext{where } 0=	ext{High} > 1=	ext{Int} > 2=	ext{Low})$$

Outcomes are classified by comparing Action severity against True severity:
- **Correct Outcome**: $	ext{Action Severity} == 	ext{True Severity}$.
- **Over-call (Safe Direction)**: $	ext{Action Severity} > 	ext{True Severity}$. The model recommends extra monitoring for a lower-risk drug.
- **Under-triage (Unsafe Direction)**: $	ext{Action Severity} < 	ext{True Severity}$. The model misclassifies a high-risk drug as safe.

### Formal Empty Set Abstain Rule
Under strict conformal prediction (`docs/STEP13C_RF_CONFORMAL_FIXED.md`), when no class probability clears threshold $1 - \hat{q}$ ($\mathcal{C}(X) = \emptyset$), the outcome is **ABSTAIN** ($	ext{set\_size} = 0$). Abstain predictions are tracked as a distinct metric and are NOT counted as correct, over-called, or under-triaged. An earlier implementation that filled empty sets with all three classes (`[0, 1, 2]`) was retracted.

### Leave-One-Drug-Out (LODO) Calibration for STACK and RF
Calibration was performed using Leave-One-Drug-Out (LODO) cross-validation across the 12 Chantest training drugs.
- **STACK LODO Scores** (`TrainAI/results/conformal_lodo_scores_rev.csv`):
  $$[0.045365, 0.094280, 0.152855, 0.163680, 0.236885, 0.255295, 0.408085, 0.453940, 0.630580, 0.703345, 0.762830, 0.986695]$$
- **RF LODO Scores** (`docs/STEP13C_RF_CONFORMAL_FIXED.md` & `results/conformal_rf_rev.csv`):
  $$[0.023518, 0.025005, 0.102810, 0.102904, 0.104039, 0.242225, 0.273810, 0.322894, 0.368977, 0.378064, 0.533987, 0.904169]$$

### Discrete Rank $\hat{q}$ & Operating Point Selection
Conformal quantile $\hat{q}(k)$ corresponds to the $k$-th smallest LODO score out of 12. For calibration size $n=12$, nominal coverage is strictly $1 - lpha = k/13$:
- **Rank 12**: $\hat{q} = 0.986695$ (STACK) / $0.904169$ (RF), Nominal Coverage $= 12/13 pprox 0.923077$
- **Rank 11**: $\hat{q} = 0.762830$ (STACK) / $0.533987$ (RF), Nominal Coverage $= 11/13 pprox 0.846154$
- **Rank 10**: $\hat{q} = 0.703345$ (STACK) / $0.378064$ (RF), Nominal Coverage $= 10/13 pprox 0.769231$

### Operating Point Evaluation (Rank 11)
Evaluating Rank 11 across platforms yields:
1. **STACK + Conformal (Rank 11, $\hat{q}=0.762830$, threshold $0.237170$)**:
   - **CHANTEST**: 11 Correct, 5 Over-called, **0 Under-triaged**, **0 Abstain** | Coverage: 12/16 (0.7500) | Avg Set Size: 1.1250
   - **NANION**: 11 Correct, 5 Over-called, **0 Under-triaged**, **0 Abstain** | Coverage: 12/16 (0.7500) | Avg Set Size: 1.1250
   - **MANUAL**: 11 Correct, 5 Over-called, **0 Under-triaged**, **0 Abstain** | Coverage: 14/16 (0.8750) | Avg Set Size: 1.3750
2. **RF + Conformal (Rank 11, $\hat{q}=0.533987$, threshold $0.466013$)**:
   - **CHANTEST**: 14 Correct, 2 Over-called, **0 Under-triaged**, **0 Abstain** | Coverage: 14/16 (0.8750 vs nom 0.8462, gap +0.0288) | Avg Set Size: 1.0000
   - **NANION**: 13 Correct, 3 Over-called, **0 Under-triaged**, **0 Abstain** | Coverage: 13/16 (0.8125 vs nom 0.8462, gap -0.0337) | Avg Set Size: 1.0000
   - **MANUAL**: 13 Correct, 3 Over-called, **0 Under-triaged**, **0 Abstain** | Coverage: 13/16 (0.8125 vs nom 0.8462, gap -0.0337) | Avg Set Size: 1.0625

### Mechanism of Under-Triage Removal for `clarithromycin` (Manual)
On the Manual platform, hard-label RF misclassifies *clarithromycin* (True Intermediate) as Low (under-triage), with probabilities $p_{	ext{high}}=0.0, p_{	ext{int}}=0.476429, p_{	ext{low}}=0.523571$. At Rank 11 (threshold $0.466013$), both $p_{	ext{int}}$ and $p_{	ext{low}}$ clear threshold, expanding the conformal prediction set to `['intermediate', 'low']` (size 2). Taking $\min(	ext{set})$ yields Intermediate (1), which **eliminates under-triage through legitimate set expansion**, rather than through empty-set abstain rules.

---

## 8. VALIDITY CHECKS

### SHAP Analysis & The `vrest` Anomaly
SHAP (SHapley Additive exPlanations) was computed using `KernelExplainer` on the frozen stacking model (`TrainAI/results/shap_global_importance_rev.csv`). The global feature importance ranking placed **qNet** first ($	ext{mean}|	ext{SHAP}| = 0.145534$) and **vrest** second ($	ext{mean}|	ext{SHAP}| = 0.115489$).

`vrest`'s Rank 2 position is highly stable across random seeds and background sizes (testing seeds 7, 42, 2024 and background sizes 100, 300).

However, a detailed split audit (`docs/STEP8_SHAP_DIAGNOSTIC.md`) revealed a structural anomaly:
- In Random Forest base learners, `vrest` accounted for 1,533 tree splits at hyper-precise 8-decimal thresholds (`min: -1.00038034`, `median: -1.00004244`, `max: -0.99996367`).
- In XGBoost base learners, `vrest` was used **zero times** across all 300 trees.

### Drug Range Overlaps and Permutation Test Results
Per `docs/STEP8_SHAP_DIAGNOSTIC.md`:
- Feature range overlap analysis across drug distributions:
  - On the 12 training drugs (66 possible pairs): `qNet` has 29 overlapping pairs, `vrest` has 40 overlapping pairs, and `carest` has 61 overlapping pairs.
  - On the 16 test drugs (120 possible pairs): `qNet` has 54 overlapping pairs, `carest` has 107 overlapping pairs, and `vrest` has 108 overlapping pairs.
- In a permutation diagnostic on Chantest test drugs (`docs/STEP8_SHAP_DIAGNOSTIC.md`):
  - Permuting `vrest` values changes on average 1.0 out of 16 drug-level predictions (zero predictions on many random seeds), with a mean drug-level macro-F1 drop of 0.073.
  - Permuting `qNet` values changes on average 3.0 out of 16 drug-level predictions, with a mean drug-level macro-F1 drop of 0.108.

### Diagnostic Conclusion
On the 16 held-out test drugs, vrest is the least separable of the features examined (108 of 120 drug pairs overlap, versus 54 of 120 for qNet), permuting it changes on average 1.0 of 16 drug-level predictions, and the XGBoost base learner does not split on it at all. Its stable rank-2 position by mean|SHAP| is therefore not supported by a corresponding effect on predictions. This is consistent with instability of the model-agnostic explainer on a feature whose coefficient of variation is on the order of 1e-4, rather than with discriminative signal. The interpretability discussion accordingly rests on qNet, which ranks first by mean|SHAP| and has the largest permutation effect.

---

## 9. ERRORS CAUGHT DURING THE WORK

As part of process history, several intermediate errors and documentation discrepancies were identified and corrected during development:

a. **Data Leakage & Score Change Misinterpretation**: An early summary claimed that a `0.694` drug-level macro-F1 score reflected "fixing data leakage" and that earlier scores were "inflated". In fact, both protocol and cross-validation setup were identical; only the underlying dataset changed (moving to `_rev`). RF and XGBoost are scale-invariant. The exact cause of the score difference remains unestablished. Corrected in `docs/STEP3_STACKING_CHANTEST.md`.
b. **Miscounted Corrupted Cells & Rows**: An early verdict block reported 25 sentinel cells across 10 rows. Comprehensive forensic audit confirmed the correct figures are **75 corrupted cells across 25 unique rows** in training plus 1 cell/row in testing (total 26 rows). Corrected in `docs/DATA_AUDIT_REV.md` and `docs/DATA_AUDIT_REV_PART2.md`.
c. **Mexiletine Remaining Row Count**: The same verdict block reported Policy A leaving mexiletine with 1,990 rows. The correct remaining count after dropping 25 corrupted rows is **1,975 rows**. Corrected in `docs/DATA_AUDIT_REV_PART2.md`.
d. **Degenerate Leave-One-Out Fragility Analysis**: An early sensitivity analysis held $lpha = 1/13$ fixed while dropping calibration drugs down to $n=11$. Because $\lceil 12(1-1/13) 
ceil = 12 > 11$, $\hat{q}$ defaulted to 1.0 (inclusion threshold 0.0), assigning all drugs High risk regardless of model probabilities. Retracted and replaced in `docs/STEP5_CONFORMAL.md` and `docs/STEP6_CONFORMAL_OPERATING_POINT.md`.
e. **Premature Operating Point Selection**: Early documentation claimed Rank 12 ($\hat{q}=0.986695$) was the tightest safe operating point. Scanning all ranks proved Rank 11 ($\hat{q}=0.762830$) was the true tightest rank achieving zero under-triage across all three platforms. Corrected in `docs/STEP6_CONFORMAL_OPERATING_POINT.md`.
f. **Degenerate Under-Triage Confidence Interval**: Early bootstrap reports listed a 95% CI of `[0.000000, 0.000000]` for under-triage. Because 0 under-triages occurred in the 16 test drugs, every bootstrap sample contained 0 under-triages, forcing a point interval. Superseded by the **Rule of Three upper bound** ($3/16 = 0.1875$ or 18.75%). Corrected in `docs/STEP9_BOOTSTRAP.md` and `docs/PAPER_NUMBERS.md`.
g. **Coverage Conflated with Correct-Outcome Proportion**: Initial drafts reported 11/16 (0.6875) as empirical coverage. Recomputation clarified that 11/16 is the correct-outcome proportion, while empirical coverage is 12/16 (0.7500) on Chantest and Nanion, and 14/16 (0.8750) on Manual. Corrected in `docs/PAPER_NUMBERS.md`.
h. **Fabricated Row Drop Reason**: Early section A attributed the 26 dropped rows to being "flagged as non-steady-state or excluded in raw data". Forensic audit established the true cause was Excel `'#VALUE!'` error strings in biomarker columns. Corrected in `docs/DATA_AUDIT_REV_PART2.md` and `docs/LOADER_VERIFICATION_REV.md`.
i. **Coverage-Miss Drug List Conflict**: A conflict between `STEP6` and `PAPER_NUMBERS` regarding which drugs missed coverage on Chantest and Nanion was resolved by independent recomputation, confirming `STEP6`'s list: Chantest misses `['domperidone', 'loratadine', 'pimozide', 'tamoxifen']`; Nanion misses `['domperidone', 'metoprolol', 'pimozide', 'tamoxifen']`. Corrected in `docs/PAPER_NUMBERS.md`.
j. **vrest separability misreported**: An intermediate summary stated vrest had zero overlapping drug ranges, implying per-drug separability. Direct recomputation from the source data (`docs/STEP10_VREST_OVERLAP_RECHECK.md`) established 40 overlapping pairs of 66 on the training split and 108 of 120 on the test split. The zero-overlap claim and the per-drug-offset interpretation derived from it are retracted.
k. Kalibrasi LODO RF pada eksekusi pertama menghasilkan skor yang jauh lebih rendah (maksimum 0.058087) dibanding eksekusi terverifikasi (maksimum 0.904169), dan tidak reprodusibel antar eksekusi. Eksekusi terverifikasi di docs/STEP13B_RF_CONFORMAL_VERIFIED.md lolos uji determinisme 3 kali dan seluruh guard pengecualian obat. Penyebab perbedaan pada eksekusi pertama tidak ditetapkan di repositori ini. Skor yang berlaku adalah hasil eksekusi terverifikasi.
l. **Incorrect Empty Set Fallback Rule**: An intermediate script version filled empty conformal sets with all three classes (`[0, 1, 2]`), forcing a constant HIGH action. Corrected to the formal **ABSTAIN** rule ($	ext{set\_size}=0$), which leaves metrics uncorrupted (`docs/STEP13C_RF_CONFORMAL_FIXED.md`).
m. **Inaccurate SANITY FLAG Claim on Bootstrap CIs**: A SANITY FLAG in `STEP14` claimed all 6 bootstrap CIs contained zero. Re-examination confirmed 5 of 6 contained zero, while the mean set size difference on Manual did not (`docs/STEP14_CONFORMAL_COMPARISON.md`).
n. Seksi H2 pada docs/PAPER_NUMBERS.md sempat memuat satu nama obat yang tidak ada dalam dataset (nicardipine), serta menyebut tiga obat latih (quinidine, mexiletine, diltiazem) sebagai bagian dari 16 obat uji. Jumlah obat 'RF benar, STACK salah' juga tertulis 4/4/2; nilai yang benar menurut docs/STEP12_STACK_VS_BASE_DIAGNOSTIC.md adalah 3/3/1 (disopyramide, loratadine, tamoxifen di chantest dan nanion; tamoxifen di manual).
o. **Discrepancy in Reported Base Learner CIs**: STACK minus base learner confidence intervals were temporarily reported in three different values across draft sections. Resolved by adopting VERBATIM values from `docs/STEP11_BASE_LEARNER_REFERENCE.md` in `docs/STEP17_CI_RECONCILIATION.md` & `docs/PAPER_NUMBERS.md`.

---

## 10. OPEN QUESTIONS

The following technical questions remain open and require further investigation:
1. **Undocumented Normalization Formula**: The exact mathematical transform used to normalize raw biophysical outputs into `_rev` features is not recorded in this repository.
2. **Unexplained Performance Delta vs. Earlier Datasets**: The exact cause of performance differences between earlier unrevised datasets and `_rev` datasets is unestablished.
3. **Control Normalization Scope**: Whether each drug was normalized against its own vehicle control run or against a shared control run is not documented here.
4. **`vrest` Split & SHAP Origin**: Why the Random Forest base learner splits on vrest 1,533 times while XGBoost splits on it zero times is unresolved, as is why KernelExplainer assigns it a stable rank-2 importance despite its minimal permutation effect.
5. **Formal Coverage Guarantee Class**: Whether the 12-refit jackknife calibration strictly satisfies split-conformal coverage guarantees or requires a cross-conformal theoretical framework is unestablished.
6. **Prospective Clinical Validation**: Because Rank 11 was selected using the 16 held-out test drugs, true prospective safety validation requires testing on a new, uninspected drug library.
7. **Re-evaluating Stacking Architecture Selection**: Whether the Stacking Classifier architecture should be reconsidered given that individual Random Forest base learners achieve higher point estimates across all three platforms while bootstrap difference CIs contain zero remains an open question. Undecided.

---

## 11. ARTIFACT INDEX

Below is the verified inventory of all core documentation, scripts, models, logs, and results in this repository:

### Documentation (`docs/`)
- `docs/STORYLINE.md`: Self-contained project narrative and comprehensive technical record.
- `docs/PAPER_NUMBERS.md`: Single canonical reference sheet for all validated project numbers.
- `docs/DATA_AUDIT_REV.md`: Primary dataset structure audit and column verification report.
- `docs/DATA_AUDIT_REV_PART2.md`: Forensic audit follow-up documenting the 26 Excel `#VALUE!` sentinel cells.
- `docs/LOADER_VERIFICATION_REV.md`: Verification report for data loader assertion guards and row counts.
- `docs/STEP3_STACKING_CHANTEST.md`: Chantest stacking model training and grouped CV verification report.
- `docs/STEP4_CROSS_PLATFORM.md`: Cross-platform generalization report on Chantest, Nanion, and Manual.
- `docs/STEP5_CONFORMAL.md`: Conformal layer formulation, LODO non-conformity scores, and initial calibration.
- `docs/STEP6_CONFORMAL_OPERATING_POINT.md`: Conformal rank sweep and operating-point under-triage evaluation report.
- `docs/STEP7_SHAP.md`: Global SHAP importance evaluation report.
- `docs/STEP8_SHAP_DIAGNOSTIC.md`: Diagnostic report investigating `vrest` split frequency, drug ranges, and permutations.
- `docs/STEP9_BOOTSTRAP.md`: 10,000-iteration drug-level bootstrap confidence interval calculation report.
- `docs/STEP10_VREST_OVERLAP_RECHECK.md`: Re-verification report on `vrest` feature range overlaps.
- `docs/STEP11_BASE_LEARNER_REFERENCE.md`: Baseline evaluation report for frozen Random Forest and XGBoost base estimators.
- `docs/STEP12_STACK_VS_BASE_DIAGNOSTIC.md`: Diagnostic report investigating meta-learner prediction overrides.
- `docs/STEP13_CONFORMAL_RF.md`: *[Marked Superseded]* Initial unverified RF conformal evaluation report.
- `docs/STEP13B_RF_CONFORMAL_VERIFIED.md`: *[Marked Superseded]* Intermediate verified RF conformal evaluation report.
- `docs/STEP13C_RF_CONFORMAL_FIXED.md`: Formal verified RF conformal prediction report with strict Abstain rules.
- `docs/STEP14_CONFORMAL_COMPARISON.md`: Paired bootstrap comparison report between RF+Conformal and STACK+Conformal.
- `docs/STEP17_CI_RECONCILIATION.md`: Audit and reconciliation report for base learner confidence intervals and directional tests.
- `docs/FIGURE_INVENTORY.md`: Inventory of all generated publication figures.
- `docs/MANIFEST.md`: Repository file manifest and directory specification.
- `docs/INDEX.md`: Directory structure and document inventory index.
- `docs/MODEL_BRIEFING.md`: *[Marked Stale]* Early technical briefing document superseded by STORYLINE.md.
- `docs/PROJECT_SUMMARY.md`: *[Marked Stale]* Early high-level summary superseded by STORYLINE.md.
- `docs/SCIENCE_CONTEXT.md`: *[Marked Stale]* Early scientific context overview superseded by STORYLINE.md.

### Results (`TrainAI/results/`)
- `TrainAI/results/cross_platform_probs_chantest_rev.csv`: Raw predicted probabilities and predictions for Chantest test drugs.
- `TrainAI/results/cross_platform_probs_nanion_rev.csv`: Raw predicted probabilities and predictions for Nanion test drugs.
- `TrainAI/results/cross_platform_probs_manual_rev.csv`: Raw predicted probabilities and predictions for Manual test drugs.
- `TrainAI/results/base_learner_reference_rev.csv`: Base learner reference metrics and paired bootstrap comparison values.
- `TrainAI/results/conformal_lodo_scores_rev.csv`: 12 LODO non-conformity calibration scores.
- `TrainAI/results/conformal_rf_rev.csv`: RF conformal prediction metrics across calibration ranks.
- `TrainAI/results/conformal_comparison_bootstrap_rev.csv`: 10,000-iteration paired bootstrap sampling distribution for RF vs STACK conformal comparison.
- `TrainAI/results/conformal_rank_sweep_rev.csv`: Full conformal rank sweep evaluations across ranks 12 to 8.
- `TrainAI/results/conformal_sets_rev.csv`: Prediction sets generated across calibration ranks.
- `TrainAI/results/bootstrap_distributions_rev.csv`: 10,000-iteration drug-level bootstrap sampling distributions.
- `TrainAI/results/shap_global_importance_rev.csv`: Global SHAP importance values for all 11 biomarkers.
- `TrainAI/results/fig1_block_diagram.pdf`: Vector PDF block diagram of the stacking and conformal architecture.
- `TrainAI/results/fig1_block_diagram.png`: PNG preview of Figure 1 block diagram.
- `TrainAI/results/fig_shap_global.pdf`: Vector PDF plot of global SHAP feature importance.
- `TrainAI/results/fig_shap_global.png`: PNG preview of SHAP global importance plot.

### Models (`TrainAI/models/`)
- `TrainAI/models/chantest_stacking_model_rev.joblib`: Frozen fitted StackingClassifier model artifact (SHA-256: `A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2`).
- `TrainAI/models/chantest_stacking_metadata_rev.json`: Feature order and tuned hyperparameter registry.

### Logs (`TrainAI/logs/`)
- `TrainAI/logs/DROPPED_ROWS_REV.csv`: Audit trail logging all 26 dropped corrupted `#VALUE!` data rows.

### Source Code (`TrainAI/src/`)
- `TrainAI/src/data_loader_rev.py`: Canonical data loader enforcing assertion guards and audit logging.
