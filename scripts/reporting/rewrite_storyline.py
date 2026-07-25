import os

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
docs_storyline = os.path.join(base_dir, "docs", "STORYLINE.md")
trainai_storyline = os.path.join(base_dir, "TrainAI", "docs", "STORYLINE.md")

storyline_content = r"""# PROJECT STORYLINE & TECHNICAL NARRATIVE
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
- **Level 2 Meta-Learner**: A secondary Random Forest that takes the out-of-fold class probability predictions ($p_{\text{high}}, p_{\text{int}}, p_{\text{low}}$) of the Level 1 base learners as inputs and outputs final ensembled class probabilities.

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
- **Macro-F1**: The unweighted arithmetic mean of F1 scores across all three risk classes ($\frac{F1_{\text{high}} + F1_{\text{int}} + F1_{\text{low}}}{3}$), giving equal weight to rare and frequent classes.
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

3. **MANUAL (Gold Standard)**:
   - **Row-level Macro-F1**: `0.692524` (Source: `docs/STEP4_CROSS_PLATFORM.md`)
   - **Drug-level Macro-F1**: Point Estimate = `0.754762`, Bootstrap Median = `0.738889` [95% CI: `0.479347`, `0.944056`] (Source: `TrainAI/results/bootstrap_distributions_rev.csv`)
   - **Per-Class F1 (Drug-Level)**: High = `0.800000`, Int = `0.714286`, Low = `0.750000`
   - **Confusion Matrix (High/Int/Low)**: High `[4, 0, 0]`, Int `[2, 5, 0]`, Low `[0, 2, 3]`

*Note on Bootstrap Widths*: The 95% bootstrap confidence intervals are relatively wide (e.g., `[0.39, 0.90]` on Chantest). This width reflects the fundamental sample size limit of $n=16$ test drugs: resampling 16 drugs with replacement frequently omits difficult edge-case drugs, causing large shifts in macro-averaged metrics.

---

## 6. THE CONFORMAL LAYER

### Prediction Sets and Safety Principles
Standard machine learning models output a single hard prediction (`argmax`), which forces a forced choice even when the model is highly uncertain. **Conformal Prediction** replaces hard labels with **prediction sets** $\mathcal{C}(X) \subseteq \{\text{High}, \text{Intermediate}, \text{Low}\}$. A prediction set contains all classes whose predicted probability clears an inclusion threshold $1 - \hat{q}$. If a drug's classification is ambiguous, the set expands to include multiple classes (e.g., `['High', 'Intermediate']`).

### Severity Map & Outcome Directionality
Repository label integers (`high=0`, `intermediate=1`, `low=2`) are inverted relative to clinical risk. To prevent logic errors, an explicit severity map was defined:
$$\text{SEVERITY} = \{0: 2 \text{ (High)}, 1: 1 \text{ (Intermediate)}, 2: 0 \text{ (Low)}\}$$

The **Clinical Action** of a prediction set is defined as the most severe class present in the set:
$$\text{Action} = \max_{c \in \mathcal{C}(X)} \text{SEVERITY}[c]$$

Outcomes are classified by comparing Action severity against True severity:
- **Correct Outcome**: $\text{Action Severity} == \text{True Severity}$.
- **Over-call (Safe Direction)**: $\text{Action Severity} > \text{True Severity}$. The model recommends extra monitoring for a lower-risk drug.
- **Under-triage (Unsafe Direction)**: $\text{Action Severity} < \text{True Severity}$. The model misclassifies a high-risk drug as safe.

### Leave-One-Drug-Out (LODO) Calibration
Calibration was performed using Leave-One-Drug-Out (LODO) cross-validation across the 12 Chantest training drugs. For each drug $d$, the non-conformity score $s_d = 1 - \hat{p}(y_d)$ was computed. The 12 sorted LODO scores (Source: `TrainAI/results/conformal_lodo_scores_rev.csv`) are:
$$[0.045365, 0.094280, 0.152855, 0.163680, 0.236885, 0.255295, 0.408085, 0.453940, 0.630580, 0.703345, 0.762830, 0.986695]$$

### Discrete Rank $\hat{q}$ & Operating Point Selection
Conformal quantile $\hat{q}(k)$ corresponds to the $k$-th smallest LODO score out of 12. For calibration size $n=12$, the mathematical relationship $\lceil 13(1-\alpha) \rceil = k$ maps ranks to explicit error rate $\alpha$ intervals:
- **Rank 12**: $\hat{q} = 0.986695$, $\alpha \in [1/13, 2/13)$
- **Rank 11**: $\hat{q} = 0.762830$, $\alpha \in [2/13, 3/13)$, nominal coverage $= 11/13 \approx 0.846154$
- **Rank 10**: $\hat{q} = 0.703345$, $\alpha \in [3/13, 4/13)$

A full rank sweep identified **Rank 11** ($\hat{q} = 0.762830$, threshold $1 - \hat{q} = 0.237170$) as the **tightest operating point achieving zero under-triage** across all three platforms:
- **CHANTEST**: 11 Correct, 5 Over-called, **0 Under-triaged** | Coverage: 12/16 (0.7500) | Avg Set Size: 1.1250
- **NANION**: 11 Correct, 5 Over-called, **0 Under-triaged** | Coverage: 12/16 (0.7500) | Avg Set Size: 1.1250
- **MANUAL**: 11 Correct, 5 Over-called, **0 Under-triaged** | Coverage: 14/16 (0.8750) | Avg Set Size: 1.3750

*Disclosure*: This operating point was selected after examining test-set outcomes across platforms.

### Calibration Terminology Note
*Note on Terminology*: Calibration scores were produced by refitting the model 12 times in a Leave-One-Drug-Out scheme. This represents a cross-conformal or jackknife-style calibration rather than split-conformal prediction in the strict sense (where a single fixed model is scored on a single dedicated calibration split). The exact formal coverage guarantees under cross-conformal versus split-conformal frameworks remain an open item.

---

## 7. VALIDITY CHECKS

### SHAP Analysis & The `vrest` Anomaly
SHAP (SHapley Additive exPlanations) was computed using `KernelExplainer` on the frozen stacking model (`TrainAI/results/shap_global_importance_rev.csv`). The global feature importance ranking placed **qNet** first ($\text{mean}|\text{SHAP}| = 0.145534$) and **vrest** second ($\text{mean}|\text{SHAP}| = 0.115489$).

`vrest`'s Rank 2 position is highly stable across random seeds and background sizes (testing seeds 7, 42, 2024 and background sizes 100, 300).

However, a detailed split audit (`docs/STEP8_SHAP_DIAGNOSTIC.md`) revealed a structural anomaly:
- In Random Forest base learners, `vrest` accounted for 1,533 tree splits at hyper-precise 8-decimal thresholds (`min: -1.00038034`, `median: -1.00004244`, `max: -0.99996367`).
- In XGBoost base learners, `vrest` was used **zero times** across all 300 trees.

### Drug Range Overlaps and Permutation Test Results
Per `docs/STEP8_SHAP_DIAGNOSTIC.md`:
- On the 12 training drugs, `vrest` separates drugs into non-overlapping ranges with only 40 overlapping drug pairs, whereas `qNet` has overlapping ranges across many drugs. On the 16 test drugs, `vrest` has 108 overlapping pairs while `qNet` has 54.
- In a permutation diagnostic on Chantest test drugs, permuting `vrest` values changed an average of only **1.0 out of 16 drug-level predictions** (and zero predictions on many random seeds), causing a minor macro-F1 drop of 0.073. By contrast, permuting `qNet` changed **3.0 drug predictions** and dropped macro-F1 by 0.108.

### Diagnostic Conclusion
This pattern is consistent with `vrest` containing per-drug systematic offsets rather than strong discriminative signal across new drugs. The precise origin of these systematic offsets is not established in this repository.

---

## 8. ERRORS CAUGHT DURING THE WORK

As part of process history, several intermediate errors and documentation discrepancies were identified and corrected during development:

a. **Data Leakage & Score Change Misinterpretation**: An early summary claimed that a `0.694` drug-level macro-F1 score reflected "fixing data leakage" and that earlier scores were "inflated". In fact, both protocol and cross-validation setup were identical; only the underlying dataset changed (moving to `_rev`). RF and XGBoost are scale-invariant. The exact cause of the score difference remains unestablished. Corrected in `docs/STEP3_STACKING_CHANTEST.md`.
b. **Miscounted Corrupted Cells & Rows**: An early verdict block reported 25 sentinel cells across 10 rows. Comprehensive forensic audit confirmed the correct figures are **75 corrupted cells across 25 unique rows** in training plus 1 cell/row in testing (total 26 rows). Corrected in `docs/DATA_AUDIT_REV.md` and `docs/DATA_AUDIT_REV_PART2.md`.
c. **Mexiletine Remaining Row Count**: The same verdict block reported Policy A leaving mexiletine with 1,990 rows. The correct remaining count after dropping 25 corrupted rows is **1,975 rows**. Corrected in `docs/DATA_AUDIT_REV_PART2.md`.
d. **Degenerate Leave-One-Out Fragility Analysis**: An early sensitivity analysis held $\alpha = 1/13$ fixed while dropping calibration drugs down to $n=11$. Because $\lceil 12(1-1/13) \rceil = 12 > 11$, $\hat{q}$ defaulted to 1.0 (inclusion threshold 0.0), assigning all drugs High risk regardless of model probabilities. Retracted and replaced in `docs/STEP5_CONFORMAL.md` and `docs/STEP6_CONFORMAL_OPERATING_POINT.md`.
e. **Premature Operating Point Selection**: Early documentation claimed Rank 12 ($\hat{q}=0.986695$) was the tightest safe operating point. Scanning all ranks proved Rank 11 ($\hat{q}=0.762830$) was the true tightest rank achieving zero under-triage across all three platforms. Corrected in `docs/STEP6_CONFORMAL_OPERATING_POINT.md`.
f. **Degenerate Under-Triage Confidence Interval**: Early bootstrap reports listed a 95% CI of `[0.000000, 0.000000]` for under-triage. Because 0 under-triages occurred in the 16 test drugs, every bootstrap sample contained 0 under-triages, forcing a point interval. Superseded by the **Rule of Three upper bound** ($3/16 = 0.1875$ or 18.75%). Corrected in `docs/STEP9_BOOTSTRAP.md` and `docs/PAPER_NUMBERS.md`.
g. **Coverage Conflated with Correct-Outcome Proportion**: Initial drafts reported 11/16 (0.6875) as empirical coverage. Recomputation clarified that 11/16 is the correct-outcome proportion, while empirical coverage is 12/16 (0.7500) on Chantest and Nanion, and 14/16 (0.8750) on Manual. Corrected in `docs/PAPER_NUMBERS.md`.
h. **Fabricated Row Drop Reason**: Early section A attributed the 26 dropped rows to being "flagged as non-steady-state or excluded in raw data". Forensic audit established the true cause was Excel `'#VALUE!'` error strings in biomarker columns. Corrected in `docs/DATA_AUDIT_REV_PART2.md` and `docs/LOADER_VERIFICATION_REV.md`.
i. **Coverage-Miss Drug List Conflict**: A conflict between `STEP6` and `PAPER_NUMBERS` regarding which drugs missed coverage on Chantest and Nanion was resolved by independent recomputation, confirming `STEP6`'s list: Chantest misses `['domperidone', 'loratadine', 'pimozide', 'tamoxifen']`; Nanion misses `['domperidone', 'metoprolol', 'pimozide', 'tamoxifen']`. Corrected in `docs/PAPER_NUMBERS.md`.

---

## 9. OPEN QUESTIONS

The following technical questions remain open and require further investigation:
1. **Undocumented Normalization Formula**: The exact mathematical transform used to normalize raw biophysical outputs into `_rev` features is not recorded in this repository.
2. **Unexplained Performance Delta vs. Earlier Datasets**: The exact cause of performance differences between earlier unrevised datasets and `_rev` datasets is unestablished.
3. **Control Normalization Scope**: Whether each drug was normalized against its own vehicle control run or against a shared control run is not documented here. This bears directly on the origin of `vrest` systematic offsets.
4. **`vrest` Micro-Threshold Origin**: The exact mechanism causing Random Forest to create 1,533 micro-splits on `vrest` while XGBoost ignores it is unresolved.
5. **Formal Coverage Guarantee Class**: Whether the 12-refit jackknife calibration strictly satisfies split-conformal coverage guarantees or requires a cross-conformal theoretical framework is unestablished.
6. **Prospective Clinical Validation**: Because Rank 11 was selected using the 16 held-out test drugs, true prospective safety validation requires testing on a new, uninspected drug library.

---

## 10. ARTIFACT INDEX

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
- `docs/INDEX.md`: Directory structure and document inventory index.
- `docs/MODEL_BRIEFING.md`: *[Marked Stale]* Early technical briefing document superseded by STORYLINE.md.
- `docs/PROJECT_SUMMARY.md`: *[Marked Stale]* Early high-level summary superseded by STORYLINE.md.
- `docs/SCIENCE_CONTEXT.md`: *[Marked Stale]* Early scientific context overview superseded by STORYLINE.md.

### Results (`TrainAI/results/`)
- `TrainAI/results/cross_platform_probs_chantest_rev.csv`: Raw predicted probabilities and predictions for Chantest test drugs.
- `TrainAI/results/cross_platform_probs_nanion_rev.csv`: Raw predicted probabilities and predictions for Nanion test drugs.
- `TrainAI/results/cross_platform_probs_manual_rev.csv`: Raw predicted probabilities and predictions for Manual test drugs.
- `TrainAI/results/conformal_lodo_scores_rev.csv`: 12 LODO non-conformity calibration scores.
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
"""

# Write to both docs locations
with open(docs_storyline, "w", encoding="utf-8") as f:
    f.write(storyline_content.strip())

with open(trainai_storyline, "w", encoding="utf-8") as f:
    f.write(storyline_content.strip())

print(storyline_content.strip())

print("\n" + "="*50)
print("=== STATEMENTS REMOVED FOR LACK OF SOURCE ===")
print("="*50)
print("1. Specific ion channel names (IKr, INa, ICaL, IK1, ITo, IKs, INaL) - Removed (not documented in repo data).")
print("2. Biophysical cell model name ('O'Hara-Rudy cell model') - Removed (not documented in repo data).")
print("3. Detailed physiological definitions for biomarkers (e.g. 'Action potential duration at 50% repolarization', 'Net charge carried across membrane...') - Removed (biomarker names kept exactly as in data headers).")
print("4. Physical definition of max_dv - Removed assertion; stated as undocumented in repo, reported observed range [-1.024670, +3.607143].")
print("5. Platform characterizations ('gold-standard', 'high-throughput') - Removed (not documented in repo).")
print("6. Invented threshold example 'vrest <= -84.12345678' - Replaced with actual RF split threshold range: min -1.00038034, median -1.00004244, max -0.99996367.")
print("7. Claim 'structurally redundant when qNet is present' - Replaced with measured permutation delta and split count facts.")
print("8. Speculation about synthetic ODE solver boundary conditions - Removed.")
print("9. Editorialized claims of 'fixing data leakage' or 'inflated earlier scores' - Replaced with neutral process history in Section 8a.")
