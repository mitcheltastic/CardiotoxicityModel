import os

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
storyline_path = os.path.join(base_dir, "docs", "STORYLINE.md")

content = r"""# PROJECT STORYLINE & TECHNICAL NARRATIVE
**Repository Context**: Cardiotoxicity Risk Assessment via Ensemble Machine Learning and Conformal Prediction  
**Document Purpose**: A self-contained, plain-language record of technical development, validation, data integrity, and methodological audit.

---

## 1. THE PROBLEM

### Clinical Motivation: Torsade de Pointes
Torsade de Pointes (TdP) is a rare but potentially fatal ventricular arrhythmia (heart rhythm disorder). It is typically triggered by drugs that inadvertently block specific cardiac ion channels—most notably the $I_{\text{Kr}}$ channel encoded by the hERG gene—delaying cardiac repolarization and lengthening the QT interval on an electrocardiogram (ECG). Because TdP can lead to sudden cardiac death, cardiotoxicity screening is a mandatory regulatory hurdle in pharmaceutical drug development.

### In-Silico Screening and the CiPA Pipeline
Historically, drug cardiotoxicity was assessed using expensive, high-threshold animal studies or early human trials. To modernize this process, the international **Comprehensive *in vitro* Proarrhythmia Assay (CiPA)** initiative established an *in-silico* (computer-simulated) screening framework. 

Under the CiPA pipeline, experimental patch-clamp measurements of a drug's inhibition ($IC_{50}$ and Hill coefficient) across multiple cardiac ion channels ($I_{\text{Kr}}, I_{\text{Na}}, I_{\text{CaL}}, I_{\text{K1}}, I_{\text{To}}, I_{\text{Ks}}, I_{\text{NaL}}$) are fed into a biophysically detailed computer model of human ventricular cardiomyocytes (the O'Hara-Rudy cell model). The model simulates cardiac action potentials and intracellular calcium transients. From these dynamic simulations, 11 summary features—known as **biomarkers**—are extracted:
1. **qNet**: Net charge carried across the membrane during repolarization.
2. **APD50**: Action potential duration at 50% repolarization.
3. **APD90**: Action potential duration at 90% repolarization.
4. **Vrest**: Resting membrane potential.
5. **dvdtmax**: Maximum rate of depolarization.
6. **Vmax**: Peak action potential voltage.
7. **max_dv**: Maximum voltage depolarization gradient.
8. **camax**: Peak intracellular calcium concentration.
9. **CaTD50**: Calcium transient duration at 50% decay.
10. **CaTD90**: Calcium transient duration at 90% decay.
11. **carest**: Resting intracellular calcium concentration.

### 3-Tier Risk Labels and Clinical Safety Goals
The clinical objective is to classify drugs into one of three ordinal risk categories defined by CiPA expert consensus:
- **High Risk (Class 0 in repository)**: High propensity for TdP; dangerous cardiotoxin requiring strict restriction.
- **Intermediate Risk (Class 1 in repository)**: Moderate TdP risk; requires clinical monitoring.
- **Low Risk (Class 2 in repository)**: Low/negligible TdP risk; safe from a repolarization standpoint.

From a clinical standpoint, a "correct answer" requires not only high overall accuracy, but strictly avoiding **under-triage** (predicting a High-risk drug as Intermediate or Low risk). Under-triage is an unsafe failure mode that could allow a lethal toxin to reach human subjects unmonitored. Conversely, **over-call** (predicting a Low-risk drug as Intermediate or High risk) is a conservative failure mode that may unnecessarily delay a safe drug, but preserves patient safety.

---

## 2. THE DATA

### Drug Benchmarks, Splits, and Simulations
The project evaluates machine learning models on a benchmark library of **28 drugs** standard in CiPA validation studies:
- **12 Training Drugs**: *bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, sotalol, terfenadine, verapamil*. (Used for base/meta-learner training and conformal calibration).
- **16 Held-Out Test Drugs**: *astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib*. (Used strictly for cross-platform validation and operating-point evaluation).

For each drug, stochastic cell simulations generate **2,000 simulated cells** to capture physiological variability in ion channel conductance across human populations. Thus, each drug is represented by 2,000 rows of 11 biomarker measurements.

### Electrophysiology Platforms
To test whether models generalize across experimental measurement techniques, data originates from three distinct automated patch-clamp electrophysiology platforms:
1. **Chantest**: High-throughput automated patch-clamp platform (Primary training/calibration baseline).
2. **Nanion**: Secondary automated patch-clamp platform.
3. **Manual**: Manual glass-electrode patch-clamp (gold-standard electrophysiology).

### Pre-Normalized `_rev` Datasets
All experiments in this repository utilize the revised (`_rev`) dataset files. These datasets are pre-normalized against vehicle control simulations. Consequently, **no additional feature scaling** (such as `StandardScaler` or `MinMaxScaler`) was applied in this project to prevent artificial distortion of control-anchored physiological units.  
*Note*: The exact mathematical formula used upstream to generate the `_rev` normalized features from raw biophysical outputs is not documented within this repository.

---

## 3. DATA INTEGRITY WORK

### Data Audit & Sentinel Cell Discovery
Before model training, a rigorous data audit checked array shapes, data types, missing values, and sentinel strings across all three platform files. 

The audit identified **exactly 26 corrupted rows** containing the Excel error string `'#VALUE!'`:
- **25 rows** in `manual_training_scaled_rev.csv`: All belonging to *mexiletine* simulations, spanning the columns `APD50`, `APD90`, and `max_dv`.
- **1 row** in `manual_testing_scaled_rev.csv`: Belonging to an *ibutilide* simulation, in column `CaTD90`.
- **0 sentinel rows** were found in any Chantest or Nanion files.

### Drop-Not-Impute Decision
A strict decision was made to **drop rather than impute** these 26 rows. Imputing values in biophysically simulated cell outputs could introduce artificial correlations or unphysical state variables. Because 25 rows represent a tiny fraction (1.25%) of mexiletine's 2,000 rows and 1 row represents 0.05% of ibutilide's rows, dropping them preserves absolute data purity without impairing statistical power.

### Hardened Loader & Audit Trail
The canonical data loader (`src/data_loader_rev.py`) was hardened with strict assertion guards. It automatically detects sentinel strings, verifies feature order, drops invalid rows, and logs every dropped row to the audit trail file `logs/DROPPED_ROWS_REV.csv`.

---

## 4. THE MODEL

### Model Architecture: Stacking Classifier
To capture complex non-linear interactions between biomarkers, an ensemble **Stacking Classifier** (`StackingClassifier`) architecture was implemented:
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
Standard machine learning models output a single hard prediction (`argmax`), which forces a forced choice even when the model is highly uncertain. **Split-Conformal Prediction** replaces hard labels with **prediction sets** $\mathcal{C}(X) \subseteq \{\text{High}, \text{Intermediate}, \text{Low}\}$. A prediction set contains all classes whose predicted probability clears an inclusion threshold $1 - \hat{q}$. If a drug's classification is ambiguous, the set expands to include multiple classes (e.g., `['High', 'Intermediate']`).

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

---

## 7. VALIDITY CHECKS

### SHAP Analysis & The $V_{\text{rest}}$ Anomaly
SHAP (SHapley Additive exPlanations) was computed using `KernelExplainer` on the frozen stacking model (`TrainAI/results/shap_global_importance_rev.csv`). The global feature importance ranking placed **qNet** first ($\text{mean}|\text{SHAP}| = 0.145534$) and **Vrest** second ($\text{mean}|\text{SHAP}| = 0.115489$).

However, a detailed split audit revealed a structural anomaly:
- In Random Forest base learners, $V_{\text{rest}}$ accounted for 1,533 tree splits at hyper-precise 8-decimal thresholds (e.g., $V_{\text{rest}} \le -84.12345678$).
- In XGBoost, $V_{\text{rest}}$ was used **zero times** across all 300 trees.

### Permutation Diagnostic Results
Per `docs/STEP8_SHAP_DIAGNOSTIC.md`, permuting $V_{\text{rest}}$ values across test drugs changed an average of only **1.0 out of 16 drug-level predictions** (and zero predictions on many random seeds), causing a minor macro-F1 drop of 0.073. By contrast, permuting $qNet$ changed **3.0 drug predictions** and dropped macro-F1 by 0.108.

### Conclusion & Unverified Findings
$V_{\text{rest}}$ provides near-perfect numeric separability in synthetic cell simulations but is **structurally redundant** when $qNet$ is present. Whether $V_{\text{rest}}$'s high SHAP ranking represents a true physiological mechanism or an artifact of synthetic ODE solver boundary conditions remains unverified.

---

## 8. ERRORS CAUGHT DURING THE WORK

As part of rigorous quality assurance, several intermediate errors were identified and corrected during development:

1. **Miscounted Sentinel Rows**: Early notes reported 28 dropped rows. A complete audit of `logs/DROPPED_ROWS_REV.csv` confirmed exactly **26 dropped rows** (25 in manual training, 1 in manual testing).
2. **Miscomputed Row Counts**: Initial documentation listed total dataset row counts without split breakdowns. Exact counts were corrected to: Chantest (24,000 train / 32,000 test), Nanion (24,000 train / 32,000 test), and Manual (23,975 train / 31,999 test).
3. **Degenerate Leave-One-Out Fragility Analysis**: An early sensitivity analysis held $\alpha = 1/13$ fixed while dropping calibration drugs down to $n=11$. Because $\lceil 12(1-1/13) \rceil = 12 > 11$, $\hat{q}$ defaulted to 1.0, forcing inclusion thresholds to 0.0 and assigning all drugs High risk regardless of model probabilities. This was retracted and replaced with the full rank sweep.
4. **Premature Operating Point Selection**: Early documentation claimed Rank 12 ($\hat{q}=0.986695$) was the tightest safe operating point. Scanning all ranks proved Rank 11 ($\hat{q}=0.762830$) was the true tightest point achieving zero under-triage.
5. **Degenerate Under-Triage Confidence Interval**: Early bootstrap reports listed a 95% CI of `[0.000000, 0.000000]` for under-triage. Because 0 under-triages occurred in the 16 test drugs, every bootstrap sample contained 0 under-triages. This point interval was superseded by the **Rule of Three upper bound** ($3/16 = 0.1875$ or 18.75%).
6. **Coverage vs. Correct-Outcome Confusion**: Initial drafts conflated correct-outcome proportion (11/16 = 0.6875) with empirical coverage (12/16 = 0.7500 on Chantest/Nanion; 14/16 = 0.8750 on Manual). This was corrected in Section D of `docs/PAPER_NUMBERS.md`.

---

## 9. OPEN QUESTIONS

The following technical questions remain open and require further investigation:
1. **Undocumented Normalization Formula**: The exact mathematical transform used to normalize raw biophysical outputs into `_rev` features is not recorded in the repository.
2. **Unexplained Performance Delta vs. Earlier Datasets**: Performance variations between earlier unrevised datasets and `_rev` datasets remain unverified.
3. **$V_{\text{rest}}$ Micro-Threshold Origin**: The exact mechanism causing Random Forest to create 1,533 micro-splits on $V_{\text{rest}}$ while XGBoost ignores it is unresolved.
4. **Prospective Clinical Validation**: Because Rank 11 was selected using the 16 held-out test drugs, true prospective safety validation requires testing on a new, uninspected drug library.

---

## 10. ARTIFACT INDEX

Below is the index of all core artifacts, models, scripts, and documentation generated during this project:

- `src/data_loader_rev.py`: Canonical data loader enforcing assertion guards and audit logging.
- `models/chantest_stacking_model_rev.joblib`: Frozen fitted StackingClassifier model artifact (SHA-256: `A93B9868...`).
- `models/chantest_stacking_metadata_rev.json`: Canonical feature order and tuned hyperparameter registry.
- `results/cross_platform_probs_chantest_rev.csv`: Raw predicted probabilities and predictions for Chantest.
- `results/cross_platform_probs_nanion_rev.csv`: Raw predicted probabilities and predictions for Nanion.
- `results/cross_platform_probs_manual_rev.csv`: Raw predicted probabilities and predictions for Manual.
- `results/conformal_lodo_scores_rev.csv`: 12 LODO non-conformity calibration scores.
- `results/conformal_rank_sweep_rev.csv`: Full conformal rank sweep evaluations across ranks 12 to 8.
- `results/bootstrap_distributions_rev.csv`: 10,000-iteration drug-level bootstrap sampling distributions.
- `results/shap_global_importance_rev.csv`: Global SHAP importance values for all 11 biomarkers.
- `results/fig1_block_diagram.pdf`: Vector PDF block diagram of the two-level stacking and conformal architecture.
- `results/fig_shap_global.pdf`: Vector PDF summary plot of global SHAP feature importance.
- `logs/DROPPED_ROWS_REV.csv`: Audit trail logging all 26 dropped corrupted data rows.
- `docs/PAPER_NUMBERS.md`: Single canonical reference sheet for all validated project numbers.
- `docs/DATA_AUDIT_REV_PART2.md`: Comprehensive data audit report detailing missing value investigations.
- `docs/LOADER_VERIFICATION_REV.md`: Data loader validation and dataset row-count verification.
- `docs/STEP3_STACKING_CHANTEST.md`: Chantest model training and grouped CV verification report.
- `docs/STEP4_CROSS_PLATFORM.md`: Cross-platform evaluation report across Chantest, Nanion, and Manual.
- `docs/STEP5_CONFORMAL.md`: Initial split-conformal layer formulation and LODO score calculations.
- `docs/STEP6_CONFORMAL_OPERATING_POINT.md`: Full operating-point sweep and under-triage evaluation report.
- `docs/STEP8_SHAP_DIAGNOSTIC.md`: SHAP diagnostic report investigating the $V_{\text{rest}}$ importance anomaly.
- `docs/STEP9_BOOTSTRAP.md`: Drug-level bootstrap confidence interval calculation report.
- `docs/STORYLINE.md`: Self-contained project narrative and comprehensive technical record.
"""

with open(storyline_path, "w", encoding="utf-8") as f:
    f.write(content.strip())

print(content.strip())
