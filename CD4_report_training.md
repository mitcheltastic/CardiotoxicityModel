# Implementation Report (CD-4): In-Silico Cardiotoxicity (TdP) Risk Prediction Pipeline & Model Serving

**Project Identity**: In-Silico Cardiotoxicity (Torsade de Pointes / TdP) Risk Assessment via Machine Learning and Conformal Prediction  
**Evaluation Benchmark**: Comprehensive CiPA In-Silico Multi-Platform Electrophysiology Dataset  
**Serving Layer**: Hugging Face Space (`mitcheltastic/tdp-cipa-screening`) with Conformal Safety Guarantees and SHAP Attribution  

---

## 0. Repository overview

### 0.1 Directory Structure (Top 2–3 Levels)

The repository layout and the designated function of each primary directory are structured as follows:

```
TrainAI/
├── .env                                  # Environment variables (contains REDACTED secrets)
├── .gitignore                            # Git exclusion rules
├── README.md                             # High-level project summary and quickstart guide
├── deploy_utils.py                       # Estimator wrappers required for deserializing models
├── conformal_safety.py                   # Conformal prediction utility functions
├── conformal_diag.py                     # Diagnostic script for conformal evaluation
├── test_conformal.py                     # Unit testing script for conformal prediction rules
├── test_local.py                         # Local testing script for prediction and explanation
├── plus_ultra.ipynb                      # Primary research notebook (Phases 0 through 5c)
├── data/                                 # Simulation dataset files across three assay platforms
│   ├── chantest_training_scaled_rev.csv  # Chantest training set (12 drugs, 24,000 rows)
│   ├── chantest_testing_scaled_rev.csv   # Chantest held-out test set (16 drugs, 32,000 rows)
│   ├── nanion_training_scaled_rev.csv    # Nanion training set (12 drugs, 24,000 rows)
│   ├── nanion_testing_scaled_rev.csv     # Nanion held-out test set (16 drugs, 32,000 rows)
│   ├── manual_training_scaled_rev.csv    # Manual training set (12 drugs, 23,975 active rows)
│   ├── manual_testing_scaled_rev.csv     # Manual held-out test set (16 drugs, 31,999 active rows)
│   └── [pre-_rev legacy CSV files]       # Historical unrevised simulation CSV files
├── src/                                  # Modular source utilities
│   └── data_loader_rev.py                # Hardened data loader enforcing sentinel checks
├── models/                               # Research model checkpoints and metadata
│   ├── chantest_stacking_model_rev.joblib # Frozen trained StackingClassifier research artifact
│   ├── chantest_stacking_metadata_rev.json# Serialized hyperparameters and CV evaluation results
│   ├── logistic_multivariate_rev.joblib  # Baseline 11-biomarker Logistic Regression model
│   └── logistic_qnet_rev.joblib          # Baseline single-feature qNet Logistic Regression model
├── results/                              # Official benchmark evaluation tables and vector figures
│   ├── bootstrap_distributions_rev.csv   # 10,000-iteration bootstrap distributions
│   ├── cross_platform_probs_*_rev.csv    # Per-drug probability outputs across platforms
│   ├── conformal_lodo_scores_rev.csv     # 12 LODO non-conformity calibration scores
│   ├── conformal_rank_sweep_rev.csv      # Conformal operating point sweep (Ranks 12–8)
│   ├── shap_global_importance_rev.csv    # Global SHAP feature attribution table
│   ├── base_learner_reference_rev.csv    # Base learner vs. stacking comparative metrics
│   ├── ordinal_binary_auc_rev.csv        # Ordinal binary threshold AUC metrics
│   ├── fig1_block_diagram.pdf / .png     # Publication architecture diagram
│   └── fig_shap_global.pdf / .png        # Vector figure of global SHAP feature importance
├── paper_figures/                        # Standalone figure generation scripts and outputs
│   ├── fig1.py / fig1.pdf                # Figure 1 generation script and PDF
│   └── fig_architecture.py / .pdf       # Detailed pipeline architecture diagram
├── docs/                                 # Step-by-step engineering audits and canonical metrics
│   ├── PAPER_NUMBERS.md                  # Single source of truth for canonical benchmark metrics
│   ├── STORYLINE.md                      # Comprehensive technical narrative and audit log
│   ├── LOADER_VERIFICATION_REV.md        # Data loader validation report
│   ├── DATA_AUDIT_REV.md / PART2.md      # Data integrity and sentinel value audit reports
│   ├── STEP3_STACKING_CHANTEST.md        # Stacking model training report on Chantest
│   ├── STEP4_CROSS_PLATFORM.md           # Cross-platform evaluation report (Nanion, Manual)
│   ├── STEP5_CONFORMAL.md                # Conformal layer formulation and LODO calibration
│   ├── STEP6_CONFORMAL_OPERATING_POINT.md# Conformal operating point selection (Rank 11)
│   ├── STEP7_SHAP.md                     # SHAP global explainability report
│   ├── STEP8_SHAP_DIAGNOSTIC.md          # Diagnostic audit of vrest importance anomaly
│   ├── STEP9_BOOTSTRAP.md                # 10,000-iteration drug-level bootstrap report
│   ├── STEP11_BASE_LEARNER_REFERENCE.md  # Reference evaluation of base learners
│   ├── STEP12_STACK_VS_BASE_DIAGNOSTIC.md# Meta-learner override pathology analysis
│   ├── STEP13C_RF_CONFORMAL_FIXED.md     # Conformal Random Forest validation report
│   ├── STEP14_CONFORMAL_COMPARISON.md    # Paired bootstrap comparison (RF vs. STACK conformal)
│   ├── STEP18A_ORDINAL_BINARY_AUC.md     # Ordinal binary AUC evaluation report
│   ├── STEP19_LOGISTIC_BASELINE.md       # Logistic regression baseline evaluation report
│   ├── STEP20_UNIFIED_TABLE.md           # Unified cross-model performance comparison
│   ├── STEP22_CIPA_BOOTSTRAP.md          # CiPA row-level bootstrap evaluation report
│   └── guide for front-end.md            # API contract documentation for frontend clients
├── hf_space/                             # Hugging Face Space production deployment package
│   ├── app.py                            # Gradio web application and REST API entrypoint
│   ├── deployable_chantest_3class.joblib # Deployed model bundle (Stacking with Logistic Regression meta)
│   ├── shap_background.joblib            # 15 k-means background centroids for KernelExplainer
│   ├── deploy_utils.py                   # Required unpickling wrapper classes
│   └── requirements.txt                  # Python dependencies for Hugging Face Space
├── scripts/                              # Automated pipeline and reporting scripts
│   ├── pipeline/                         # Training, evaluation, bootstrap, and SHAP scripts
│   ├── reporting/                        # Verification, audit reconciliation, and figure scripts
│   └── legacy/                           # Archived prototype scripts
├── logs/                                 # Runtime logs and audit trails
│   └── DROPPED_ROWS_REV.csv              # Audit record of all 26 dropped corrupted data rows
└── artifacts/                            # Obsolete pre-_rev artifacts (quarantined; do not cite)
```

### 0.2 Code Partitioning: Training vs. Deployed HF Space

| Component | Files / Directories | Execution Scope & Primary Purpose |
| :--- | :--- | :--- |
| **Training Pipeline** | `src/data_loader_rev.py`, `scripts/pipeline/*.py`, `plus_ultra.ipynb`, `models/` | Ingests raw simulation CSVs, performs data cleaning and grouped CV tuning, fits base learners and meta-learners, computes LODO calibration scores, runs bootstrap evaluations, and calculates global SHAP importance. |
| **Deployed HF Space** | `hf_space/app.py`, `hf_space/deployable_chantest_3class.joblib`, `hf_space/shap_background.joblib`, `hf_space/deploy_utils.py`, `hf_space/requirements.txt` | Standalone Gradio application serving prediction and on-demand local SHAP explanations; deployed to Hugging Face Spaces (`mitcheltastic/tdp-cipa-screening`). Loaded at container startup. |

---

## 1. Dataset

### 1.1 Dataset Files & Format

The benchmark evaluation is conducted on three electrophysiology assay platforms (`chantest`, `nanion`, `manual`). All canonical experiments utilize the revised (`_rev`) datasets generated on 2026-07-25 (`docs/PAPER_NUMBERS.md` L4–10, `docs/LOADER_VERIFICATION_REV.md` L22–171).

| Dataset File Path | Format | Total Rows | Columns | Cleaned Active Rows | Offending Dropped Rows |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `data/chantest_training_scaled_rev.csv` | CSV | 24,000 | 14 | 24,000 | 0 |
| `data/chantest_testing_scaled_rev.csv` | CSV | 32,000 | 14 | 32,000 | 0 |
| `data/nanion_training_scaled_rev.csv` | CSV | 24,000 | 15 | 24,000 | 0 |
| `data/nanion_testing_scaled_rev.csv` | CSV | 32,000 | 15 | 32,000 | 0 |
| `data/manual_training_scaled_rev.csv` | CSV | 24,000 | 14 | 23,975 | 25 (mexiletine) |
| `data/manual_testing_scaled_rev.csv` | CSV | 32,000 | 14 | 31,999 | 1 (ibutilide) |

*(Note: Pre-_rev historical files `data/*_training.csv` and `data/*_testing.csv` contain 24,001 and 32,001 lines respectively including headers, but are quarantined as obsolete).*

### 1.2 Drug Library & Clinical Risk Categories

The dataset consists of **28 reference compounds** from the Comprehensive In Vitro Proarrhythmia Assay (CiPA) initiative, partitioned into a 12-drug training split and a 16-drug held-out testing split (`docs/PAPER_NUMBERS.md` L9–10, `src/data_loader_rev.py` L130).

#### Training Drugs (12 Compounds)
- **High Risk (4)**: `bepridil`, `dofetilide`, `quinidine`, `sotalol`
- **Intermediate Risk (4)**: `chlorpromazine`, `cisapride`, `ondansetron`, `terfenadine`
- **Low Risk (4)**: `diltiazem`, `mexiletine`, `ranolazine`, `verapamil`

#### Testing Drugs (16 Held-Out Compounds)
- **High Risk (4)**: `azimilide`, `disopyramide`, `ibutilide`, `vandetanib`
- **Intermediate Risk (7)**: `astemizole`, `clarithromycin`, `clozapine`, `domperidone`, `droperidol`, `pimozide`, `risperidone`
- **Low Risk (5)**: `loratadine`, `metoprolol`, `nifedipine`, `nitrendipine`, `tamoxifen`

### 1.3 Train / Test Split Protocol

The train/test partition is strictly performed **by drug** (whole-drug holdout), following the benchmark partitioning of Fuadah et al. (2024) and Li et al. (2019). It is not a randomized row-level split (`docs/STORYLINE.md` L38–43).

#### Code Verifying the Split
The disjoint partitioning is asserted programmatically in `src/data_loader_rev.py` (L87–94) and `scripts/pipeline/run_step3_stacking.py` (L87–94):

```python
# Citation: scripts/pipeline/run_step3_stacking.py (Lines 87-94)
train_drugs = set(groups_train.unique())
test_drugs  = set(groups_test.unique())
overlap = train_drugs.intersection(test_drugs)

log(f"\nASSERT: train_drugs INTERSECT test_drugs is empty: {'PASS' if len(overlap)==0 else 'FAIL'}")
if len(overlap) > 0:
    raise AssertionError(f"ERROR: Overlapping drugs found between train and test: {overlap}")
```

#### Risk of Data Contamination
There is **zero risk** that the same drug appears in both train and test sets:
1. The 12 training drugs and 16 testing drugs are physically disjoint sets ($\text{train\_drugs} \cap \text{test\_drugs} = \emptyset$).
2. Stochastic cell simulations (2,000 virtual cell replicates per drug) were generated independently for each compound.
3. Every script in the pipeline executes an explicit assertion verifying that the intersection set is empty prior to training or evaluation.

### 1.4 Class Distribution

Each drug is represented by 2,000 stochastic cell simulation rows (with the exception of dropped sentinel rows in `manual`).

| Split | Risk Tier | Drug Count | Row Count (Chantest / Nanion) | Row Count (Manual) | Percentage (Rows) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | High | 4 | 8,000 | 8,000 | 33.33% (Manual: 33.37%) |
| | Intermediate | 4 | 8,000 | 8,000 | 33.33% (Manual: 33.37%) |
| | Low | 4 | 8,000 | 7,975 | 33.33% (Manual: 33.26%) |
| | **Total Train** | **12** | **24,000** | **23,975** | **100.0%** |
| **Test** | High | 4 | 8,000 | 7,999 | 25.00% (Manual: 25.00%) |
| | Intermediate | 7 | 14,000 | 14,000 | 43.75% (Manual: 43.75%) |
| | Low | 5 | 10,000 | 10,000 | 31.25% (Manual: 31.25%) |
| | **Total Test** | **16** | **32,000** | **31,999** | **100.0%** |

---

## 2. Features / Biomarkers

### 2.1 Canonical Biomarkers Used by the Final Model

The model uses **exactly 11 electrophysiological cardiac biomarkers** (`CANONICAL_BIOMARKERS`) derived from simulated action potentials and intracellular calcium transients. The ordering is canonical and must be maintained identically across training and inference (`src/data_loader_rev.py` L7–10, `models/chantest_stacking_metadata_rev.json` L12–24, `docs/context.md` L68–90).

| Index | Column Name | Scientific Notation | Description | Physiological Unit |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `qNet` | $q_{\text{net}}$ | Net charge of total transmembrane ionic currents | pA·s (or $\mu\text{C}$) |
| 2 | `dvdtmax` | $dV_m/dt_{\max}$ | Maximum upstroke velocity during depolarization | V/s |
| 3 | `vmax` | $V_{m,\text{peak}}$ | Peak action potential membrane voltage | mV |
| 4 | `vrest` | $V_{m,\text{rest}}$ | Resting membrane potential | mV |
| 5 | `APD50` | $\text{APD}_{50}$ | Action potential duration at 50% repolarization | ms |
| 6 | `APD90` | $\text{APD}_{90}$ | Action potential duration at 90% repolarization | ms |
| 7 | `max_dv` | $dV_m/dt_{\text{repol}}$ | Maximum repolarization derivative velocity | V/s *(documented in `docs/context.md` L85)* |
| 8 | `camax` | $\text{Ca}_{\text{peak}}$ | Peak intracellular calcium concentration | $\mu\text{M}$ (or mM) |
| 9 | `carest` | $\text{Ca}_{\text{rest}}$ | Diastolic resting intracellular calcium concentration | $\mu\text{M}$ (or mM) |
| 10 | `CaTD50` | $\text{CaD}_{50}$ | Calcium transient duration at 50% recovery | ms |
| 11 | `CaTD90` | $\text{CaD}_{90}$ | Calcium transient duration at 90% recovery | ms |

*Total Count: Exactly 11 biomarker features.*

### 2.2 Dropped Columns & Rationale

The raw dataset CSV files contain 14 columns (Chantest and Manual) or 15 columns (Nanion). Exactly 3 or 4 columns are dropped during ingestion (`src/data_loader_rev.py` L101–108):

| Dropped Column | Present In | Description & Rationale for Exclusion |
| :--- | :--- | :--- |
| `drug_name` | All files | Compound identifier string. Extracted as the grouping variable (`groups`) for `StratifiedGroupKFold` and drug-level aggregation; dropped from feature matrix $X$ to prevent direct label leakage. |
| `Sample_ID` | All files | Simulation replicate index (1 to 2,000). Dropped as non-biological metadata. |
| `label` (or `Class`) | All files | Target clinical risk tier string (`high`, `intermediate`, `low`). Extracted and encoded as target vector $y$. |
| `Training_or_testing` | Nanion files | Dataset split metadata string (`training` or `testing`). Dropped as partition metadata. |

---

## 3. Preprocessing and Normalization

### 3.1 Preprocessing Sequence

The preprocessing pipeline executes the following deterministic steps in order (`src/data_loader_rev.py` L56–108):

1. **Raw Ingestion with Type Guard**: Raw CSV files are read with `dtype=str` and `keep_default_na=False` to prevent pandas from coercing sentinel error strings into silent floating-point NaNs.
2. **Sentinel Cell Forensic Audit**: Every cell in the 11 biomarker columns is audited against a blacklist of sentinel error strings (`#VALUE!`, `#DIV/0!`, `#N/A`, `#REF!`, `NaN`, `inf`, etc.).
3. **Drop-Not-Impute Policy**: Corrupted rows are dropped in-memory. Exactly **26 corrupted rows** containing the Excel formula error `'#VALUE!'` were identified and removed:
   - 25 rows in `manual_training_scaled_rev.csv` (all belonging to *mexiletine* simulations across columns `APD50`, `APD90`, and `max_dv`).
   - 1 row in `manual_testing_scaled_rev.csv` (*ibutilide* simulation in column `CaTD90`).
   - 0 rows in any Chantest or Nanion files.
   - All dropped rows are logged to `logs/DROPPED_ROWS_REV.csv`.
4. **Label Mapping**: Target labels are mapped to integer classes using:
   $$\text{LABEL\_MAP} = \{\text{'high'}: 0, \text{'intermediate'}: 1, \text{'low'}: 2\}$$
5. **Type Casting**: The 11 canonical biomarker columns are converted explicitly to `np.float64`.

### 3.2 Normalization Method

- **Method**: **No feature scaling is applied in the `_rev` pipeline**. Features are pre-normalized upstream against control simulations by the data provider (`src/data_loader_rev.py` L26–30, `scripts/pipeline/run_step3_stacking.py` L10–15).
- **Assertion Guard**: All training and evaluation scripts contain explicit assertions that raise a `RuntimeError` if any scaler module (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, etc.) is imported or instantiated.
- **Historical Note (Pre-_rev)**: In the legacy pipeline (`plus_ultra.ipynb` Cell 11), `StandardScaler` was fitted on the Chantest training split only and serialized inside `artifacts/deployable_chantest_3class.joblib`.
- **Inference in Hugging Face Space**: In `hf_space/app.py` (L65–67), the function `_preprocess` checks:
  ```python
  def _preprocess(raw):
      if 'scaler' in bundle and bundle['scaler'] is not None:
          return bundle['scaler'].transform(raw)
      return raw
  ```
  Because the updated bundle `hf_space/deployable_chantest_3class.joblib` (rebuilt on 2026-09-01 via `scripts/pipeline/build_hf_bundle_rev.py`) does not include a `'scaler'` key, `_preprocess` passes through the raw 11 biomarker inputs directly without modification.

---

## 4. Model

### 4.1 Architecture & Hyperparameters

The model architecture is a two-level **Stacking Classifier** (`sklearn.ensemble.StackingClassifier`). Two versions exist in the repository:
1. **Research Benchmark Model** (`models/chantest_stacking_model_rev.joblib`): Stacking with Random Forest meta-learner (`scripts/pipeline/run_step3_stacking.py`).
2. **Deployed Production Model** (`hf_space/deployable_chantest_3class.joblib`): Stacking with Logistic Regression meta-learner (`scripts/pipeline/build_hf_bundle_rev.py`).

```python
# Citation: scripts/pipeline/build_hf_bundle_rev.py (Lines 30-60)
rf_base = RandomForestClassifier(
    n_estimators=400,
    criterion='entropy',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

xgb_base = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=10,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)

# Logistic Regression meta-learner produces smooth calibrated probabilities
meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# Grouped CV splits to prevent drug leakage during out-of-fold meta-feature generation
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=False)
cv_splits = list(sgkf.split(X_train.values, y_train.values, groups=drugs_train))

stacking_model = StackingClassifier(
    estimators=[('RF', rf_base), ('XGB', xgb_base)],
    final_estimator=meta_learner,
    cv=cv_splits,
    stack_method='predict_proba',
    n_jobs=-1
)
```

#### Detailed Hyperparameters Table

| Learner Role | Algorithm | Hyperparameter | Configured Value | Verification Source |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1 Base** | `RandomForestClassifier` | `n_estimators` | `400` | `models/chantest_stacking_metadata_rev.json` L29 |
| | | `criterion` | `'entropy'` | `models/chantest_stacking_metadata_rev.json` L27 |
| | | `max_depth` | `None` | `models/chantest_stacking_metadata_rev.json` L28 |
| | | `random_state` | `42` | `scripts/pipeline/run_step3_stacking.py` L30 |
| **Level 1 Base** | `XGBClassifier` | `n_estimators` | `300` | `models/chantest_stacking_metadata_rev.json` L34 |
| | | `learning_rate` | `0.1` | `models/chantest_stacking_metadata_rev.json` L32 |
| | | `max_depth` | `10` | `models/chantest_stacking_metadata_rev.json` L33 |
| | | `eval_metric` | `'mlogloss'` | `scripts/pipeline/build_hf_bundle_rev.py` L43 |
| | | `random_state` | `42` | `models/chantest_stacking_metadata_rev.json` L4 |
| **Level 2 Meta (Research)** | `RandomForestClassifier` | `random_state` | `42` | `models/chantest_stacking_metadata_rev.json` L37 |
| **Level 2 Meta (Deployed)** | `LogisticRegression` | `C` | `1.0` | `scripts/pipeline/build_hf_bundle_rev.py` L48 |
| | | `max_iter` | `1000` | `scripts/pipeline/build_hf_bundle_rev.py` L48 |
| | | `random_state` | `42` | `scripts/pipeline/build_hf_bundle_rev.py` L48 |

### 4.2 Stacking Mechanism & Leakage-Free CV

- **Implementation**: Built via `sklearn.ensemble.StackingClassifier`.
- **CV Scheme for Out-of-Fold Meta-Features**: 4-fold `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` grouped strictly by `drug_name`. The fold split indices are pre-generated as a list:
  ```python
  cv_splits = list(sgkf.split(X_train.values, y_train.values, groups=drugs_train))
  ```
  and passed to `StackingClassifier(cv=cv_splits)`. This ensures that out-of-fold probability predictions are computed with 100% whole-drug holdout isolation (`scripts/pipeline/run_step3_stacking.py` L181–186).
- **Passthrough**: `passthrough=False` (the meta-learner receives only the out-of-fold predicted class probabilities from Level 1, not the raw biomarker features).
- **Meta-Learner Feed**: `stack_method='predict_proba'` (predicted probability vectors $[p_{\text{high}}, p_{\text{int}}, p_{\text{low}}]$ feed Level 2).
- **Class-Imbalance Handling**: The training split has a balanced 1:1:1 ratio across classes (4 drugs each, 8,000 samples per class). No class weighting (`class_weight`) or synthetic resampling (SMOTE) was used.
- **Random Seed**: `SEED = 42` is set uniformly across all random operations.

### 4.3 Artifact Storage & Serialization

| Artifact File | Format | File Size | Location | Role & Description |
| :--- | :---: | :---: | :--- | :--- |
| `chantest_stacking_model_rev.joblib` | Joblib pickle | 4,124,191 B (3.93 MB) | `models/` | Frozen research benchmark model (meta = Random Forest). SHA-256: `A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2` |
| `deployable_chantest_3class.joblib` | Joblib pickle (dict bundle) | 3,936,550 B (3.75 MB) | `hf_space/` | Deployed production model bundle (meta = Logistic Regression), containing model, class maps, biomarker order, and conformal operating parameters. |
| `shap_background.joblib` | Joblib pickle | 2,644 B | `hf_space/` | 15 k-means background centroids generated from training data for runtime SHAP explanations. |

---

## 5. Experiments and Evaluation Results

### 5.1 Final Model Performance (Research Stacking Model)

All numbers are verified verbatim from canonical sources (`docs/PAPER_NUMBERS.md` L24–52, `docs/STEP3_STACKING_CHANTEST.md` L219–259, `results/bootstrap_distributions_rev.csv`, `docs/STEP20_UNIFIED_TABLE.md` L54–70).

#### Primary Evaluation: Drug-Level Performance (16 Held-Out Test Drugs)

*Note: In safety pharmacology (CiPA), drug-level evaluation is the primary metric because preclinical safety decisions evaluate compound entities rather than isolated cell simulation replicates (`docs/STEP3_STACKING_CHANTEST.md` L270–274).*

| Assay Platform | Drug-Level Macro-F1 (Point) | Bootstrap Median F1 [95% CI] | Accuracy (Correct/16) | Precision (High / Int / Low) | Recall (High / Int / Low) | Per-Class F1 (High / Int / Low) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chantest** | **0.694444** | 0.679739 [0.391534, 0.896825] | 0.6875 (11/16) | 0.600 / 0.625 / 1.000 | 0.750 / 0.714 / 0.600 | 0.667 / 0.667 / 0.750 |
| **Nanion** | **0.621032** | 0.595238 [0.311111, 0.851852] | 0.6250 (10/16) | 0.600 / 0.556 / 1.000 | 0.750 / 0.714 / 0.400 | 0.667 / 0.625 / 0.571 |
| **Manual** | **0.754762** | 0.738889 [0.479347, 0.944056] | 0.7500 (12/16) | 0.667 / 0.714 / 1.000 | 1.000 / 0.714 / 0.600 | 0.800 / 0.714 / 0.750 |

#### Drug-Level Confusion Matrices (Rows = True, Columns = Pred)

- **Chantest** (`docs/PAPER_NUMBERS.md` L33):
  $$\begin{pmatrix} \text{High} \\ \text{Intermediate} \\ \text{Low} \end{pmatrix} \rightarrow \begin{pmatrix} 3 & 1 & 0 \\ 2 & 5 & 0 \\ 0 & 2 & 3 \end{pmatrix}$$
  *(Disopyramide under-triaged to Intermediate; Domperidone and Pimozide over-called to High; Loratadine and Tamoxifen over-called to Intermediate).*
- **Nanion** (`docs/PAPER_NUMBERS.md` L42):
  $$\begin{pmatrix} 3 & 1 & 0 \\ 2 & 5 & 0 \\ 0 & 3 & 2 \end{pmatrix}$$
  *(Metoprolol, Loratadine, Tamoxifen over-called to Intermediate).*
- **Manual** (`docs/PAPER_NUMBERS.md` L51):
  $$\begin{pmatrix} 4 & 0 & 0 \\ 2 & 5 & 0 \\ 0 & 2 & 3 \end{pmatrix}$$
  *(Zero High-risk misses; Metoprolol and Tamoxifen over-called to Intermediate).*

#### Secondary Evaluation: Row-Level Performance (32,000 Test Rows)

| Assay Platform | Row-Level Macro-F1 | Row-Level Accuracy | Row-Level Confusion Matrix |
| :--- | :---: | :---: | :--- |
| **Chantest** | 0.688289 | 0.682344 | High: [6591, 1402, 7]; Int: [4178, 9785, 37]; Low: [1, 4540, 5459] |
| **Nanion** | 0.632400 | 0.628875 | High: [6289, 1709, 2]; Int: [4277, 9690, 33]; Low: [4, 5849, 4147] |
| **Manual** | 0.692524 | 0.686834 | High: [7468, 529, 2]; Int: [4296, 9668, 36]; Low: [0, 5155, 4845] |

#### Ordinal Binary Threshold AUC

Evaluated at clinical cut-points $\text{Ambang } 1: P(\text{risk} > \text{Low})$ and $\text{Ambang } 2: P(\text{risk} > \text{Intermediate})$ (`docs/STEP18A_ORDINAL_BINARY_AUC.md` L8–28, L84–94):

| Platform | Evaluation Level | AUC Ambang 1: $P(\text{risk} > \text{Low})$ | AUC Ambang 2: $P(\text{risk} > \text{Int})$ |
| :--- | :--- | :---: | :---: |
| **Chantest** | Drug-level | 0.981818 | 0.875000 |
| | Row-level | 0.910365 | 0.857566 |
| **Nanion** | Drug-level | 0.981818 | 0.875000 |
| | Row-level | 0.808630 | 0.842805 |
| **Manual** | Drug-level | 0.927273 | 0.958333 |
| | Row-level | 0.835618 | 0.924437 |

### 5.2 Base Learner Reference Performance

Evaluated on frozen base estimators without retraining (`docs/PAPER_NUMBERS.md` Section H1, `docs/STEP11_BASE_LEARNER_REFERENCE.md` L1–40):

| Platform | Estimator | Drug-Level Macro-F1 | Correct / 16 | Under-Triaged Count | AUC Ambang 1 | AUC Ambang 2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Chantest** | Random Forest (RF) | **0.877778** | 14/16 | 0 | 1.0000 | 0.9167 |
| | XGBoost (XGB) | 0.819373 | 13/16 | 0 | 1.0000 | 0.9167 |
| | Stacking (STACK) | 0.694444 | 11/16 | 1 (*disopyramide*) | 0.9818 | 0.8750 |
| **Nanion** | Random Forest (RF) | **0.819373** | 13/16 | 0 | 0.9636 | 0.8958 |
| | XGBoost (XGB) | 0.754762 | 12/16 | 0 | 0.9818 | 0.9375 |
| | Stacking (STACK) | 0.621032 | 10/16 | 1 (*disopyramide*) | 0.9818 | 0.8750 |
| **Manual** | Random Forest (RF) | 0.755556 | 12/16 | 1 (*clarithromycin*) | 0.9455 | 0.9583 |
| | XGBoost (XGB) | **0.819373** | 13/16 | 0 | 0.8727 | 0.9792 |
| | Stacking (STACK) | 0.754762 | 12/16 | 0 | 0.9273 | 0.9583 |

#### Paired Bootstrap Differences (STACK minus Base Learner, 10,000 Iterations)
- **Chantest**:
  - STACK minus RF: Diff point = `-0.183333`, Bootstrap Median = `-0.187302` [95% CI: `-0.448155, +0.000000`].
  - STACK minus XGB: Diff point = `-0.124929`, Bootstrap Median = `-0.123151` [95% CI: `-0.357156, +0.000000`].
- **Nanion**:
  - STACK minus RF: Diff point = `-0.198341`, Bootstrap Median = `-0.197739` [95% CI: `-0.454772, +0.000000`].
  - STACK minus XGB: Diff point = `-0.133730`, Bootstrap Median = `-0.126623` [95% CI: `-0.374185, +0.000000`].
- **Manual**:
  - STACK minus RF: Diff point = `-0.000794`, Bootstrap Median = `+0.000000` [95% CI: `-0.197902, +0.188889`].
  - STACK minus XGB: Diff point = `-0.064611`, Bootstrap Median = `-0.061601` [95% CI: `-0.241282, +0.000000`].

*(Finding: Individual base learners achieve higher point estimates than the Stacking model across platforms, but all 95% bootstrap difference CIs encompass zero; `docs/PAPER_NUMBERS.md` L217–227).*

### 5.3 Comparison with Other Approaches

#### 1. Baseline Logistic Regression Models (`docs/STEP19_LOGISTIC_BASELINE.md`, `docs/STEP20_UNIFIED_TABLE.md`)
- **Multivariate LR (11 biomarkers)**: Drug-level Macro-F1 = `0.8194` (Chantest), `0.7548` (Nanion), `0.8194` (Manual). Correct: 13/16 (Chantest), 12/16 (Nanion), 13/16 (Manual). Zero under-triage across all three platforms.
- **Single-Biomarker LR (`qNet`-only)**: Drug-level Macro-F1 = `0.8120` (Chantest), `0.7566` (Nanion), `0.7515` (Manual). Correct: 13/16, 12/16, 12/16. Under-triage: 1 on each platform (*disopyramide* on Chantest/Nanion; *risperidone* on Manual).

#### 2. Historical Combinatorial Sweep (612 Configurations, Pre-_rev)
Documented in `artifacts/phase3_protocolA_results.csv` and `docs/context.md` (L215–244):
- Evaluated 3 datasets × 17 model combinations × 6 ensemble types.
- **Hard Voting (7 base learners: RF, XGB, LR, SVM, ANN1–3)**: Macro-F1 = `0.826` (3-class), `0.902` (binary).
- **Soft Voting (7 base learners)**: Macro-F1 = `0.824` (3-class), `0.901` (binary).
- **Stacking (7 base learners, LR meta)**: Macro-F1 = `0.802`.
- **Stacking (3 ANN base learners, LR meta)**: Macro-F1 = `0.820`.
- **Pure-ML Stacking (RF+XGB, meta=RF)**: Macro-F1 = `0.828004` (highest amongst pure-ML models).

#### 3. Feature Count Comparison (14 vs. 11 Biomarkers)
- **Status**: **NOT FOUND** as an experimental comparison. The number 14 corresponds to the total number of raw columns in the input CSV files (11 biomarkers + 3 metadata columns), not a 14-biomarker feature subset.

#### 4. Normalization Comparison (Z-Score vs. Min-Max)
- **Status**: **NOT FOUND**. Min-Max scaling was never tested in any experimental script and is explicitly banned by assertion guards in `scripts/pipeline/run_step3_stacking.py` (L10–15). Z-score scaling (`StandardScaler`) was used only in the pre-_rev pipeline.

### 5.4 Bootstrap and Confidence Interval Analysis

1. **Drug-Level Bootstrap (10,000 Iterations, Seed 42)** (`docs/STEP9_BOOTSTRAP.md`, `results/bootstrap_distributions_rev.csv`):
   - Macro-F1 Median [95% CI]: Chantest = `0.679739 [0.391534, 0.896825]`; Nanion = `0.595238 [0.311111, 0.851852]`; Manual = `0.738889 [0.479347, 0.944056]`.
   - **Under-Triage Rate Rule-of-Three Bound**: Because 0 under-triages were observed across the test set under conformal Rank 11, the standard bootstrap CI collapses to `[0, 0]`. Applying the statistical Rule of Three ($3/n = 3/16$) establishes a conservative **95% upper bound of 18.75%** (`0.187500`).
2. **CiPA Replicate Bootstrap (Li et al. Protocol, 10,000 Iterations)** (`docs/STEP22_CIPA_BOOTSTRAP.md`, `results/cipa_bootstrap_rev.csv`):
   - Resamples 1 simulated cell replicate per drug across 16 drugs paired across models.
   - Chantest AUC [95% CI]: High = `0.854 [0.771, 0.938]`; Intermediate = `0.746 [0.651, 0.849]`; Low = `0.918 [0.709, 1.000]`.

### 5.5 Inventory of Saved Plot Files

| Plot File Path | Dimensions | File Size | Description |
| :--- | :---: | :---: | :--- |
| `results/fig1_block_diagram.pdf` | 7.16 × 3.70 in | 37.7 KB | Vector block diagram of data ingestion, grouped CV, and stacking architecture. |
| `results/fig1_block_diagram.png` | Raster preview | 114.5 KB | PNG preview of Figure 1. |
| `results/fig1_pipeline_color.pdf` | Vector | 40.2 KB | Alternative full-color pipeline block diagram. |
| `results/fig_shap_global.pdf` | 3.40 × 3.40 in | 11.4 KB | Vector bar plot of 11-biomarker global SHAP importance (Stacking model). |
| `results/fig_shap_global.png` | Raster preview | 32.9 KB | PNG preview of SHAP global importance. |
| `results/fig_shap_xgb.pdf` | 3.50 × 3.00 in | 11.8 KB | Vector bar plot of SHAP importance for the XGBoost base learner. |
| `results/fig_shap_xgb.png` | Raster preview | 33.1 KB | PNG preview of XGBoost SHAP bar plot. |
| `paper_figures/fig_architecture.pdf` | 12.0 × 6.50 in | 32.4 KB | End-to-end architecture and deployment pipeline vector diagram. |

*(Note: 15 legacy PNG plots reside in `artifacts/figures/` but are quarantined as obsolete).*

---

## 6. Explainability (SHAP)

### 6.1 SHAP Explainer Formulation & Target

- **Explainer Type**: `shap.KernelExplainer` (model-agnostic kernel explainer) (`scripts/pipeline/run_step7_shap.py` L60–64, `hf_space/app.py` L14).
- **Target Function**: The full fitted stacking model's **`predict_proba`** pipeline function is explained as a unified black box.
  - *Rationale*: StackingClassifier is not supported by `shap.TreeExplainer`. Explaining the Level 2 meta-learner alone would attribute importance only to Level 1 prediction probabilities, losing the direct biological relationship to the 11 cardiac biomarkers (`scripts/pipeline/run_step7_shap.py` L60–64).

### 6.2 Background Reference Dataset

- **Training Pipeline Explainer** (`scripts/pipeline/run_step7_shap.py` L81): 100 random rows sampled from the Chantest training dataset (`shap.sample(X_train_full, 100, random_state=42)`).
- **Hugging Face Space Explainer** (`scripts/pipeline/build_hf_bundle_rev.py` L91–95, `hf_space/app.py` L13): **15 k-means centroids** generated via `shap.kmeans(X_train.values, 15)` and persisted to `hf_space/shap_background.joblib` (2,644 bytes). K-means summarization was selected to minimize runtime inference latency in the web container.

### 6.3 Explainability Outputs & Serving Mode

- **Batch Pipeline Outputs**: Global importance rankings saved to `results/shap_global_importance_rev.csv` and rendered to `results/fig_shap_global.pdf`.
- **Hugging Face Serving Mode**: Computed **dynamically per request** on user demand when the client submits a request to the `/explain` endpoint (`hf_space/app.py` L99–134). The function executes `explainer.shap_values(Xs, nsamples=500)` for the predicted class.

### 6.4 Top-Ranked Biomarkers (Global Importance)

Verified from `results/shap_global_importance_rev.csv` (aggregated mean absolute SHAP across classes on Chantest test split):

| Rank | Biomarker Feature | Mean \|SHAP\| (High) | Mean \|SHAP\| (Int) | Mean \|SHAP\| (Low) | Aggregated Mean \|SHAP\| |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `qNet` | 0.129443 | 0.132843 | 0.174315 | **0.145534** |
| 2 | `vrest` | 0.028388 | 0.171309 | 0.146771 | **0.115489** |
| 3 | `max_dv` | 0.038575 | 0.052062 | 0.049949 | **0.046862** |
| 4 | `APD50` | 0.053535 | 0.046643 | 0.039963 | **0.046714** |
| 5 | `APD90` | 0.016335 | 0.053746 | 0.045357 | **0.038479** |
| 6 | `vmax` | 0.017772 | 0.022985 | 0.028067 | **0.022941** |
| 7 | `dvdtmax` | 0.005932 | 0.019149 | 0.013368 | **0.012816** |
| 8 | `camax` | 0.012177 | 0.010168 | 0.008178 | **0.010174** |
| 9 | `CaTD50` | 0.011085 | 0.010114 | 0.006511 | **0.009237** |
| 10 | `CaTD90` | 0.008137 | 0.007551 | 0.005108 | **0.006932** |
| 11 | `carest` | 0.004111 | 0.006886 | 0.007589 | **0.006195** |

#### The `vrest` Diagnostic Finding (`docs/STEP8_SHAP_DIAGNOSTIC.md`, `docs/STORYLINE.md` L221–240)
Although `vrest` ranks #2 by mean|SHAP|, a forensic tree split audit revealed that:
1. In Random Forest, `vrest` was split 1,533 times at hyper-precise 8-decimal threshold differences where coefficient of variation is on the order of $10^{-4}$.
2. In XGBoost, `vrest` was split **0 times** across all 300 trees.
3. In permutation testing, permuting `vrest` changed only 1.0 of 16 drug predictions (macro-F1 drop of 0.073), whereas permuting `qNet` changed 3.0 drug predictions (macro-F1 drop of 0.108).
4. **Conclusion**: `vrest`'s high SHAP ranking is an artifact of KernelExplainer instability on collinear/low-variance features. The primary biological interpretability anchor is **`qNet`** (Rank 1).

---

## 7. Hugging Face Space (Model Serving)

### 7.1 Architecture & Serving Framework

- **Serving Framework**: **Gradio** (`gradio==5.23.0`) hosted on Hugging Face Spaces (`mitcheltastic/tdp-cipa-screening`).
- **Entry File**: `hf_space/app.py`.
- **Runtime Model Loading**: The model bundle (`deployable_chantest_3class.joblib`) and background centroids (`shap_background.joblib`) are **loaded once into global memory at startup** (`hf_space/app.py` L12–14).
- **Authentication**: **None** (public Space). If set to private, requests require `Authorization: Bearer <HF_TOKEN>`.
- **Container Infrastructure**: Free CPU basic tier (`cpu-basic`, 2 vCPU, 16 GB RAM), exposing standard Gradio port `7860`. No custom `Dockerfile` is used (standard Hugging Face Gradio SDK).

### 7.2 API Endpoint Specifications

The Space exposes two primary endpoints via Gradio's REST API (`docs/guide for front-end.md`, `hf_space/app.py` L169–170). Gradio implements a two-step execution pattern: `POST /gradio_api/call/<fn>` returns an `event_id`, followed by `GET /gradio_api/call/<fn>/<event_id>` which streams results via Server-Sent Events (SSE).

| Endpoint Path | HTTP Method | Input Parameters | Output Components | HTTP Error Codes |
| :--- | :---: | :--- | :--- | :---: |
| `/gradio_api/call/predict` | `POST` | 11 float numbers in canonical order: `qNet`, `dvdtmax`, `vmax`, `vrest`, `APD50`, `APD90`, `max_dv`, `camax`, `carest`, `CaTD50`, `CaTD90` | 1. Class probabilities dict (`Label`)<br>2. Predicted risk tier string (`Textbox`)<br>3. Conformal prediction set dict (`JSON`)<br>4. Recommended action UI markdown (`Markdown`) | 422 (Validation), 405 (Method/Prefix error), 503 (Cold start) |
| `/gradio_api/call/explain` | `POST` | Same 11 float numbers in canonical order | Single JSON object containing predicted class, base value, and 11 feature SHAP contributions | 422 (Validation), 405 (Prefix error), 503 (Cold start) |

### 7.3 Request and Response Examples

#### 1. Prediction Endpoint (`/gradio_api/call/predict`)

**Request Payload:**
```json
{
  "data": [
    43.637118,
    72.879391,
    41.826454,
    -88.012307,
    338.5,
    441.25,
    -0.238297,
    0.000254,
    0.000076,
    337.5,
    767.25
  ]
}
```

**Response Payload (from SSE stream):**
```json
[
  {
    "high": 0.989995,
    "intermediate": 0.009950,
    "low": 0.000055
  },
  "Predicted Risk Tier: High",
  {
    "prediction_set": ["high"],
    "recommended_action": "high",
    "set_size": 1,
    "is_ambiguous": false,
    "out_of_distribution": false,
    "alpha": 0.154,
    "q_hat": 0.6275,
    "calibration_note": "Conformal threshold calibrated at drug level on probability vectors averaged over 2000 replicate samples per drug (leave-one-drug-out over 12 training drugs). This response is based on a single input row; the coverage guarantee does not strictly transfer to single-row input."
  },
  "### Recommended Action: High"
]
```

#### 2. Explanation Endpoint (`/gradio_api/call/explain`)

**Request Payload:**
```json
{
  "data": [
    43.637118,
    72.879391,
    41.826454,
    -88.012307,
    338.5,
    441.25,
    -0.238297,
    0.000254,
    0.000076,
    337.5,
    767.25
  ]
}
```

**Response Payload:**
```json
{
  "predicted_class": "high",
  "base_value": 0.334712,
  "contributions": [
    { "biomarker": "qNet", "value": 43.637118, "shap": 0.300812 },
    { "biomarker": "dvdtmax", "value": 72.879391, "shap": -0.004123 },
    { "biomarker": "vmax", "value": 41.826454, "shap": 0.012450 },
    { "biomarker": "vrest", "value": -88.012307, "shap": 0.083210 },
    { "biomarker": "APD50", "value": 338.5, "shap": 0.045120 },
    { "biomarker": "APD90", "value": 441.25, "shap": 0.108950 },
    { "biomarker": "max_dv", "value": -0.238297, "shap": 0.061200 },
    { "biomarker": "camax", "value": 0.000254, "shap": 0.015200 },
    { "biomarker": "carest", "value": 0.000076, "shap": 0.003100 },
    { "biomarker": "CaTD50", "value": 337.5, "shap": 0.010400 },
    { "biomarker": "CaTD90", "value": 767.25, "shap": 0.009200 }
  ]
}
```

### 7.4 Input Validation, Latency & Logging

- **Input Validation**: Limited to Gradio's standard numeric type casting (`gr.Number`). **No explicit Pydantic schema or physical range constraints** (e.g. bounding APD values to positive numbers) are implemented in `app.py`.
- **Latency Measurement & Logging**: **NOT FOUND**. Neither `hf_space/app.py` nor `deploy_utils.py` contains latency timers, request logging middleware, or structured audit hooks.

---

## 8. Exact Versions

### 8.1 Python Version

- **Local Development / Training Environment**: `Python 3.12` (`Python 3.12.0` on system PATH, `docs/context_nextstep.md` L167).
- **Hugging Face Space Virtual Environment**: `Python 3.12.9` (verified from `.venv/pyvenv.cfg` in `hf_space/`).

### 8.2 Package Dependencies

#### Training Environment
*(Note: There is NO root-level `requirements.txt` or `pyproject.toml` in the repository. Dependencies are determined from `models/chantest_stacking_metadata_rev.json` L5–6, `README.md` L112–121, and `docs/context_nextstep.md` L167).*

- `scikit-learn`: `1.8.0` (strictly pinned in model metadata)
- `xgboost`: `3.2.0` (strictly pinned in model metadata)
- `shap`: `0.52.0`
- `numpy`: installed (version unpinned at root)
- `pandas`: installed (version unpinned at root)
- `joblib`: installed (version unpinned at root)
- `matplotlib`: installed (version unpinned at root)
- `seaborn`: installed (version unpinned at root)
- `tensorflow` / `keras`: `2.21.0` *(historical Phase 2 ANN tuning)*
- `scikeras`: `0.13.0` *(historical Phase 2 ANN tuning)*

#### Hugging Face Space Environment (`hf_space/requirements.txt`)

- `scikit-learn==1.8.0`
- `xgboost==3.2.0`
- `shap==0.52.0`
- `numpy`
- `pandas`
- `joblib`
*(Gradio `5.23.0` and FastAPI `0.115.11` are supplied automatically by the Hugging Face Spaces Gradio SDK).*

---

## 9. History of Design Decisions

### 9.1 Chronological Git Log Summary

```
774f09f | 2026-09-01 | scripts: add automatic .env loading in deploy_hf.py
8c50fff | 2026-09-01 | hf_space: upgrade Stacking Classifier to use Logistic Regression meta-learner and pre-scaled _rev dataset
3c8b3e1 | 2026-09-01 | docs/fig1: update Figure 1 pipeline diagram and add XGBoost conformal/SHAP results
9b5045e | 2026-07-31 | code archiving
c336a1a | 2026-07-27 | figures update
5692e00 | 2026-07-25 | Re-run full pipeline on _rev datasets
f6f5114 | 2026-07-25 | progress
baa6fd4 | 2026-07-16 | Add conformal prediction evaluation and rendering fixes
838b051 | 2026-07-06 | push deployment
42498ca | 2026-06-30 | Initial commit: TdP cardiotoxicity prediction pipeline
```

### 9.2 Key Methodological Transitions & Rationales

#### 1. Transition from Voting to Stacking Ensembles
- **Origin**: The undergraduate thesis proposal ("sempro") originally specified a 3-tier consensus voting approach as the primary architecture (`docs/context.md` L21–22).
- **Sweep Evidence**: In the Phase 3 combinatorial sweep across 612 configurations (`artifacts/phase3_protocolA_results.csv`, `docs/PROJECT_SUMMARY.md` L123–124), Stacking ensembles consistently outperformed soft/hard voting:
  - Best Stacking F1 = `0.828004`
  - Best Voting F1 = `0.821999` (Hard Voting: `0.826`, Soft Voting: `0.824`)
- **Decision**: Stacking (`meta=RF`) was selected because it achieved higher predictive performance and invariant victory across all three dataset splits (`docs/PROJECT_SUMMARY.md` L124).

#### 2. Elimination of Deep Learning (ANN) & Choice of Pure-ML Base Learners
- **Origin**: Early iterations featured 7 base learners: 4 ML (RF, XGB, LR, SVM) and 3 DL ANNs (1, 2, and 3 hidden layers in Keras/TensorFlow via SciKeras; `docs/context.md` L146–166).
- **Examiner Feedback & Technical Rationale**: Sempro thesis examiners objected to mixing ML and DL paradigms within the same ensemble (`docs/context.md` L414–417). Furthermore, pure-ML `RF+XGB` stacking achieved nearly identical performance to mixed ensembles (`0.828` vs. `0.817`–`0.842`) while eliminating deep learning dependencies (TensorFlow/Keras/SciKeras), cutting inference latency, and ensuring complete model explainability (`docs/PROJECT_SUMMARY.md` L132).
- **Candidate Libraries (LightGBM & CatBoost)**: Mentioned as potential future work in `docs/context.md` (L425), but never implemented in the pipeline.

#### 3. Discovery of Drug-Level Data Leakage (The "Crown Jewel")
- **Problem**: Naive row-level `StratifiedKFold` cross-validation yielded an inflated Macro-F1 of $\approx 0.999$. Because 2,000 simulated cells represent the same physical compound, row-level splitting scattered identical compound signatures across training and validation folds, causing classifiers to memorize drug-specific features.
- **Diagnostic**: Evaluating under whole-drug holdout collapsed validation F1 to $\approx 0.460$ (`README.md` L69, `docs/context_nextstep.md` L132–137).
- **Resolution**: Enforced 4-fold `StratifiedGroupKFold` grouped strictly by `drug_name` across all hyperparameter tuning, out-of-fold stacking meta-feature generation, and bootstrap validation.

#### 4. Dataset Replacement (`_rev`) & Abandonment of StandardScaler
- **Transition (2026-07-25, commit `5692e00`)**: The simulation data provider supplied revised datasets (`_rev`) where biophysical outputs were pre-normalized against control simulations (`docs/STORYLINE.md` L51–54).
- **Decision**: Applying `StandardScaler` or `MinMaxScaler` on pre-normalized features would distort the control scaling. All scalers were removed from the training pipeline and barred by assertion guards. All prior numbers in `artifacts/` were quarantined as obsolete (`artifacts/OBSOLETE_README.md`).

#### 5. Meta-Learner Replacement for Production Serving (commit `8c50fff`)
- **Pathology Discovered in Research Model**: In `docs/STEP12_STACK_VS_BASE_DIAGNOSTIC.md`, the Random Forest meta-learner was found to override correct predictions from both base learners (*disopyramide*, *loratadine*) and compress predictions into the Intermediate tier. Furthermore, decision tree leaves generated discrete step-function probability outputs (often `1.0 / 0.0`).
- **Resolution (2026-09-01)**: The script `scripts/pipeline/build_hf_bundle_rev.py` was created to rebuild the deployable model using **`LogisticRegression(C=1.0, max_iter=1000)`** as the meta-learner. This produced smooth, well-calibrated class probabilities suitable for the Gradio interface and client applications.

---

## 10. Gaps and Discrepancies

The following discrepancies, hard-coded values, and documentation gaps were identified in the repository audit:

### 10.1 Discrepancy Between Research Checkpoint and Deployed Space Model
- **Finding**: The research model artifact cited throughout the academic documentation (`models/chantest_stacking_model_rev.joblib`, SHA-256 `A93B9868...`) utilizes a **Random Forest meta-learner** (`RandomForestClassifier(random_state=42)`). In contrast, the model deployed in the Hugging Face Space (`hf_space/deployable_chantest_3class.joblib`) utilizes a **Logistic Regression meta-learner** (`LogisticRegression(max_iter=1000, random_state=42)`).
- **Impact**: Predictions and class probabilities generated by the live Hugging Face Space will differ slightly from the static research evaluation tables in `docs/PAPER_NUMBERS.md`.

### 10.2 Conformal Calibration Value Mismatch
- **In Research Benchmark**: The adopted operating point is **Rank 11** with $\hat{q} = 0.762830$ (inclusion threshold $1 - \hat{q} = 0.237170$), nominal coverage $11/13 \approx 84.62\%$, and $\alpha = 2/13 \approx 0.1538$ (`docs/PAPER_NUMBERS.md` L58).
- **In Hugging Face Space**: `hf_space/app.py` (L16–17) hardcodes `Q_HAT = 0.6275` (inclusion threshold $1 - \hat{q} = 0.3725$) and `ALPHA = 0.154`.
- **Cause**: The threshold `0.6275` was computed on the newly fitted Logistic Regression bundle in `build_hf_bundle_rev.py` (L75), rather than the research Random Forest stacking model. The two thresholds reflect different underlying probability distributions.

### 10.3 Single-Row Inference vs. Drug-Averaged Calibration
- **Finding**: Conformal non-conformity scores were calibrated at the **drug level** by averaging probability vectors over 2,000 virtual cell replicates per drug ($s_d = 1 - \bar{p}[y]$). In production (`hf_space/app.py`), the model evaluates a single raw input row.
- **Mitigation in Code**: `hf_space/app.py` displays a hardcoded disclaimer (`CALIBRATION_NOTE` L19–24) acknowledging that the nominal $84.6\%$ coverage guarantee strictly applies to drug-level aggregated averages and does not formally transfer to single-cell inputs.

### 10.4 Frontend API Contract Drift
- **Finding**: `docs/guide for front-end.md` (L49–59) documents that `/predict` returns a 2-element array `[probs, tier]`.
- **Actual Code**: In `hf_space/app.py` (L97, L169), `predict()` returns a **4-element tuple**: `[prob_dict, legacy_str, conformal_dict, ui_str]`. Any client parsing the response as a 2-element list will fail.

### 10.5 Missing Input Range Bounds
- **Finding**: Inputs in `hf_space/app.py` are declared as unbounded `gr.Number()` fields. No checks exist to reject negative values for physiologically positive parameters (e.g. `APD50`, `APD90`, `CaTD50`, `CaTD90`). Submitting unphysical numbers (e.g. negative durations) will produce synthetic inferences without raising an error.

### 10.6 Vestigial Scaler Check in Serving Layer
- **Finding**: `hf_space/app.py` (L65–67) retains a check for `bundle['scaler']`. Because the `_rev` bundle intentionally omitted the scaler, this code path is dead and returns raw inputs directly.

### 10.7 Undocumented Upstream Simulation Normalization
- **Finding**: The mathematical equation and exact baseline values used by the external data provider to normalize biophysical simulation outputs into `_rev` features are not documented anywhere in this repository (`docs/STORYLINE.md` L270).

### 10.8 Missing Root-Level Dependency Specifications
- **Finding**: There is no root-level `requirements.txt`, `environment.yml`, or `pyproject.toml` documenting the complete training environment. Only `hf_space/requirements.txt` exists.

---

*Report compiled on 2026-09-28 based strictly on verified code, metadata, git history, and evaluation artifacts in the repository.*
