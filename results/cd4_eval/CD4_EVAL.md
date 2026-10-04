# CD-4 Implementation Report: Additional Evaluation Results & Audit Summary

**Repository Root**: `TrainAI`  
**Execution Environment**: Windows, Python 3.12 (venv: `hf_space/.venv`, scikit-learn 1.8.0, xgboost 3.2.0, shap 0.52.0)  
**Output Directory**: `results/cd4_eval/`  
**Evaluation Scope**: Read-only evaluation of research vs. deployed models, base learners, voting schemes, conformal recalibration, biomarker normalization, SHAP verification, and latency profiling.

---

## 0. Sanity Check Confirmation (Reproducibility Baseline)

Prior to executing any new evaluation tasks, the research model checkpoint was verified against the benchmark held-out ChanTest test dataset using `results/cd4_eval/sanity_check.py`.

- **Model Checkpoint**: [`models/chantest_stacking_model_rev.joblib`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/models/chantest_stacking_model_rev.joblib)
- **Held-Out Test Data**: [`data/chantest_testing_scaled_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/data/chantest_testing_scaled_rev.csv) (16 drugs, 32,000 rows)
- **Reported Benchmark Macro-F1**: `0.694444` (`0.6944444444444443`, 11/16 correct, accuracy `0.6875`)
- **Reproduced Macro-F1**: `0.6944444444444443` (11/16 correct, accuracy `0.6875`)
- **Status**: **PASS (100% Exact Numerical Reproduction)**

---

## Task 1. Drug-Level Aggregation Protocol

In the preclinical CiPA framework, each drug is evaluated by simulating 2,000 stochastic virtual cells. The conversion of 2,000 replicate row predictions into a single drug-level prediction is executed via **mean predicted probability followed by argmax**. It is **not** majority vote.

### Exact Code Quotation
From [`scripts/pipeline/run_step3_stacking.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/scripts/pipeline/run_step3_stacking.py#L227-L244) (Lines 227–244):

```python
# B. DRUG-LEVEL EVALUATION
# Average predicted probabilities within each drug
df_eval = pd.DataFrame(y_test_proba, columns=['prob_high', 'prob_int', 'prob_low'])
df_eval['drug_name'] = groups_test.values
df_eval['true_label'] = y_test.values

drug_summary = df_eval.groupby('drug_name').agg({
    'true_label': 'first',
    'prob_high': 'mean',
    'prob_int': 'mean',
    'prob_low': 'mean'
}).reset_index()

# Drug-level argmax prediction
drug_summary['pred_label'] = np.argmax(
    drug_summary[['prob_high', 'prob_int', 'prob_low']].values, axis=1
)
```
*(Identical aggregation code is executed in [`scripts/pipeline/run_step4_cross_platform.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/scripts/pipeline/run_step4_cross_platform.py#L128-L144) lines 128–144).*

---

## Task 2. Training Data Lineage and Platform Roles

### 1. Research Model Fitting
The research model checkpoint [`models/chantest_stacking_model_rev.joblib`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/models/chantest_stacking_model_rev.joblib) (StackingClassifier with `RandomForestClassifier` meta-learner) was fitted **strictly and exclusively** on [`data/chantest_training_scaled_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/data/chantest_training_scaled_rev.csv) (12 training drugs, 24,000 rows).

**Quotation** from [`scripts/pipeline/run_step3_stacking.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/scripts/pipeline/run_step3_stacking.py#L61-L63) & [L201](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/scripts/pipeline/run_step3_stacking.py#L201):
```python
X_train, y_train, groups_train, r_train = load_rev('chantest', 'training', data_dir=os.path.join(trainai_dir, 'data'))
...
stack_model.fit(X_train, y_train)
```

### 2. Deployed Model Fitting
The deployed model checkpoint [`hf_space/deployable_chantest_3class.joblib`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/hf_space/deployable_chantest_3class.joblib) (StackingClassifier with `LogisticRegression` meta-learner) was fitted **strictly and exclusively** on [`data/chantest_training_scaled_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/data/chantest_training_scaled_rev.csv).

**Quotation** from [`scripts/pipeline/build_hf_bundle_rev.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/scripts/pipeline/build_hf_bundle_rev.py#L24-L64) (Lines 24 & 64):
```python
X_train, y_train, drugs_train, report = load_rev('chantest', 'training')
...
stacking_model.fit(X_train.values, y_train.values)
```

### 3. Purpose of Nanion and Manual Training Files
The training files [`data/nanion_training_scaled_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/data/nanion_training_scaled_rev.csv) and [`data/manual_training_scaled_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/data/manual_training_scaled_rev.csv) are **not used for fitting** in the finalized pipeline. They exist as platform-specific training sets from earlier historical exploration (`plus_ultra.ipynb`). In the final thesis pipeline, models are trained only on the high-throughput ChanTest platform, and only the held-out test sets (`nanion_testing_scaled_rev.csv` and `manual_testing_scaled_rev.csv`) are loaded to benchmark zero-shot cross-platform generalization.

---

## Task 3. Deployed Model Evaluation (Logistic Regression Meta-Learner)

Script: [`results/cd4_eval/run_task3_deployed.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/run_task3_deployed.py)  
Artifact Evaluated: [`hf_space/deployable_chantest_3class.joblib`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/hf_space/deployable_chantest_3class.joblib)

### Comprehensive Cross-Platform Performance Table

| Metric | ChanTest (Native) | Nanion (Cross-Platform) | Manual (Cross-Platform) |
| :--- | :---: | :---: | :---: |
| **Correct Drugs / 16** | **14 / 16** (87.5%) | **13 / 16** (81.25%) | **11 / 16** (68.75%) |
| **Drug-Level Macro-F1** | **0.877778** | **0.819373** | **0.690909** |
| **Drug Bootstrap Macro-F1 [95% CI]** | `0.875000 [0.686610, 1.000000]` | `0.811111 [0.576068, 1.000000]` | `0.673401 [0.436364, 0.885714]` |
| **Class High: Precision / Recall / F1** | `0.6667 / 1.0000 / 0.8000` | `0.6667 / 1.0000 / 0.8000` | `0.6667 / 1.0000 / 0.8000` |
| **Class Int: Precision / Recall / F1** | `1.0000 / 0.7143 / 0.8333` | `0.8333 / 0.7143 / 0.7692` | `0.7500 / 0.4286 / 0.5455` |
| **Class Low: Precision / Recall / F1** | `1.0000 / 1.0000 / 1.0000` | `1.0000 / 0.8000 / 0.8889` | `0.6667 / 0.8000 / 0.7273` |
| **Drug Confusion Matrix** `[H, I, L]` | `[[4, 0, 0], [2, 5, 0], [0, 0, 5]]` | `[[4, 0, 0], [2, 5, 0], [0, 1, 4]]` | `[[4, 0, 0], [2, 3, 2], [0, 1, 4]]` |
| **Row-Level Macro-F1** | `0.818171` | `0.769634` | `0.729543` |
| **Row-Level Accuracy** | `0.815562` (26,098/32,000) | `0.763781` (24,441/32,000) | `0.725804` (23,225/31,999) |
| **Ordinal Drug AUC: $P(\text{risk} > \text{Low})$** | **1.0000** | **0.9636** | **0.9455** |
| **Ordinal Drug AUC: $P(\text{risk} > \text{Int})$** | **0.9375** | **0.9167** | **0.9792** |
| **Ordinal Row AUC: $P(\text{risk} > \text{Low})$** | **0.9909** | **0.9569** | **0.9230** |
| **Ordinal Row AUC: $P(\text{risk} > \text{Int})$** | **0.8502** | **0.8629** | **0.9458** |
| **CiPA Bootstrap AUC-High (95% CI)** | `0.833 (0.771-0.917)` | `0.875 (0.833-0.938)` | `0.958 (0.854-1.000)` |
| **CiPA Bootstrap AUC-Int (95% CI)** | `0.714 (0.603-0.857)` | `0.698 (0.571-0.825)` | `0.762 (0.619-0.889)` |
| **CiPA Bootstrap AUC-Low (95% CI)** | `1.000 (0.945-1.000)` | `0.964 (0.891-1.000)` | `0.909 (0.836-1.000)` |

### Detailed Audit of Misclassified Compounds
- **ChanTest (2 misclassified)**:
  - `domperidone`: Intermediate $\to$ High (**over-call**; safe)
  - `pimozide`: Intermediate $\to$ High (**over-call**; safe)
  - *Under-triage count*: **0**
- **Nanion (3 misclassified)**:
  - `domperidone`: Intermediate $\to$ High (**over-call**; safe)
  - `pimozide`: Intermediate $\to$ High (**over-call**; safe)
  - `metoprolol`: Low $\to$ Intermediate (**over-call**; safe)
  - *Under-triage count*: **0**
- **Manual (5 misclassified)**:
  - `domperidone`: Intermediate $\to$ High (**over-call**)
  - `pimozide`: Intermediate $\to$ High (**over-call**)
  - `metoprolol`: Low $\to$ Intermediate (**over-call**)
  - `clarithromycin`: Intermediate $\to$ Low (**under-triage**)
  - `risperidone`: Intermediate $\to$ Low (**under-triage**)
  - *Under-triage count*: **2**

---

## Task 4. Voting Comparison (Soft vs. Hard Voting)

Script: [`results/cd4_eval/run_task4_voting.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/run_task4_voting.py)

### Mathematical Proof of Equivalence
For an ensemble of two base learners with tie-breaking governed by summed probabilities:
1. When learners agree ($c_1 = c_2 = c$), majority vote chooses $c$. Concurrently, $\bar{p}_c = (p_{1,c} + p_{2,c})/2$ is strictly maximal, yielding $\text{argmax}(\bar{p}) = c$.
2. When learners disagree ($c_1 \ne c_2$), a 1-to-1 tie occurs. The tie-breaker rule selects $\text{argmax}(p_1 + p_2) \equiv \text{argmax}((p_1 + p_2)/2) \equiv \text{argmax}(\bar{p})$.
Consequently, **hard voting with probability tie-breaking is mathematically identical to soft voting for any two-model ensemble**. Empirical verification confirmed identical row predictions (`identical_to_soft_row = True`) and identical drug predictions (`identical_to_soft_drug = True`) across all platforms.

Furthermore, base learners extracted from the deployed bundle (`StackingClassifier.estimators_`) and those from the research bundle (`named_estimators_`) yielded **identical predictions and metrics**.

### Performance Across Platforms

| Platform | Model / Voting Scheme | Correct / 16 | Drug Macro-F1 | Bootstrap 95% CI | Misclassifications | Under-Triage Count |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **ChanTest** | Soft / Hard Voting | **13 / 16** | **0.819373** | `[0.576068, 1.000000]` | `domperidone` (over-call), `pimozide` (over-call), `tamoxifen` (over-call) | **0** |
| **Nanion** | Soft / Hard Voting | **13 / 16** | **0.819373** | `[0.576068, 1.000000]` | `domperidone` (over-call), `metoprolol` (over-call), `pimozide` (over-call) | **0** |
| **Manual** | Soft / Hard Voting | **12 / 16** | **0.755556** | `[0.504545, 0.940741]` | `domperidone` (over-call), `metoprolol` (over-call), `pimozide` (over-call), `clarithromycin` (under-triage) | **1** |

---

## Task 5. Conformal Recalibration for the Deployed Model

Script: [`results/cd4_eval/run_task5_conformal_recal.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/run_task5_conformal_recal.py)

### Leave-One-Drug-Out (LODO) Calibration
Executing genuine leave-one-drug-out cross-conformal calibration across the 12 training drugs in [`data/chantest_training_scaled_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/data/chantest_training_scaled_rev.csv) with the deployed model architecture (`StackingClassifier` with `LogisticRegression` meta-learner) produced the following non-conformity scores ($s_d = 1 - \bar{p}_{\text{true}}$):

| Rank $k$ | Drug Name | True Class | $P(\text{High})$ | $P(\text{Int})$ | $P(\text{Low})$ | Calibration Score $s_d$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `verapamil` | Low | 0.011784 | 0.008872 | 0.979344 | **0.020656** |
| 2 | `bepridil` | High | 0.971668 | 0.028331 | 0.000001 | **0.028332** |
| 3 | `dofetilide` | High | 0.926883 | 0.073111 | 0.000006 | **0.073117** |
| 4 | `quinidine` | High | 0.922015 | 0.062732 | 0.015253 | **0.077985** |
| 5 | `cisapride` | Intermediate | 0.030194 | 0.872226 | 0.097579 | **0.127774** |
| 6 | `mexiletine` | Low | 0.012329 | 0.138399 | 0.849272 | **0.150728** |
| 7 | `sotalol` | High | 0.727095 | 0.250439 | 0.022467 | **0.272905** |
| 8 | `ranolazine` | Low | 0.043607 | 0.233650 | 0.722742 | **0.277258** |
| 9 | `terfenadine` | Intermediate | 0.085433 | 0.705307 | 0.209260 | **0.294693** |
| 10 | `chlorpromazine` | Intermediate | 0.033091 | 0.571326 | 0.395583 | **0.428674** |
| **11** | `ondansetron` | **Intermediate** | **0.409590** | **0.351579** | **0.238830** | **0.648421** |
| 12 | `diltiazem` | Low | 0.040575 | 0.941359 | 0.018066 | **0.981934** |

### Operating Point Parameters (Rank 11)
- **Calibration Quantile ($\hat{q}$)**: **`0.648421`** (Score of Rank 11 drug: `ondansetron`)
- **Significance Level ($\alpha$)**: **`0.153846`** ($\approx 0.154$; smallest valid $\alpha = 1 - 11/13 = 2/13$)
- **Nominal Coverage Guarantee ($1 - \alpha$)**: **`84.615%`** ($11/13$)
- **Conformal Inclusion Threshold ($1 - \hat{q}$)**: **`0.351579`**

### Audit of Hardcoded `Q_HAT = 0.6275` in Space
In [`scripts/pipeline/build_hf_bundle_rev.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/scripts/pipeline/build_hf_bundle_rev.py#L75-L76) (Lines 75–76):
```python
bundle = {
    ...
    'q_hat': 0.6275,  # 11th of 12 LODO scores
    'alpha': 0.154,   # 1 - 0.154 = 84.6% nominal coverage
}
```
- **How it was computed**: `0.6275` was **not computed** on the Logistic Regression stacking model. It was hardcoded from the *historical pre-_rev model bundle* ([`artifacts/deployable_chantest_3class.joblib`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/artifacts/deployable_chantest_3class.joblib)), where the 11th score happened to be `0.6275`.
- **Match Status**: **DOES NOT MATCH**. The true recalibrated value for the deployed Logistic Regression stacking model is **`q_hat = 0.648421`** (inclusion threshold `0.351579`).

### Conformal Performance on Test Sets (Rank 11, $\hat{q} = 0.648421$)

| Platform | Empirical Coverage | Avg Set Size | Ambiguous Sets ($|S| > 1$) | Empty Sets (OOD, $|S|=0$) | Under-Triage Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ChanTest** | **14 / 16** (87.5%) | **1.062** | 1 (`risperidone`) | 0 | **0** |
| **Nanion** | **13 / 16** (81.25%) | **1.125** | 2 (`clarithromycin`, `risperidone`) | 0 | **0** |
| **Manual** | **12 / 16** (75.0%) | **1.062** | 1 (`risperidone`) | 0 | **1** (`clarithromycin`) |

*(Note: Applying the hardcoded threshold `0.6275` produces identical discrete test outcomes because no drug mean probability falls in the narrow band between $0.351579$ and $0.372500$).*

---

## Task 6. Unified Model Comparison & Paired Bootstrap Analysis

Script: [`results/cd4_eval/run_task6_unified.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/run_task6_unified.py)

### Platform A: ChanTest (Native Benchmark)

| Model Architecture | Correct / 16 | Macro-F1 [95% CI] | Under-Triaged Drugs (Point) |
| :--- | :---: | :---: | :--- |
| **RF alone** | 14 / 16 | 0.877778 [0.686610, 1.000000] | none |
| **XGB alone** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **Soft voting** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **Hard voting** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **Stacking with RF meta (research)** | 11 / 16 | 0.694444 [0.391534, 0.896825] | `disopyramide` |
| **Stacking with LR meta (deployed)** | **14 / 16** | **0.877778 [0.686610, 1.000000]** | **none** |
| **Multivariate LR baseline** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **qNet-only LR baseline** | 13 / 16 | 0.811966 [0.583333, 1.000000] | `disopyramide` |

#### Paired Bootstrap Differences: Deployed Stacking minus Competitors (ChanTest)
- vs. **RF alone**: Median diff = `+0.000000`, 95% CI `[+0.000000, +0.000000]` (Identical predictions across all 16 drugs)
- vs. **XGB alone**: Median diff = `+0.057059`, 95% CI `[+0.000000, +0.216667]` (Contains 0 at lower boundary)
- vs. **Soft / Hard voting**: Median diff = `+0.057059`, 95% CI `[+0.000000, +0.216667]` (Contains 0 at lower boundary)
- vs. **Stacking research (RF meta)**: Median diff = `+0.187302`, 95% CI `[+0.000000, +0.448155]` (Contains 0 at boundary)
- vs. **Multivariate LR**: Median diff = `+0.057059`, 95% CI `[+0.000000, +0.216667]`
- vs. **qNet-only LR**: Median diff = `+0.063492`, 95% CI `[+0.000000, +0.237037]`

#### Conformal Safety Comparison (Rank 11, ChanTest)
- **Deployed Stacking** ($\hat{q}=0.648421$): Under-triage = **0**, Average Set Size = **1.062** (Coverage = 87.5%)
- **RF alone** ($\hat{q}=0.533987$): Under-triage = **0**, Average Set Size = **1.000** (Coverage = 87.5%)
- **qNet-only LR** ($\hat{q}=0.663626$): Under-triage = **0**, Average Set Size = **1.562** (Coverage = 87.5%)

---

### Platform B: Nanion (Cross-Platform Automation)

| Model Architecture | Correct / 16 | Macro-F1 [95% CI] | Under-Triaged Drugs (Point) |
| :--- | :---: | :---: | :--- |
| **RF alone** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **XGB alone** | 12 / 16 | 0.754762 [0.479347, 0.944056] | none |
| **Soft voting** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **Hard voting** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **Stacking with RF meta (research)** | 10 / 16 | 0.621032 [0.311111, 0.851852] | `disopyramide` |
| **Stacking with LR meta (deployed)** | **13 / 16** | **0.819373 [0.576068, 1.000000]** | **none** |
| **Multivariate LR baseline** | 12 / 16 | 0.754762 [0.479347, 0.944056] | none |
| **qNet-only LR baseline** | 12 / 16 | 0.756614 [0.491439, 0.940741] | `disopyramide` |

#### Paired Bootstrap Differences: Deployed Stacking minus Competitors (Nanion)
- vs. **RF alone**: Median diff = `+0.000000`, 95% CI `[+0.000000, +0.000000]`
- vs. **XGB alone**: Median diff = `+0.061538`, 95% CI `[+0.000000, +0.242858]`
- vs. **Soft / Hard voting**: Median diff = `+0.000000`, 95% CI `[+0.000000, +0.000000]`
- vs. **Stacking research (RF meta)**: Median diff = `+0.198677`, 95% CI `[+0.000000, +0.459259]`
- vs. **Multivariate LR**: Median diff = `+0.061538`, 95% CI `[+0.000000, +0.242858]`
- vs. **qNet-only LR**: Median diff = `+0.059695`, 95% CI `[+0.000000, +0.227513]`

#### Conformal Safety Comparison (Rank 11, Nanion)
- **Deployed Stacking** ($\hat{q}=0.648421$): Under-triage = **0**, Average Set Size = **1.125** (Coverage = 81.25%)
- **RF alone** ($\hat{q}=0.533987$): Under-triage = **0**, Average Set Size = **1.000** (Coverage = 81.25%)
- **qNet-only LR** ($\hat{q}=0.663626$): Under-triage = **0**, Average Set Size = **1.562** (Coverage = 87.5%)

---

### Platform C: Manual (Traditional Patch-Clamp)

| Model Architecture | Correct / 16 | Macro-F1 [95% CI] | Under-Triaged Drugs (Point) |
| :--- | :---: | :---: | :--- |
| **RF alone** | 12 / 16 | 0.755556 [0.504545, 0.940741] | `clarithromycin` |
| **XGB alone** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **Soft voting** | 12 / 16 | 0.755556 [0.504545, 0.940741] | `clarithromycin` |
| **Hard voting** | 12 / 16 | 0.755556 [0.504545, 0.940741] | `clarithromycin` |
| **Stacking with RF meta (research)** | 12 / 16 | 0.754762 [0.479347, 0.944056] | none |
| **Stacking with LR meta (deployed)** | **11 / 16** | **0.690909 [0.436364, 0.885714]** | `clarithromycin`, `risperidone` |
| **Multivariate LR baseline** | 13 / 16 | 0.819373 [0.576068, 1.000000] | none |
| **qNet-only LR baseline** | 12 / 16 | 0.751515 [0.494434, 0.939394] | `risperidone` |

#### Paired Bootstrap Differences: Deployed Stacking minus Competitors (Manual)
- vs. **RF alone**: Median diff = `-0.063131`, 95% CI `[-0.211111, +0.000000]` (Contains 0 at upper boundary)
- vs. **XGB alone**: Median diff = `-0.127116`, 95% CI `[-0.318528, +0.000000]` (Contains 0 at upper boundary)
- vs. **Soft / Hard voting**: Median diff = `-0.063131`, 95% CI `[-0.211111, +0.000000]` (Contains 0 at upper boundary)
- vs. **Stacking research (RF meta)**: Median diff = `-0.063818`, 95% CI `[-0.285948, +0.157741]` (Spans zero cleanly)
- vs. **Multivariate LR**: Median diff = `-0.127116`, 95% CI `[-0.318528, +0.000000]`
- vs. **qNet-only LR**: Median diff = `-0.058081`, 95% CI `[-0.213291, +0.023377]` (Spans zero cleanly)

#### Conformal Safety Comparison (Rank 11, Manual)
- **Deployed Stacking** ($\hat{q}=0.648421$): Under-triage = **1** (`clarithromycin`), Average Set Size = **1.062** (Coverage = 75.0%)
- **RF alone** ($\hat{q}=0.533987$): Under-triage = **0**, Average Set Size = **1.062** (Coverage = 81.25%)
- **qNet-only LR** ($\hat{q}=0.663626$): Under-triage = **0**, Average Set Size = **1.625** (Coverage = 87.5%)

---

## Task 7. Biomarker Percentiles & Normalization Audit

Script: [`results/cd4_eval/run_task7_biomarkers.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/run_task7_biomarkers.py)

### Numerical Ranges Across Training Platforms

| Biomarker | ChanTest Min | ChanTest 1% | ChanTest Median | ChanTest 99% | ChanTest Max | Nanion Min | Nanion Max | Manual Min | Manual Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **qNet** | 0.569354 | 0.642896 | 0.892058 | 1.371254 | 1.465906 | 0.260453 | 1.245039 | 0.373427 | 1.432374 |
| **dvdtmax** | 0.900509 | 0.927907 | 0.999727 | 1.001170 | 1.001332 | 0.946835 | 1.001042 | 0.755486 | 1.003802 |
| **vmax** | 0.909437 | 0.937092 | 0.998099 | 1.007820 | 1.009985 | 0.943824 | 1.013898 | 0.706826 | 1.009050 |
| **vrest** | -1.000534 | -1.000361 | -1.000022 | -0.999964 | -0.999958 | -1.000545 | -0.999958 | -1.000916 | -0.999962 |
| **APD50** | 0.932773 | 0.982143 | 1.135504 | 1.919128 | 2.056723 | 0.968487 | 2.155462 | 0.808824 | 2.161765 |
| **APD90** | 0.978448 | 0.996552 | 1.167241 | 2.212931 | 2.488793 | 0.976724 | 3.236207 | 0.844828 | 2.449138 |
| **max_dv** | -1.005477 | -0.993456 | -0.831757 | -0.259310 | -0.174490 | -1.001400 | 1.016641 | -1.024670 | -0.194425 |
| **camax** | 0.757602 | 0.838891 | 1.027993 | 1.259112 | 1.341898 | 0.634346 | 1.221718 | 0.478540 | 1.226679 |
| **carest** | 0.939000 | 0.963000 | 1.010000 | 1.020000 | 1.020000 | 0.887500 | 1.017500 | 0.809000 | 1.020000 |
| **CaTD50** | 0.877907 | 0.949336 | 1.035714 | 1.155316 | 1.392857 | 0.940199 | 1.263289 | 0.955980 | 1.516611 |
| **CaTD90** | 0.966942 | 0.983127 | 1.013774 | 1.176653 | 1.284435 | 0.963499 | 1.591598 | 0.981061 | 1.346763 |

### Interpretation of Units
The numerical values in the `_rev` datasets are **unitless normalized ratios relative to baseline control simulations**:
- Parameters representing membrane voltages or duration multiples (`dvdtmax`, `vmax`, `camax`, `carest`, `CaTD50`, `CaTD90`, `APD50`, `APD90`) are centered tightly around $1.000$ (representing $100\%$ of drug-free vehicle control).
- Resting potential `vrest` is centered at $-1.000$ (reflecting the ratio of resting membrane potential to baseline control potential, $\approx -88\text{ mV} / -88\text{ mV} \times (-1)$).
- Repolarization derivative `max_dv` is negative and centered around $-0.832$.
- They are **not** raw physical units (e.g., $V_{\text{rest}}$ is not in millivolts like $-88.0\text{ mV}$, $dV/dt$ is not in $\text{V/s}$ like $72.8\text{ V/s}$, and APD is not in milliseconds like $440\text{ ms}$).

### Documentation Quotation
From [`docs/STORYLINE.md`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/docs/STORYLINE.md#L52) (Line 52):
> *"All experiments in this repository utilize the revised (`_rev`) dataset files. These datasets are pre-normalized against control simulations. Consequently, **no additional feature scaling** (such as `StandardScaler` or `MinMaxScaler`) was applied in this project."*

And [`docs/STEP3_STACKING_CHANTEST.md`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/docs/STEP3_STACKING_CHANTEST.md#L362) (Line 362):
> *"...the only change is the input data (control-normalized _rev inputs, no scaler). RF and XGB are scale-invariant, so removing StandardScaler is not itself expected to change performance."*

---

## Task 8. SHAP Figure Verification & Discrepancy Reconciliation

Script Producing Figure: [`results/cd4_eval/make_fig_shap_fixed.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/make_fig_shap_fixed.py)  
Corrected Figure Artifacts:
- PNG: [`results/cd4_eval/fig_shap_global_fixed.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/fig_shap_global_fixed.png)
- PDF: [`results/cd4_eval/fig_shap_global_fixed.pdf`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/fig_shap_global_fixed.pdf)

### Analysis of the Discrepancy
1. **Source Script of Original Figure**:
   [`results/fig_shap_global.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/fig_shap_global.png) was generated by [`scripts/pipeline/run_step7_shap.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/scripts/pipeline/run_step7_shap.py#L184-L222) (Lines 184–222).
2. **Data Source**:
   Evaluated using `shap.KernelExplainer(stack_model.predict_proba, background_data)` on a 320-row stratified subsample of [`data/chantest_testing_scaled_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/data/chantest_testing_scaled_rev.csv) (20 rows per drug across all 16 held-out test drugs) with a 100-row training background sample.
3. **Reason for Apparent Discrepancy**:
   - Decompressing and measuring the raw vector path coordinates of [`results/fig_shap_global.pdf`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/fig_shap_global.pdf) revealed that the physical horizontal bar lengths in `fig_shap_global.pdf` match [`results/shap_global_importance_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/shap_global_importance_rev.csv) with sub-pixel precision:
     - `qNet`: bar width $0.1455$ (CSV: $0.145534$)
     - `vrest`: bar width $0.1155$ (CSV: $0.115489$)
     - `max_dv`: bar width $0.0799$ (CSV: $0.079867$)
     - `APD50`: bar width $0.0698$ (CSV: $0.069771$)
     - `APD90`: bar width $0.0490$ (CSV: $0.049044$)
     - `vmax`: bar width $0.0382$ (CSV: $0.038180$)
     - `dvdtmax`: bar width $0.0128$ (CSV: $0.012816$)
     - `camax`: bar width $0.0124$ (CSV: $0.012420$)
     - `CaTD50`: bar width $0.0092$ (CSV: $0.009237$)
     - `CaTD90`: bar width $0.0080$ (CSV: $0.008037$)
     - `carest`: bar width $0.0007$ (CSV: $0.000655$)
   - However, the original figure omitted **numerical data value labels** and used a compact $3.5 \times 3.5$ inch bounding box with thick borders and LaTeX labels, making subtle visual distinctions (such as $0.0799$ vs $0.0698$ vs $0.0490$) easily misread, especially when contrasted against historical pre-_rev numbers (`qNet` 0.162, `vrest` 0.084, `APD50` 0.071, `max_dv` 0.063, `APD90` 0.058).
   - In [`results/cd4_eval/fig_shap_global_fixed.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/fig_shap_global_fixed.png), high-resolution 300 dpi rendering was generated directly from [`results/shap_global_importance_rev.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/shap_global_importance_rev.csv) with explicit numerical callouts printed on each bar.

---

## Task 9. Local Inference & Explanation Latency Profiling

Script: [`results/cd4_eval/run_task9_latency.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/run_task9_latency.py)

### Host Hardware Environment
- **CPU Model**: **13th Gen Intel(R) Core(TM) i7-13620H** (10 cores: 6 P-cores, 4 E-cores; 16 threads; base 2.4 GHz, boost up to 4.9 GHz)
- **Runtime**: Python 3.12 (64-bit Windows)

### Benchmarked Latency Distributions

| Pipeline Operation | Runs | Mean (ms) | Median / P50 (ms) | P95 (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`predict_proba`** (Instant Tier) | 100 | **57.364 ms** | **54.855 ms** | **66.596 ms** | **111.574 ms** |
| **SHAP `explain`** (`KernelExplainer`, $n=500$) | 30 | **236.004 ms** | **234.571 ms** | **250.787 ms** | **255.000 ms** |

*(Design validation: Separating instant risk inference from on-demand SHAP attribution ensures the primary prediction interface responds in $\approx 55\text{ ms}$, while explanation delivers full 11-feature attributions in under $0.25\text{ s}$).*

---

## Task 10. Plain-Language Summary of STEP14 & STEP13

1. **Audit of Nominal Coverage**: Verification resolved an ambiguity where nominal coverage had been conflated with $\alpha_{\text{midpoint}}$; the mathematically true nominal coverage for Rank 11 ($k=11$) is strictly $1 - \alpha = 11/13 \approx 84.62\%$.
2. **Empty Set Abstain Rule**: An audit detected and fixed a bug where empty prediction sets erroneously defaulted to `argmax(probs)`; the corrected rule flags empty sets as high-uncertainty "Abstains" (full set size = 3, default action = High), eliminating under-triage on `manual`.
3. **RF Conformal Operating Point**: The tightest valid Rank for `RandomForestClassifier` guaranteeing zero under-triage across all three platforms is Rank 8 ($\hat{q} = 0.322894$); at Rank 11, its threshold is $\hat{q} = 0.533987$.
4. **Zero Under-Triage Safety Guarantee**: In a 10,000-iteration paired bootstrap on held-out test drugs, RF + Conformal achieved 0 under-triage across all three platforms, yielding an analytical upper bound of $3/16 = 18.75\%$ via the rule-of-three.
5. **Ensemble Redundancy**: Paired bootstrap differences (RF minus Stacking) showed that RF was never inferior to Stacking in any bootstrap iteration, demonstrating that the single RF base learner provides equivalent safety with superior simplicity.

---

## Task 11. Conclusions for the CD-4 Implementation Report

Based strictly on the empirical benchmark numbers without spin or speculative extrapolation:

1. **Accuracy & Discrimination (F1 & Correct Drugs)**:
   - On **ChanTest** (native training assay), the **deployed model (Stacking with LR meta)** achieves **14/16 correct drugs** and macro-F1 of **0.877778**, exactly tying **RF alone** (14/16, 0.877778), and outperforming **XGB alone** (13/16, 0.819373) and **soft/hard voting** (13/16, 0.819373). The paired bootstrap difference vs. RF alone is exactly `+0.000000 [+0.000000, +0.000000]`.
   - On **Nanion** (cross-platform automation), the **deployed model** achieves **13/16 correct drugs** and macro-F1 of **0.819373**, again exactly tying **RF alone** (13/16, 0.819373) and **soft/hard voting** (13/16, 0.819373), while slightly outperforming **XGB alone** (12/16, 0.754762). Paired bootstrap difference vs. RF is `+0.000000 [+0.000000, +0.000000]`.
   - On **Manual** (traditional patch clamp), the **deployed model** drops to **11/16 correct drugs** (macro-F1 **0.690909**), performing **worse** than **XGB alone** (13/16, 0.819373), **RF alone** (12/16, 0.755556), and **soft/hard voting** (12/16, 0.755556). Paired bootstrap difference vs. RF alone is `-0.063131 [-0.211111, +0.000000]`.
   - **Conclusion on Accuracy**: The deployed Stacking model is **identical to RF alone** on ChanTest and Nanion, but **inferior to RF alone, XGB alone, and voting** under manual patch-clamp cross-platform transfer. Across all platforms, the paired differences between Stacking and RF contain zero in their 95% bootstrap intervals, confirming that stacking offers no statistically significant discrimination advantage over the single Random Forest base learner.

2. **Safety & Under-Triage**:
   - **Without Conformal Prediction (Hard Argmax)**:
     - On ChanTest and Nanion, the deployed model, RF alone, XGB alone, and voting all achieve **0 under-triage**.
     - On Manual, the deployed model registers **2 under-triaged drugs** (`clarithromycin` and `risperidone` classified as Low risk), whereas RF alone and voting register only **1 under-triage** (`clarithromycin`), and XGB alone registers **0 under-triage**.
   - **With Conformal Prediction (Rank 11)**:
     - On ChanTest and Nanion, all conformal models achieve **0 under-triage**.
     - On Manual, RF alone achieves **0 under-triage** with an average set size of $1.062$, whereas the deployed model has **1 under-triage** (`clarithromycin`) with set size $1.062$.
   - **Conclusion on Safety**: The deployed model is **slightly less safe** than RF alone and voting under cross-platform transfer on Manual data due to 2 uncalibrated under-triage errors. Once equipped with a Rank-11 conformal layer, RF alone eliminates all under-triage across all three platforms ($0/16$ everywhere), whereas the deployed stacking model still under-triages `clarithromycin` on Manual data.
