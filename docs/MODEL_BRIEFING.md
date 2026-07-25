> [WARNING] STALE - OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

> ⚠️ STALE — OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

# MODEL_BRIEFING.md — Technical Model & Evaluation Reference

This consolidated technical briefing documents the exact hyperparameter tuning spaces, model sweep benchmarks, bootstrap confidence intervals, split-conformal prediction methodology, and deployed web app API realities for the CiPA Cardiotoxicity (TdP Risk) classifier codebase (`TrainAI`).

---

## 1. Training & Tuning (from `plus_ultra.ipynb`)

### Preprocessing & Missing Value Handling
* **Scaler:** `StandardScaler` fit strictly on training splits per dataset, then applied to transform train and test splits separately. No test set information is leaked during scaling.
* **Missing Value (`NaN`) Cleaning:** 
  * `chantest_train`: 25 rows dropped (remaining shape: 23,975 $\times$ 11).
  * `chantest_test`: 1 row dropped (remaining shape: 31,999 $\times$ 11).
* **Additional Preprocessing:** **None** beyond `dropna()` on the 11 biomarker features and `StandardScaler` transformation.

### Grouped Cross-Validation Setup
* **CV Scheme:** `StratifiedGroupKFold(n_splits=4)`
* **Grouping Key:** `drug_name` (12 unique training drugs in `chantest`, exactly 3 unique drugs per fold).
* **Scoring Metric:** `f1_macro` (`scoring='f1_macro'`)
* **Random Seed:** `SEED = 42` (enforced via `random.seed(42)`, `np.random.seed(42)`, `tf.random.set_seed(42)`).

### Hyperparameter Search Space & Final Chosen Parameters
Tuned via `GridSearchCV` on `chantest` 3-class using `StratifiedGroupKFold(4)`:

| Learner | Hyperparameter Search Space | Final Chosen Parameters (`chantest` 3-Class) | Grouped-CV Macro-F1 |
| :--- | :--- | :--- | :--- |
| **Random Forest (`RF`)** | `criterion`: `['gini', 'entropy']`<br>`n_estimators`: `[200, 400]`<br>`max_depth`: `[None, 20]` | `criterion='gini'`<br>`max_depth=None`<br>`n_estimators=400` | **0.794673** |
| **XGBoost (`XGB`)** | `learning_rate`: `[0.001, 0.1]`<br>`max_depth`: `[6, 10]`<br>`n_estimators`: `[300, 500]` | `learning_rate=0.1`<br>`max_depth=6`<br>`n_estimators=300` | **0.742924** |
| **Logistic Regression (`LR`)** | `C`: `[0.01, 1.0]`<br>`penalty`: `['l2']`<br>`solver`: `['lbfgs']`<br>`max_iter`: 2000 | `C=1.0`<br>`penalty='l2'`<br>`solver='lbfgs'` | **0.665812** |
| **Support Vector Machine (`SVM`)** | `C`: `[1.0, 10.0]`<br>`gamma`: `['scale', 'auto']`<br>`probability`: `True` | `C=10.0`<br>`gamma='scale'` | **0.728721** |
| **`ANN1`** (1 Hidden Layer) | `KerasClassifier` (Adam, sparse categorical crossentropy, epochs=30, batch_size=64)<br>`model__nodes`: `[32, 64]`<br>`model__lr`: `[0.0001, 0.01]` | `model__lr=0.0001`<br>`model__nodes=32` | **0.760914** |
| **`ANN2`** (2 Hidden Layers) | `KerasClassifier` (Adam, sparse categorical crossentropy, epochs=30, batch_size=64)<br>`model__nodes`: `[64, 128]`<br>`model__lr`: `[0.01]` | `model__lr=0.01`<br>`model__nodes=128` | **0.725664** |
| **`ANN3`** (3 Hidden Layers) | `KerasClassifier` (Adam, sparse categorical crossentropy, epochs=30, batch_size=64)<br>`model__nodes`: `[128]`<br>`model__lr`: `[0.01]` | `model__lr=0.01`<br>`model__nodes=128` | **0.775998** |

### Stacking Meta-Learner Configuration
The final deployable model uses a `StackingClassifier`:
* **Base Learners:** `RF` and `XGB` (wrapped via `PreFittedEstimatorWrapper` with `groups=True` metadata routing).
* **Meta-Learner (`final_estimator`):** `RandomForestClassifier(random_state=42)` (wrapped via `FinalEstimatorWrapper`).
* **Cross-Validation (`cv`):** `StratifiedGroupKFold(n_splits=4)` with `cv.set_split_request(groups=True)` grouped by `drug_name`.
* **Stack Method:** `stack_method='predict_proba'`
* **Passthrough:** `passthrough=False`

---

## 2. The 612-Config Sweep (from `phase3_protocolA_results.csv`)

### Sweep Variation Parameters
The sweep comprises 612 total model configurations ($3 \times 2 \times 17 \times 6 = 612$):
1. **3 Datasets:** `chantest`, `nanion`, `manual`
2. **2 Scenarios:** `3class` (High/Int/Low), `binary` (High vs Low)
3. **17 Base Estimator Combinations:**
   * **Pure ML (11 combos):** `RF+XGB`, `RF+LR`, `RF+SVM`, `XGB+LR`, `XGB+SVM`, `LR+SVM`, `RF+XGB+LR`, `RF+XGB+SVM`, `RF+LR+SVM`, `XGB+LR+SVM`, `RF+XGB+LR+SVM`
   * **Pure DL (4 combos):** `ANN1+ANN2`, `ANN1+ANN3`, `ANN2+ANN3`, `ANN1+ANN2+ANN3`
   * **Mixed ML+DL (2 combos):** `mixed-all-7` (all 7 models), `paper-faithful 3-ANN`
4. **6 Ensemble Configurations:**
   * **Voting (2):** `voting` (hard), `voting` (soft)
   * **Stacking (4):** `stacking` (`meta=LR`), `stacking` (`meta=SVC`), `stacking` (`meta=RF`), `stacking` (`meta=XGB`)

### Top 5 Configurations by Macro-F1 (`chantest` 3-Class Target)

| Rank | Combo | Ensemble Family | Detail | Paradigm | Test Macro-F1 | Accuracy |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **1** | `RF+XGB` | `stacking` | `meta=RF` | ML | **0.828004** | **0.824187** |
| **2** | `RF+XGB+LR` | `stacking` | `meta=RF` | ML | **0.827086** | **0.823406** |
| **3** | `RF+XGB+SVM` | `stacking` | `meta=RF` | ML | **0.826735** | **0.823063** |
| **4** | `RF+LR` | `stacking` | `meta=RF` | ML | **0.825332** | **0.821844** |
| **5** | `RF+LR+SVM` | `stacking` | `meta=RF` | ML | **0.825200** | **0.821844** |

*(Top 5 overall across ALL 612 rows including binary scenario: `chantest binary voting hard RF+XGB` = 0.946543, `chantest binary stacking meta=RF RF+XGB+LR` = 0.946078, `chantest binary stacking meta=LR RF+LR` = 0.945689, `chantest binary stacking meta=RF RF+XGB+SVM` = 0.945165, `chantest binary stacking meta=SVC RF+SVM` = 0.944718).*

### Mixed ML+DL Best vs. Chosen Pure-ML `RF+XGB` (Exact Fact Check)
* **Chosen Pure-ML (`RF+XGB` Stacking `meta=RF`):** `test_macro_f1` = **0.828004** (accuracy = **0.824187**).
* **Best Mixed ML+DL (`mixed-all-7` Stacking `meta=RF`):** `test_macro_f1` = **0.816995** (accuracy = **0.811187**).
* **Fact Check on ~0.842:** A score of ~0.842 for mixed ML+DL is **NOT FOUND** in `phase3_protocolA_results.csv` for `chantest 3class`. The leak-free Protocol A CSV records **0.816995** for `mixed-all-7` Stacking `meta=RF`. As documented in `PROJECT_SUMMARY.md` Section 5, scores near ~0.842 occurred only in earlier leaky validation scenarios; under the rigorous leak-free Protocol A, the pure-ML model strictly outperformed the mixed model.

---

## 3. Bootstrap (from `phase4_bootstrap_summary.csv`)

### Resampling Protocol
* **Resampling Level:** **Drug-level resampling** (1 row randomly sampled per drug from its 2,000 test rows, yielding exactly 16 rows per iteration across the 16 test drugs).
* **Number of Iterations:** **10,000** iterations.
* **Model State:** Frozen deployable model (`chantest` 3-class).

### Tabular Summary (Median [95% Confidence Interval])

| Dataset | Macro-F1 Median [95% CI] | AUC High [95% CI] | AUC Intermediate [95% CI] | AUC Low [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
| **`chantest`** | **0.819373** [0.694017, 0.878307] | **0.916667** [0.875000, 0.958333] | **0.825397** [0.714286, 0.912698] | **1.000000** [0.836364, 1.000000] |
| **`nanion`** | **0.679365** [0.623377, 0.754762] | **0.916667** [0.875000, 0.958333] | **0.658730** [0.571429, 0.746031] | **0.700000** [0.645455, 0.800000] |
| **`manual`** | **0.750000** [0.599868, 0.822222] | **0.947917** [0.875000, 1.000000] | **0.809523** [0.698413, 0.936508] | **0.872727** [0.745455, 0.963636] |

*(Per-class recall medians: `chantest` High 1.000000 [0.750, 1.000], Int 0.714286 [0.429, 0.857], Low 1.000000 [0.600, 1.000]; `nanion` High 1.000000 [1.000, 1.000], Int 0.714286 [0.571, 0.857], Low 0.400000 [0.400, 0.600]; `manual` High 1.000000 [0.750, 1.000], Int 0.714286 [0.571, 1.000], Low 0.400000 [0.200, 0.600]).*

---

## 4. Conformal Prediction (from `conformal_*.py` files)

### The Method & Calibration Parameters
* **Framework:** Drug-level split-conformal prediction.
* **Target Significance Level ($\alpha$):** **0.154** ($2/13$, corresponding to $\frac{2}{n+1}$ with $n=12$ calibration drugs).
* **Conformal Quantile Threshold ($\hat{q}$):** **0.6275** ($\lceil (n+1)(1-\alpha) \rceil = \lceil 13 \times 0.846 \rceil = 11$-th smallest score out of 12).
* **Nonconformity Score Definition:** 
  $$s = 1 - p_{\text{mean}}[\text{true\_class}]$$
  where $p_{\text{mean}}$ is the drug's mean predicted probability vector averaged over its 2,000 simulated samples.
* **Prediction Set Inclusion Criterion:** A class $c$ is included in the prediction set iff:
  $$1 - p_{\text{mean}}[c] \le \hat{q} \iff p_{\text{mean}}[c] \ge 1 - \hat{q} = 0.3725$$

### Leave-One-Drug-Out (LODO) Calibration (`conformal_diag.py`)
* Evaluated across the 12 `chantest` training drugs.
* For each held-out drug: refit scaler on remaining 11 drugs, refit stacking pipeline (`RF+XGB`, meta `RF`, internal `StratifiedGroupKFold(4)` on 11 drugs), predict on held-out drug, average probabilities over 2,000 samples, and compute nonconformity score $s$.
* **Sorted 12 LODO Nonconformity Scores:**
  `0.0181, 0.0878, 0.1084, 0.1634, 0.1714, 0.2967, 0.3939, 0.4584, 0.5599, 0.6149, 0.6275, 0.9898`
* **Quantile Breakdown:**
  * $\alpha = 0.077 \implies \hat{q} = 0.9898$ (index 12 — **degenerate**, set by single outlier `diltiazem`, producing uninformative full-support sets).
  * $\alpha = 0.154 \implies \hat{q} = 0.6275$ (index 11 — **tightest valid operating point**).
  * $\alpha = 0.231 \implies \hat{q} = 0.6149$ (index 10).

### Ordinal vs. Flat Conformal (`scripts/legacy/conformal_ordinal.py`)
* **Ordinal Nonconformity Score:** $s = \sum_{c=0}^2 p_{\text{mean}}[c] \cdot |\text{severity}(c) - \text{severity}(\text{true\_class})|$ with severity mapping High=2, Int=1, Low=0. Quantile $\hat{q} = 0.7012$ at $\alpha=0.154$.
* **Contiguity Finding:** At the valid operating point $\alpha=0.154$, **flat conformal produced ZERO non-contiguous sets** across all datasets (the non-contiguous `{High, Low}` set for pimozide occurred only under degenerate $\alpha=0.077$).
* **Comparison Result:** Ordinal conformal produced 2 extra over-calls (`tamoxifen` $\to \{Int, Low\}$ on chantest and nanion) with identical coverage and identical zero under-triage. Ordinal conformal was **rejected**; flat conformal is retained as the primary method.

### Clinical Readout Rules & Safety Metrics (`conformal_safety.py`)
* **Conformal Action:** $\max_{\text{severity}}(\text{prediction\_set})$, with severity High > Int > Low.
* **Outcome Classifications:**
  * `correct`: `conformal_action == true_class`
  * `over-called`: `severity(conformal_action) > severity(true_class)`
  * `under-triaged`: `severity(conformal_action) < severity(true_class)` (**sole unsafe clinical outcome**)
  * `abstain`: `prediction_set` is empty ($len = 0$).

#### Headline Table: Hard-Label vs. Conformal ($\alpha=0.154$)

| Dataset | System | Correct | Over-called | Under-triaged | Avg Set Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CHANTEST** | hard-label (argmax) | 14 | 2 | 0 | 1.00 |
| **CHANTEST** | conformal @ $\alpha=0.154$ | 14 | 2 | 0 | 1.00 |
| **NANION** | hard-label (argmax) | 13 | 3 | 0 | 1.00 |
| **NANION** | conformal @ $\alpha=0.154$ | 13 | 3 | 0 | 1.00 |
| **MANUAL** | hard-label (argmax) | 12 | 3 | **1** | 1.00 |
| **MANUAL** | conformal @ $\alpha=0.154$ | 13 | 3 | **0** | 1.0625 |

*Key Clinical Safety Finding:* Conformal prediction eliminates the sole under-triaged critical miss (`clarithromycin` on manual) by expanding its set to `{Int, Low}` (action `Int`) at a marginal cost of $+0.0625$ set size, achieving **zero under-triaged drugs across all datasets**.

### Empirical vs. Nominal Coverage (`conformal_safety.py`)
* **Nominal Target ($1-\alpha$):** **0.8460** (84.6%).
* **Empirical Coverage (16 Test Drugs):**
  * `CHANTEST` (in-distribution / calibration domain): **0.8750** (14/16 drugs covered, diff $+0.0290$).
  * `NANION` (domain-shifted): **0.8125** (13/16 drugs covered, diff $-0.0335$).
  * `MANUAL` (domain-shifted): **0.8125** (13/16 drugs covered, diff $-0.0335$).

### Unit Test Assertions (`test_conformal.py`)
`test_conformal.py` asserts exact behavior for `conformal_from_proba`:
1. `clarithromycin/manual` `{high: 0.0028, intermediate: 0.4734, low: 0.5238}` $\implies$ `prediction_set`: `['intermediate', 'low']`, `recommended_action`: `'intermediate'`, `is_ambiguous`: `True`, `out_of_distribution`: `False`.
2. `domperidone/manual` `{high: 0.9991, intermediate: 0.0000, low: 0.0009}` $\implies$ `prediction_set`: `['high']`, `recommended_action`: `'high'`, `is_ambiguous`: `False`, `out_of_distribution`: `False`.
3. `loratadine/chantest` `{high: 0.0, intermediate: 0.0, low: 1.0}` $\implies$ `prediction_set`: `['low']`, `recommended_action`: `'low'`, `is_ambiguous`: `False`, `out_of_distribution`: `False`.
4. `synthetic empty-set check` `{high: 0.35, intermediate: 0.35, low: 0.30}` $\implies$ `prediction_set`: `[]`, `recommended_action`: `'high'` (argmax fallback), `is_ambiguous`: `True`, `out_of_distribution`: `True`, `warning`: `"No class met the conformal threshold. Result is not covered by the guarantee."`.

---

## 5. Deployed App Reality-Check (from `hf_space/app.py`)

### API Endpoints and Exact Output Structures

The Gradio web application (`hf_space/app.py`) exposes **2 active endpoints**:

#### Endpoint 1: `/predict` (`predict_btn.click`, `api_name="predict"`)
* **Inputs:** 11 float inputs (`qNet`, `dvdtmax`, `vmax`, `vrest`, `APD50`, `APD90`, `max_dv`, `camax`, `carest`, `CaTD50`, `CaTD90`).
* **Returns:** A 4-element tuple (`prob_out`, `tier_out`, `conformal_out`, `action_out`):
  1. `prob_out` (`gr.Label`): Python dictionary mapping class names to float probabilities (e.g., `{"high": 0.0028, "intermediate": 0.4734, "low": 0.5238}`).
  2. `tier_out` (`gr.Textbox`, `visible=False`): Legacy prediction string (e.g., `"Predicted Risk Tier: Low"`).
  3. **`conformal_out` (`gr.JSON`) — YES, A CONFORMAL SET IS RETURNED!** Returns a full JSON object:
     ```json
     {
       "prediction_set": ["intermediate", "low"],
       "recommended_action": "intermediate",
       "set_size": 2,
       "is_ambiguous": true,
       "out_of_distribution": false,
       "alpha": 0.154,
       "q_hat": 0.6275,
       "calibration_note": "Conformal threshold calibrated at drug level on probability\nvectors averaged over 2000 replicate samples per drug (leave-one-drug-out over 12\ntraining drugs). This response is based on a single input row; the coverage\nguarantee does not strictly transfer to single-row input."
     }
     ```
  4. `action_out` (`gr.Markdown`): Prominently displays the primary `recommended_action`, prediction set, ambiguity rationale, or out-of-distribution warning banner.

#### Endpoint 2: `/explain` (`explain_btn.click`, `api_name="explain"`)
* **Inputs:** Same 11 float inputs.
* **Returns:** A single JSON object (`explain_out` / `gr.JSON`):
  ```json
  {
    "predicted_class": "high",
    "base_value": 0.3333333,
    "contributions": [
      {"biomarker": "qNet", "value": 43.637118, "shap": 0.1542},
      ...
    ]
  }
  ```
