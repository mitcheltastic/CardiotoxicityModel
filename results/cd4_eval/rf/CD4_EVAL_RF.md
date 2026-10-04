# Comprehensive Evaluation Report: Random Forest Base Learner (CD-4)

**Evaluation Target:** `RandomForestClassifier` base learner inside [`hf_space/deployable_chantest_3class.joblib`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/hf_space/deployable_chantest_3class.joblib) (`bundle['model'].estimators_[0]`).  
**Verification vs Research Model:** Bit-level equivalence confirmed against `models/chantest_stacking_model_rev.joblib` (`named_estimators_['rf']`).  
**Random Seed & Bootstrapping:** Seed 42, 10,000 bootstrap iterations for drug-level macro-F1 and CiPA replicate OvR AUCs.  
**Execution Environment:** `Python 3.11`, CPU: `Intel64 Family 6 Model 186 Stepping 2` (13th Gen Intel Core i7-13620H).  
**Evaluation Script:** [`results/cd4_eval/rf/run_eval_rf.py`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/rf/run_eval_rf.py).

---

## 1. Model Confirmation & Verification

The Random Forest base learner was extracted from the deployable model bundle and audited against the standalone research stacking model:

| Property | Deployable Bundle RF (`estimators_[0]`) | Research Stacking RF (`named_estimators_['rf']`) | Match Status |
|:---|:---|:---|:---:|
| **Class** | `sklearn.ensemble._forest.RandomForestClassifier` | `sklearn.ensemble._forest.RandomForestClassifier` | **IDENTICAL** |
| **Number of Trees** | 400 | 400 | **IDENTICAL** |
| **Criterion** | `entropy` | `entropy` | **IDENTICAL** |
| **Max Depth** | `None` | `None` | **IDENTICAL** |
| **Random State** | 42 | 42 | **IDENTICAL** |
| **ChanTest Test Max Prob Diff** | `0.00000000` | `0.00000000` | **BIT-IDENTICAL (0.0)** |
| **Nanion Test Max Prob Diff** | `0.00000000` | `0.00000000` | **BIT-IDENTICAL (0.0)** |
| **Manual Test Max Prob Diff** | `0.00000000` | `0.00000000` | **BIT-IDENTICAL (0.0)** |

*Note:* The only structural parameter difference between bundles is `n_jobs=-1` (deployable bundle) vs `n_jobs=None` (research model), which does not affect model weights, decision paths, or predicted probabilities.

---

## 2. Multi-Platform Test Performance

Evaluations conducted on the three held-out CiPA test sets (16 drugs each: 4 High, 7 Intermediate, 5 Low). Replicate averaging maps 2,000 virtual cells per drug to a single drug-level consensus probability vector.

### 2.1 Overall Platform Summary

| Metric | ChanTest Test Split | Nanion Automated Patch Split | Manual Patch Clamp Split |
|:---|:---:|:---:|:---:|
| **Total Test Rows** | 32,000 | 32,000 | 31,999 |
| **Drug-Level Accuracy** | **14/16 (87.50%)** | **13/16 (81.25%)** | **12/16 (75.00%)** |
| **Drug Macro-F1** | **0.877778** | **0.819373** | **0.755556** |
| **Drug Macro-F1 [95% CI]** | **0.877778 [0.686610, 1.000000]** | **0.819373 [0.576068, 1.000000]** | **0.755556 [0.504545, 0.940741]** |
| **Row-Level Accuracy** | 0.817906 | 0.767125 | 0.732710 |
| **Row-Level Macro-F1** | 0.821534 | 0.773452 | 0.737576 |
| **Ordinal Drug AUC (Risk > Low)** | **1.0000** | **0.9636** | **0.9455** |
| **Ordinal Drug AUC (Risk > Int)** | **0.9167** | **0.8958** | **0.9583** |
| **Ordinal Row AUC (Risk > Low)** | 0.9871 | 0.9356 | 0.9325 |
| **Ordinal Row AUC (Risk > Int)** | 0.8774 | 0.8824 | 0.9514 |
| **CiPA OvR AUC (High) [95% CI]** | 0.875 [0.854, 0.938] | 0.875 [0.854, 0.938] | 0.958 [0.896, 1.000] |
| **CiPA OvR AUC (Int) [95% CI]** | 0.746 [0.683, 0.873] | 0.722 [0.643, 0.833] | 0.730 [0.643, 0.833] |
| **CiPA OvR AUC (Low) [95% CI]** | 1.000 [0.927, 1.000] | 0.945 [0.836, 1.000] | 0.927 [0.855, 1.000] |

---

### 2.2 Per-Class Precision, Recall, and F1-Scores

| Platform | Class | Precision | Recall | F1-Score | Support |
|:---|:---|:---:|:---:|:---:|:---:|
| **ChanTest** | **High** | 0.6667 | 1.0000 | 0.8000 | 4 |
| | **Intermediate** | 1.0000 | 0.7143 | 0.8333 | 7 |
| | **Low** | 1.0000 | 1.0000 | 1.0000 | 5 |
| **Nanion** | **High** | 0.6667 | 1.0000 | 0.8000 | 4 |
| | **Intermediate** | 0.8333 | 0.7143 | 0.7692 | 7 |
| | **Low** | 1.0000 | 0.8000 | 0.8889 | 5 |
| **Manual** | **High** | 0.6667 | 1.0000 | 0.8000 | 4 |
| | **Intermediate** | 0.8000 | 0.5714 | 0.6667 | 7 |
| | **Low** | 0.8000 | 0.8000 | 0.8000 | 5 |

---

### 2.3 Drug Confusion Matrices `[H, I, L]`

Rows represent True Risk Tiers; columns represent Predicted Risk Tiers. Order: High (0), Intermediate (1), Low (2).

#### ChanTest Test Split
```
               Predicted High    Predicted Int    Predicted Low
True High            4                 0                0
True Int             2                 5                0
True Low             0                 0                5
```

#### Nanion Test Split
```
               Predicted High    Predicted Int    Predicted Low
True High            4                 0                0
True Int             2                 5                0
True Low             0                 1                4
```

#### Manual Test Split
```
               Predicted High    Predicted Int    Predicted Low
True High            4                 0                0
True Int             2                 4                1
True Low             0                 1                4
```

---

### 2.4 Drug Misclassifications & Clinical Safety Audits

Under clinical pharmacology criteria, misclassifying a drug to a less severe tier (**under-triage**) introduces fatal proarrhythmic risk, whereas misclassifying to a higher risk tier (**over-call**) is safe and conservative.

| Platform | Drug Name | True Tier | Predicted Tier | Error Type | Clinical Impact |
|:---|:---|:---:|:---:|:---:|:---|
| **ChanTest** | `domperidone` | Intermediate | High | **Over-call** | Conservative triage (safe) |
| **ChanTest** | `pimozide` | Intermediate | High | **Over-call** | Conservative triage (safe) |
| **Nanion** | `domperidone` | Intermediate | High | **Over-call** | Conservative triage (safe) |
| **Nanion** | `metoprolol` | Low | Intermediate | **Over-call** | Conservative triage (safe) |
| **Nanion** | `pimozide` | Intermediate | High | **Over-call** | Conservative triage (safe) |
| **Manual** | `clarithromycin` | Intermediate | Low | **Under-triage** | ⚠️ Hazard under uncalibrated hard-labeling |
| **Manual** | `domperidone` | Intermediate | High | **Over-call** | Conservative triage (safe) |
| **Manual** | `metoprolol` | Low | Intermediate | **Over-call** | Conservative triage (safe) |
| **Manual** | `pimozide` | Intermediate | High | **Over-call** | Conservative triage (safe) |

*Observation:* Under hard-label argmax decisions, Random Forest exhibits **0 under-triage errors on ChanTest**, **0 on Nanion**, and **1 on Manual** (`clarithromycin`). As shown in Section 3, conformal prediction completely eliminates this under-triage error.

---

## 3. Full Rank-11 LODO Conformal Calibration Table

### 3.1 Calibration Parameters
- **Calibration Set:** 12 ChanTest training drugs evaluated under Leave-One-Drug-Out (LODO).
- **Non-Conformity Metric:** $s_d = 1 - \bar{p}_{\text{true}}$ (probability shortfall on true class).
- **Target Quantile Rank:** **Rank 11** of 12 ($k = 11$, corresponding to calibration drug `sotalol`).
- **Quantile Cutoff ($\hat{q}$):** **`0.533987`**
- **Smallest Valid Significance Level ($\alpha$):** $\alpha = \frac{n + 1 - k}{n + 1} = \frac{13 - 11}{13} = \frac{2}{13} \approx \mathbf{0.153846}$
- **Guaranteed Nominal Coverage ($1 - \alpha$):** $\frac{11}{13} \approx \mathbf{84.615\%}$
- **Conformal Inclusion Threshold ($1 - \hat{q}$):** **`0.466013`** (A class $c$ enters the prediction set $\Gamma(x)$ if $\bar{p}_c \ge 0.466013$).

---

### 3.2 12-Drug LODO Calibration Table (Sorted by Non-Conformity Score $s_d$)

| Rank | Drug Name | True Class | $P(\text{High})$ | $P(\text{Intermediate})$ | $P(\text{Low})$ | Score $s_d = 1 - P(\text{True})$ | Operating Role |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | `cisapride` | Intermediate | 0.023487 | **0.976482** | 0.000030 | `0.023518` | Calibrated |
| **2** | `dofetilide` | High | **0.974995** | 0.024734 | 0.000271 | `0.025005` | Calibrated |
| **3** | `terfenadine` | Intermediate | 0.058741 | **0.897190** | 0.044069 | `0.102810` | Calibrated |
| **4** | `mexiletine` | Low | 0.006357 | 0.096546 | **0.897096** | `0.102904` | Calibrated |
| **5** | `bepridil` | High | **0.895961** | 0.002966 | 0.101073 | `0.104039` | Calibrated |
| **6** | `chlorpromazine` | Intermediate | 0.001469 | **0.757775** | 0.240756 | `0.242225` | Calibrated |
| **7** | `quinidine` | High | **0.726190** | 0.018756 | 0.255054 | `0.273810` | Calibrated |
| **8** | `ranolazine` | Low | 0.043715 | 0.279179 | **0.677106** | `0.322894` | Calibrated |
| **9** | `verapamil` | Low | 0.018283 | 0.350695 | **0.631023** | `0.368977` | Calibrated |
| **10** | `ondansetron` | Intermediate | 0.264494 | **0.621936** | 0.113570 | `0.378064` | Calibrated |
| **11** | `sotalol` | High | **0.466013** | 0.479129 | 0.054859 | **`0.533987`** | **Operating Cutoff ($\hat{q}$)** |
| **12** | `diltiazem` | Low | 0.000000 | 0.904169 | **0.095831** | `0.904169` | Excluded ($>\hat{q}$) |

---

### 3.3 Conformal Test Performance Across Platforms at Rank 11

| Conformal Metric | ChanTest Test Split | Nanion Automated Patch Split | Manual Patch Clamp Split |
|:---|:---:|:---:|:---:|
| **Nominal Coverage Guarantee** | 84.615% | 84.615% | 84.615% |
| **Empirical Coverage** | **14/16 (87.50%)** | **13/16 (81.25%)** | **13/16 (81.25%)** |
| **Average Prediction Set Size** | **1.0000** | **1.0000** | **1.0625** |
| **Ambiguous Drugs ($|\Gamma| > 1$)** | **0** | **0** | **1** (`clarithromycin`) |
| **Empty Sets ($|\Gamma| = 0$)** | **0** | **0** | **0** |
| **Under-Triage Count** | **0** | **0** | **0** |

#### Resolution of `clarithromycin` Under-Triage on Manual Platform:
On the manual patch clamp split, uncalibrated RF hard-labeling predicted `clarithromycin` as Low ($P(\text{High})=0.0000, P(\text{Int})=0.4905, P(\text{Low})=0.5095$), causing an under-triage error. Under Rank-11 Conformal Prediction ($1 - \hat{q} = 0.466013$):
- Both `intermediate` ($0.4905 \ge 0.466013$) and `low` ($0.5095 \ge 0.466013$) exceed the threshold.
- Conformal set: $\Gamma(\text{clarithromycin}) = \{\text{Intermediate}, \text{Low}\}$.
- Under the conservative clinical decision policy (selecting the most severe tier in the set), the recommended action escalates to **Intermediate**.
- Result: **Zero under-triage errors across all three evaluation platforms**.

---

## 4. Global SHAP Feature Importance (TreeExplainer)

Computed using `shap.TreeExplainer` on the exact 320-row stratified subsample of ChanTest test data used in `run_step7_shap.py` (Seed 42, 20 rows per drug across all 16 test drugs).

### 4.1 SHAP Importance Table

| Rank | Biomarker | Symbol / Axis Label | mean\|SHAP\| High | mean\|SHAP\| Int | mean\|SHAP\| Low | Aggregated mean\|SHAP\| |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **1** | `qNet` | $\text{qNet}$ | 0.102894 | 0.100464 | 0.113063 | **0.105474** |
| **2** | `APD50` | $\text{APD}_{50}$ | 0.114910 | 0.087185 | 0.085419 | **0.095838** |
| **3** | `APD90` | $\text{APD}_{90}$ | 0.109898 | 0.077531 | 0.046391 | **0.077940** |
| **4** | `max_dv` | $dV_m/dt_{repol}$ | 0.077992 | 0.060949 | 0.028774 | **0.055905** |
| **5** | `vrest` | $V_{m,rest}$ | 0.006523 | 0.077863 | 0.075980 | **0.053455** |
| **6** | `vmax` | $V_{m,peak}$ | 0.012476 | 0.053961 | 0.066290 | **0.044243** |
| **7** | `camax` | $\text{Ca}_{peak}$ | 0.026492 | 0.022308 | 0.012748 | **0.020516** |
| **8** | `CaTD50` | $\text{CaD}_{50}$ | 0.005612 | 0.024272 | 0.025476 | **0.018454** |
| **9** | `CaTD90` | $\text{CaD}_{90}$ | 0.016635 | 0.017235 | 0.014321 | **0.016064** |
| **10** | `carest` | $\text{Ca}_{rest}$ | 0.001547 | 0.010551 | 0.012063 | **0.008053** |
| **11** | `dvdtmax` | $dV_m/dt_{max}$ | 0.002083 | 0.003724 | 0.002403 | **0.002737** |

*Artifacts saved:*
- CSV Table: [`results/cd4_eval/rf/rf_shap_importance.csv`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/rf/rf_shap_importance.csv)
- High-Resolution Figure (300 dpi): [`results/cd4_eval/rf/fig_shap_global_rf.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/rf/fig_shap_global_rf.png)

---

## 5. Local Latency Profiling on One Input Row

Tested on a single representative input row (`azimilide` test sample) following exact deployment pre-processing:
- Host CPU: `13th Gen Intel(R) Core(TM) i7-13620H` (10 physical cores / 16 threads, up to 4.90 GHz)
- RF `predict_proba`: 100 benchmark iterations (10 warmup iterations)
- `TreeExplainer` explain: 30 benchmark iterations (1 warmup iteration)
- `KernelExplainer` explain: 30 benchmark iterations with 15 k-means background centroids, `nsamples=500` (1 warmup iteration)

### 5.1 Latency Benchmark Results

| Operation / Explainer | Repetitions | Mean (ms) | P50 / Median (ms) | P95 (ms) | Max (ms) | Relative Speed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RF `predict_proba`** | 100 | 38.0599 | 36.2974 | 48.2780 | 80.3379 | Baseline |
| **`shap.TreeExplainer`** | 30 | **1.3723** | **1.3700** | **1.5065** | **1.7956** | **99.8x faster** |
| **`shap.KernelExplainer` (15 centroids, $n=500$)** | 30 | 136.9688 | 136.3202 | 144.3449 | 150.4776 | 1.0x (Reference) |

*Key Findings:*
1. **TreeExplainer Acceleration:** For tree-based ensembles like Random Forest, `TreeExplainer` executes in **1.37 ms** per sample, compared to **136.97 ms** for `KernelExplainer`, yielding a **~100x latency reduction**.
2. **Real-Time Deployment Feasibility:** At 1.37 ms per inference, full SHAP feature attribution can be computed synchronously inside API response cycles without introducing noticeable UI latency.

*Artifact saved:* [`results/cd4_eval/rf/rf_latency_benchmark.json`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/1.%20RISET%20ML/5.%20Here%20We%20Go%20Bismillah/TrainAI/results/cd4_eval/rf/rf_latency_benchmark.json)

---

## 6. Summary Comparison: RF vs Deployed Stacking Classifier

| Metric | Random Forest (`estimators_[0]`) | Deployed Stacking (`StackingClassifier`) | Delta ($\Delta_{\text{RF} - \text{Stack}}$) |
|:---|:---:|:---:|:---:|
| **ChanTest Drug Macro-F1** | **0.877778** (14/16) | **0.819373** (13/16) | **+0.058405 (+1 drug)** |
| **Nanion Drug Macro-F1** | **0.819373** (13/16) | **0.819373** (13/16) | 0.000000 (Identical) |
| **Manual Drug Macro-F1** | **0.755556** (12/16) | **0.755556** (12/16) | 0.000000 (Identical) |
| **Conformal Calibration ($\hat{q}$)** | **0.533987** (`sotalol`) | **0.648421** (`ondansetron`) | -0.114434 (Tighter) |
| **Conformal Inclusions Threshold ($1 - \hat{q}$)** | **0.466013** | **0.351579** | +0.114434 |
| **Conformal Under-Triage (All Platforms)** | **0** | **0** | 0 (Zero hazard) |
| **Local SHAP Latency (Mean)** | **1.37 ms** (TreeExplainer) | **140.23 ms** (KernelExplainer) | **~100x speedup** |
