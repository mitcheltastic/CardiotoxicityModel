# Step 19: Baseline Logistic Regression Models & Conformal Calibration

This document reports the implementation, hyperparameter tuning, cross-platform performance evaluation, Leave-One-Drug-Out (LODO) conformal calibration, reference guard verification, and sanity checks for two baseline models:
1. **Model A**: Multivariate Logistic Regression (11 canonical biomarkers).
2. **Model B**: Single-Biomarker Logistic Regression (`qNet`-only).

All models were trained strictly on the `chantest` training dataset (12 drugs) using pre-normalized `_rev` features without additional scaling, and evaluated zero-shot across `chantest`, `nanion`, and `manual` testing datasets (16 drugs each).

---

## 1. Model Configuration & Hyperparameter Tuning

- **Algorithm**: `LogisticRegression(max_iter=5000, random_state=42, multi_class='multinomial', solver='lbfgs')`
- **Validation**: `GridSearchCV` using `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` grouped by `drug_name`.
- **Metric**: `f1_macro`
- **Parameter Grid**: $C \in [0.01, 0.1, 1.0, 10.0]$

### Tuning Results
- **Model A (Multivariate, 11 biomarkers)**: Selected $C = 0.01$, CV Macro-F1 = `0.836161`
- **Model B (Single-biomarker, qNet-only)**: Selected $C = 0.01$, CV Macro-F1 = `0.956046`

---

## 2. LANGKAH 1 — Guard Assertions

| Guard Check | Requirement | Result |
| :--- | :--- | :--- |
| **Pipeline Scaler Assert** | Zero scaling transformers (`StandardScaler`, etc.) in pipeline | **PASS** |
| **Train Drug Count** | Exactly 12 drugs in `chantest` training set | **PASS** |
| **Test Drug Count** | Exactly 16 drugs in `chantest`, `nanion`, `manual` testing sets | **PASS** |
| **Drug Disjointness** | Training set drugs and testing set drugs are disjoint | **PASS** |

---

## 3. LANGKAH 2 — Cross-Platform Performance

### Table 1: Classification Performance (Drug-Level & Row-Level)

| Model | Platform | Drug Macro-F1 | Row Macro-F1 | Benar / 16 | Over-called | Under-triaged |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Model A (Multivariate) | `chantest` | 0.8194 | 0.8168 | 13/16 | 3 | 0 |
| Model A (Multivariate) | `nanion` | 0.7548 | 0.7707 | 12/16 | 4 | 0 |
| Model A (Multivariate) | `manual` | 0.8194 | 0.7485 | 13/16 | 3 | 0 |
| Model B (qNet-only) | `chantest` | 0.8120 | 0.7523 | 13/16 | 2 | 1 |
| Model B (qNet-only) | `nanion` | 0.7566 | 0.7186 | 12/16 | 3 | 1 |
| Model B (qNet-only) | `manual` | 0.7515 | 0.7153 | 12/16 | 3 | 1 |

### Table 2: Ordinal Binary AUC Performance

| Model | Platform | AUC Ambang 1: $P(\text{risk} > \text{low})$ | AUC Ambang 2: $P(\text{risk} > \text{int})$ |
| :--- | :--- | :---: | :---: |
| Model A (Multivariate) | `chantest` | 1.0000 | 0.9583 |
| Model A (Multivariate) | `nanion` | 0.9455 | 0.9375 |
| Model A (Multivariate) | `manual` | 0.9091 | 0.9583 |
| Model B (qNet-only) | `chantest` | 1.0000 | 0.9583 |
| Model B (qNet-only) | `nanion` | 0.9818 | 0.9375 |
| Model B (qNet-only) | `manual` | 0.9091 | 0.9583 |

---

## 4. LANGKAH 3 — LODO Conformal Calibration

12-fold Leave-One-Drug-Out (LODO) calibration performed on `chantest` training set (12 genuine refits per model). Non-conformity scores: $s_d = 1 - p_{\text{mean}}[y_{\text{true}}]$.

### 12 LODO Non-Conformity Scores (Sorted Ascending)

- **Model A (Multivariate)**: `[np.float64(0.000863), np.float64(0.133607), np.float64(0.155923), np.float64(0.456524), np.float64(0.528024), np.float64(0.555531), np.float64(0.573047), np.float64(0.584763), np.float64(0.585469), np.float64(0.624157), np.float64(0.645363), np.float64(0.659639)] `
- **Model B (qNet-only)**: `[np.float64(0.164126), np.float64(0.38422), np.float64(0.449076), np.float64(0.453186), np.float64(0.508326), np.float64(0.553514), np.float64(0.580099), np.float64(0.598758), np.float64(0.657271), np.float64(0.659738), np.float64(0.663626), np.float64(0.667614)] `

### Rank Sweep ($k = 12 \dots 8$)

#### Model A (Multivariate)
| Rank $k$ | $\hat{q}$ | Threshold ($1-\hat{q}$) | Alpha Interval | Nominal Coverage |
| :---: | :---: | :---: | :---: | :---: |
| 12 | 0.659639 | 0.340361 | [0.0769, 0.1538) | 92.31% |
| 11 | 0.645363 | 0.354637 | [0.1538, 0.2308) | 84.62% |
| 10 | 0.624157 | 0.375843 | [0.2308, 0.3077) | 76.92% |
| 9 | 0.585469 | 0.414531 | [0.3077, 0.3846) | 69.23% |
| 8 | 0.584763 | 0.415237 | [0.3846, 0.4615) | 61.54% |

#### Model B (qNet-only)
| Rank $k$ | $\hat{q}$ | Threshold ($1-\hat{q}$) | Alpha Interval | Nominal Coverage |
| :---: | :---: | :---: | :---: | :---: |
| 12 | 0.667614 | 0.332386 | [0.0769, 0.1538) | 92.31% |
| 11 | 0.663626 | 0.336374 | [0.1538, 0.2308) | 84.62% |
| 10 | 0.659738 | 0.340262 | [0.2308, 0.3077) | 76.92% |
| 9 | 0.657271 | 0.342729 | [0.3077, 0.3846) | 69.23% |
| 8 | 0.598758 | 0.401242 | [0.3846, 0.4615) | 61.54% |

---

## 5. LANGKAH 4 — Conformal Prediction Set Results

### Prediction Set Evaluation Summary

| Model | Rank $k$ | Platform | Benar | Over | Under | Abstain | Avg Size | Size Distribution (0/1/2/3) | Empirical Cov | Nominal Cov | Diff |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model A (Multivariate) | 12 | `chantest` | 10 | 6 | 0 | 0 | 1.44 | 0/9/7/0 | 87.5% | 92.3% | -4.8% |
| Model A (Multivariate) | 12 | `nanion` | 11 | 5 | 0 | 0 | 1.38 | 0/10/6/0 | 87.5% | 92.3% | -4.8% |
| Model A (Multivariate) | 12 | `manual` | 11 | 5 | 0 | 0 | 1.31 | 0/11/5/0 | 81.2% | 92.3% | -11.1% |
| Model A (Multivariate) | 11 | `chantest` | 10 | 6 | 0 | 0 | 1.38 | 0/10/6/0 | 87.5% | 84.6% | +2.9% |
| Model A (Multivariate) | 11 | `nanion` | 11 | 5 | 0 | 0 | 1.38 | 0/10/6/0 | 87.5% | 84.6% | +2.9% |
| Model A (Multivariate) | 11 | `manual` | 11 | 5 | 0 | 0 | 1.31 | 0/11/5/0 | 81.2% | 84.6% | -3.4% |
| Model A (Multivariate) | 10 | `chantest` | 10 | 6 | 0 | 0 | 1.38 | 0/10/6/0 | 87.5% | 76.9% | +10.6% |
| Model A (Multivariate) | 10 | `nanion` | 11 | 5 | 0 | 0 | 1.38 | 0/10/6/0 | 87.5% | 76.9% | +10.6% |
| Model A (Multivariate) | 10 | `manual` | 11 | 5 | 0 | 0 | 1.31 | 0/11/5/0 | 81.2% | 76.9% | +4.3% |
| Model A (Multivariate) | 9 | `chantest` | 12 | 4 | 0 | 0 | 1.19 | 0/13/3/0 | 87.5% | 69.2% | +18.3% |
| Model A (Multivariate) | 9 | `nanion` | 11 | 5 | 0 | 0 | 1.19 | 0/13/3/0 | 81.2% | 69.2% | +12.0% |
| Model A (Multivariate) | 9 | `manual` | 11 | 5 | 0 | 0 | 1.25 | 0/12/4/0 | 81.2% | 69.2% | +12.0% |
| Model A (Multivariate) | 8 | `chantest` | 12 | 4 | 0 | 0 | 1.19 | 0/13/3/0 | 87.5% | 61.5% | +26.0% |
| Model A (Multivariate) | 8 | `nanion` | 11 | 5 | 0 | 0 | 1.19 | 0/13/3/0 | 81.2% | 61.5% | +19.7% |
| Model A (Multivariate) | 8 | `manual` | 11 | 5 | 0 | 0 | 1.25 | 0/12/4/0 | 81.2% | 61.5% | +19.7% |
| Model B (qNet-only) | 12 | `chantest` | 8 | 8 | 0 | 0 | 1.56 | 0/7/9/0 | 87.5% | 92.3% | -4.8% |
| Model B (qNet-only) | 12 | `nanion` | 9 | 7 | 0 | 0 | 1.56 | 0/7/9/0 | 87.5% | 92.3% | -4.8% |
| Model B (qNet-only) | 12 | `manual` | 9 | 7 | 0 | 0 | 1.62 | 0/6/10/0 | 87.5% | 92.3% | -4.8% |
| Model B (qNet-only) | 11 | `chantest` | 8 | 8 | 0 | 0 | 1.56 | 0/7/9/0 | 87.5% | 84.6% | +2.9% |
| Model B (qNet-only) | 11 | `nanion` | 9 | 7 | 0 | 0 | 1.56 | 0/7/9/0 | 87.5% | 84.6% | +2.9% |
| Model B (qNet-only) | 11 | `manual` | 9 | 7 | 0 | 0 | 1.62 | 0/6/10/0 | 87.5% | 84.6% | +2.9% |
| Model B (qNet-only) | 10 | `chantest` | 8 | 8 | 0 | 0 | 1.56 | 0/7/9/0 | 87.5% | 76.9% | +10.6% |
| Model B (qNet-only) | 10 | `nanion` | 9 | 7 | 0 | 0 | 1.56 | 0/7/9/0 | 87.5% | 76.9% | +10.6% |
| Model B (qNet-only) | 10 | `manual` | 9 | 7 | 0 | 0 | 1.62 | 0/6/10/0 | 87.5% | 76.9% | +10.6% |
| Model B (qNet-only) | 9 | `chantest` | 8 | 8 | 0 | 0 | 1.56 | 0/7/9/0 | 87.5% | 69.2% | +18.3% |
| Model B (qNet-only) | 9 | `nanion` | 9 | 7 | 0 | 0 | 1.50 | 0/8/8/0 | 87.5% | 69.2% | +18.3% |
| Model B (qNet-only) | 9 | `manual` | 9 | 7 | 0 | 0 | 1.62 | 0/6/10/0 | 87.5% | 69.2% | +18.3% |
| Model B (qNet-only) | 8 | `chantest` | 12 | 2 | 0 | 2 | 0.88 | 2/14/0/0 | 75.0% | 61.5% | +13.5% |
| Model B (qNet-only) | 8 | `nanion` | 11 | 3 | 1 | 1 | 0.94 | 1/15/0/0 | 68.8% | 61.5% | +7.2% |
| Model B (qNet-only) | 8 | `manual` | 11 | 3 | 1 | 1 | 0.94 | 1/15/0/0 | 68.8% | 61.5% | +7.2% |

### Tightest Operating Points (Zero Under-triage & Zero Abstain)
- **Model A (Multivariate)**: Rank `8`
- **Model B (qNet-only)**: Rank `9`

---

## 6. LANGKAH 5 — Reference Guard (STACK Model + Conformal Rank 11)

Re-calculated STACK model conformal prediction sets at Rank 11 ($\hat{q} = 0.762830$) from `results/cross_platform_probs_*_rev.csv`:

| Metric | Expected Value | Chantest | Nanion | Manual | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correct Predictions** | 11 / 11 / 11 | 11 | 11 | 11 | **PASS** |
| **Under-triaged Count** | 0 / 0 / 0 | 0 | 0 | 0 | **PASS** |
| **Average Set Size** | 1.1250 / 1.1250 / 1.3750 | 1.1250 | 1.1250 | 1.3750 | **PASS** |

**Reference Guard Overall Status**: **PASS**

---

## 7. SANITY FLAGS

- ⚠️ **WARNING**: Non-zero abstain rank: Model Model B (qNet-only), Rank 8 on chantest (2 abstains)
- ⚠️ **WARNING**: Non-zero abstain rank: Model Model B (qNet-only), Rank 8 on nanion (1 abstains)
- ⚠️ **WARNING**: Non-zero abstain rank: Model Model B (qNet-only), Rank 8 on manual (1 abstains)

---
*Report generated automatically by `scripts/pipeline/run_step19_logistic_baseline.py`.*