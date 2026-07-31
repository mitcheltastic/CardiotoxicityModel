# Step 20: Unified Final Results Table & Cross-Model Comparison

This document presents the unified final performance table comparing all four models (**Stacking**, **Random Forest**, **LR Multivariate**, **LR qNet-only**) and the **ALWAYS-HIGH** baseline across all three assay platforms (`chantest`, `nanion`, `manual`) at a uniform operating point (Rank 11, nominal coverage $= 11/13 \approx 84.62\%$).

---

## 1. LANGKAH 1 — LODO Score Comparison Across All Models

| Index | Stacking | Random Forest | LR Multivariate | LR qNet-only |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.045365 | 0.023518 | 0.000863 | 0.164126 |
| 2 | 0.094280 | 0.025005 | 0.133607 | 0.384220 |
| 3 | 0.152855 | 0.102810 | 0.155923 | 0.449076 |
| 4 | 0.163680 | 0.102904 | 0.456524 | 0.453186 |
| 5 | 0.236885 | 0.104039 | 0.528024 | 0.508326 |
| 6 | 0.255295 | 0.242225 | 0.555531 | 0.553514 |
| 7 | 0.408085 | 0.273810 | 0.573047 | 0.580099 |
| 8 | 0.453940 | 0.322894 | 0.584763 | 0.598758 |
| 9 | 0.630580 | 0.368977 | 0.585469 | 0.657271 |
| 10 | 0.703345 | 0.378064 | 0.624157 | 0.659738 |
| 11 | 0.762830 | 0.533987 | 0.645363 | 0.663626 |
| 12 | 0.986695 | 0.904169 | 0.659639 | 0.667614 |

### Quantile $\hat{q}$ at Ranks $12 \dots 8$

| Rank $k$ | Stacking | Random Forest | LR Multivariate | LR qNet-only |
| :---: | :---: | :---: | :---: | :---: |
| 12 | 0.986695 | 0.904169 | 0.659639 | 0.667614 |
| 11 | 0.762830 | 0.533987 | 0.645363 | 0.663626 |
| 10 | 0.703345 | 0.378064 | 0.624157 | 0.659738 |
| 9 | 0.630580 | 0.368977 | 0.585469 | 0.657271 |
| 8 | 0.453940 | 0.322894 | 0.584763 | 0.598758 |

---

## 2. LANGKAH 2 & 3 — Uniform Rank Sweep Analysis & Trade-offs

### Model Compliance per Rank (0 Under-triage & 0 Abstain across 3 platforms)

- **Rank 12**: 4 / 4 models compliant (Stacking, Random Forest, LR Multivariate, LR qNet-only)
- **Rank 11**: 4 / 4 models compliant (Stacking, Random Forest, LR Multivariate, LR qNet-only) — **Tightest Uniform Rank**
- **Rank 10**: 2 / 4 models compliant (LR Multivariate, LR qNet-only; Stacking has 1 under-triage on `nanion`, RF has 1 abstain on `chantest` & 3 on `manual`)
- **Rank 9**: 2 / 4 models compliant (LR Multivariate, LR qNet-only; Stacking has 1 under-triage on `nanion`, RF has 1 abstain on `chantest`, 1 on `nanion`, 3 on `manual`)
- **Rank 8**: 1 / 4 models compliant (LR Multivariate; Stacking has 1 under on `nanion` & abstains, LR qNet-only has 1 under on `nanion` & `manual`, RF has abstains)

> **Conclusion**: **Rank 11** ($q_{\hat{h}}$ thresholds, nominal coverage $11/13 pprox 84.62\%$) is the tightest single uniform rank that achieves zero under-triage and zero abstain across all 3 platforms for all 4 models simultaneously.

---

## 3. LANGKAH 4 — Unified Final Performance Table (Rank 11 Operating Point)

### Markdown Summary Table

| Model | Platform | Drug Macro-F1 | AUC Ambang 1 | AUC Ambang 2 | Hard Benar | Hard Over | Hard Under | Conformal Benar | Conformal Over | Conformal Under | Conformal Abstain | Avg Set Size | Emp Cov | Nom Cov |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stacking | `chantest` | 0.6944 | 0.9818 | 0.8750 | 11/16 | 4 | 1 | 11/16 | 5 | 0 | 0 | 1.12 | 75.0% | 84.6% |
| Stacking | `nanion` | 0.6210 | 0.9818 | 0.8750 | 10/16 | 5 | 1 | 11/16 | 5 | 0 | 0 | 1.12 | 75.0% | 84.6% |
| Stacking | `manual` | 0.7548 | 0.9273 | 0.9583 | 12/16 | 4 | 0 | 11/16 | 5 | 0 | 0 | 1.38 | 87.5% | 84.6% |
| Random Forest | `chantest` | 0.8778 | 1.0000 | 0.9167 | 14/16 | 2 | 0 | 14/16 | 2 | 0 | 0 | 1.00 | 87.5% | 84.6% |
| Random Forest | `nanion` | 0.8194 | 0.9636 | 0.8958 | 13/16 | 3 | 0 | 13/16 | 3 | 0 | 0 | 1.00 | 81.2% | 84.6% |
| Random Forest | `manual` | 0.7556 | 0.9455 | 0.9583 | 12/16 | 3 | 1 | 13/16 | 3 | 0 | 0 | 1.06 | 81.2% | 84.6% |
| LR Multivariate | `chantest` | 0.8194 | 1.0000 | 0.9583 | 13/16 | 3 | 0 | 10/16 | 6 | 0 | 0 | 1.38 | 87.5% | 84.6% |
| LR Multivariate | `nanion` | 0.7548 | 0.9455 | 0.9375 | 12/16 | 4 | 0 | 11/16 | 5 | 0 | 0 | 1.38 | 87.5% | 84.6% |
| LR Multivariate | `manual` | 0.8194 | 0.9091 | 0.9583 | 13/16 | 3 | 0 | 11/16 | 5 | 0 | 0 | 1.31 | 81.2% | 84.6% |
| LR qNet-only | `chantest` | 0.8120 | 1.0000 | 0.9583 | 13/16 | 2 | 1 | 8/16 | 8 | 0 | 0 | 1.56 | 87.5% | 84.6% |
| LR qNet-only | `nanion` | 0.7566 | 0.9818 | 0.9375 | 12/16 | 3 | 1 | 9/16 | 7 | 0 | 0 | 1.56 | 87.5% | 84.6% |
| LR qNet-only | `manual` | 0.7515 | 0.9091 | 0.9583 | 12/16 | 3 | 1 | 9/16 | 7 | 0 | 0 | 1.62 | 87.5% | 84.6% |
| ALWAYS-HIGH | `chantest` | 0.1333 | 0.5000 | 0.5000 | 4/16 | 12 | 0 | 4/16 | 12 | 0 | 0 | 1.00 | 100.0% | 84.6% |
| ALWAYS-HIGH | `nanion` | 0.1333 | 0.5000 | 0.5000 | 4/16 | 12 | 0 | 4/16 | 12 | 0 | 0 | 1.00 | 100.0% | 84.6% |
| ALWAYS-HIGH | `manual` | 0.1333 | 0.5000 | 0.5000 | 4/16 | 12 | 0 | 4/16 | 12 | 0 | 0 | 1.00 | 100.0% | 84.6% |

---

### LaTeX Code for Publication Table

```latex
\begin{table*}[t]
\centering
\caption{Unified Final Performance and Conformal Set Prediction Results across 4 Models and 3 Assay Platforms at Rank 11 Operating Point ($84.62\%$ Nominal Coverage).}
\label{tab:unified_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccc}
\toprule
\textbf{Model} & \textbf{Platform} & \textbf{Drug F1} & \textbf{AUC 1} & \textbf{AUC 2} & \textbf{Hard Correct} & \textbf{Hard Under} & \textbf{Conf Correct} & \textbf{Conf Under} & \textbf{Abstain} & \textbf{Avg Size} & \textbf{Emp Cov} \\
\midrule
Stacking & chantest & 0.6944 & 0.9818 & 0.8750 & 11/16 & 1 & 11/16 & 0 & 0 & 1.12 & 75.0\% \\
Stacking & nanion & 0.6210 & 0.9818 & 0.8750 & 10/16 & 1 & 11/16 & 0 & 0 & 1.12 & 75.0\% \\
Stacking & manual & 0.7548 & 0.9273 & 0.9583 & 12/16 & 0 & 11/16 & 0 & 0 & 1.38 & 87.5\% \\
Random Forest & chantest & 0.8778 & 1.0000 & 0.9167 & 14/16 & 0 & 14/16 & 0 & 0 & 1.00 & 87.5\% \\
Random Forest & nanion & 0.8194 & 0.9636 & 0.8958 & 13/16 & 0 & 13/16 & 0 & 0 & 1.00 & 81.2\% \\
Random Forest & manual & 0.7556 & 0.9455 & 0.9583 & 12/16 & 1 & 13/16 & 0 & 0 & 1.06 & 81.2\% \\
LR Multivariate & chantest & 0.8194 & 1.0000 & 0.9583 & 13/16 & 0 & 10/16 & 0 & 0 & 1.38 & 87.5\% \\
LR Multivariate & nanion & 0.7548 & 0.9455 & 0.9375 & 12/16 & 0 & 11/16 & 0 & 0 & 1.38 & 87.5\% \\
LR Multivariate & manual & 0.8194 & 0.9091 & 0.9583 & 13/16 & 0 & 11/16 & 0 & 0 & 1.31 & 81.2\% \\
LR qNet-only & chantest & 0.8120 & 1.0000 & 0.9583 & 13/16 & 1 & 8/16 & 0 & 0 & 1.56 & 87.5\% \\
LR qNet-only & nanion & 0.7566 & 0.9818 & 0.9375 & 12/16 & 1 & 9/16 & 0 & 0 & 1.56 & 87.5\% \\
LR qNet-only & manual & 0.7515 & 0.9091 & 0.9583 & 12/16 & 1 & 9/16 & 0 & 0 & 1.62 & 87.5\% \\
ALWAYS-HIGH & chantest & 0.1333 & 0.5000 & 0.5000 & 4/16 & 0 & 4/16 & 0 & 0 & 1.00 & 100.0\% \\
ALWAYS-HIGH & nanion & 0.1333 & 0.5000 & 0.5000 & 4/16 & 0 & 4/16 & 0 & 0 & 1.00 & 100.0\% \\
ALWAYS-HIGH & manual & 0.1333 & 0.5000 & 0.5000 & 4/16 & 0 & 4/16 & 0 & 0 & 1.00 & 100.0\% \\
\bottomrule
\end{tabular}%
}
\end{table*}
```

---

## 4. LANGKAH 5 — Under-triaged Drugs List on Hard Label

Below is the exhaustive list of drugs under-triaged by each model under hard-label prediction, including their class probability profiles ($p_{\text{high}}, p_{\text{int}}, p_{\text{low}}$):

| Model | Platform | Drug Name | True Label | Predicted Label | $p_{\text{high}}$ | $p_{\text{int}}$ | $p_{\text{low}}$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Stacking | `chantest` | `disopyramide` | `high` | `intermediate` | 0.4387 | 0.4958 | 0.0655 |
| Stacking | `nanion` | `disopyramide` | `high` | `intermediate` | 0.2645 | 0.7108 | 0.0247 |
| Random Forest | `manual` | `clarithromycin` | `intermediate` | `low` | 0.0000 | 0.4764 | 0.5236 |
| LR qNet-only | `chantest` | `disopyramide` | `high` | `intermediate` | 0.3809 | 0.4003 | 0.2188 |
| LR qNet-only | `nanion` | `disopyramide` | `high` | `intermediate` | 0.3803 | 0.4013 | 0.2184 |
| LR qNet-only | `manual` | `risperidone` | `intermediate` | `low` | 0.1976 | 0.3944 | 0.4080 |

---

## 5. LANGKAH 6 — Guard Verification

- **Stacking Conformal Rank 11 Correctness**: Expected 11/11/11 $\rightarrow$ Chantest=11, Nanion=11, Manual=11 (**PASS**)
- **Stacking Conformal Rank 11 Under-triage**: Expected 0/0/0 $\rightarrow$ Chantest=0, Nanion=0, Manual=0 (**PASS**)
- **Stacking Conformal Rank 11 Set Size**: Expected 1.1250/1.1250/1.3750 $\rightarrow$ Chantest=1.1250, Nanion=1.1250, Manual=1.3750 (**PASS**)

**Overall Guard Status**: **PASS**

---

## 6. SANITY FLAGS

NONE (All AUCs $\ge 0.5$, clean convergence, zero abstains for compliant models).

---
*Report generated automatically by `scripts/reporting/run_step20_unified_table.py`.*