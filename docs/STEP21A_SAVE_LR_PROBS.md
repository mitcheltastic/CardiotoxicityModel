# Step 21A: Saving Drug-Level Probabilities for Logistic Regression Baseline Models

This document details the generation, verification, and integrity audits of drug-level probability CSV files for both baseline Logistic Regression models (**Model A: Multivariate** and **Model B: qNet-only**).

---

## 1. LANGKAH 1 & 2 — Model Training & Persistence

- **Model A (Multivariate LR, 11 biomarkers)**: `LogisticRegression(C=0.01, max_iter=5000, random_state=42, solver='lbfgs')`
  - Saved to: `models/logistic_multivariate_rev.joblib`
  - SHA-256: `21F1C076CAF8886F0E068DF0A71143F2FF8CAD366E5CBDDDC6F577B5127C0958`
- **Model B (qNet-only LR, 1 feature)**: `LogisticRegression(C=0.01, max_iter=5000, random_state=42, solver='lbfgs')`
  - Saved to: `models/logistic_qnet_rev.joblib`
  - SHA-256: `90C2B6D44826F3BF1A3AEE7C90A93937C0CAB8E34E6872E96ACA5E36F3C9B6B7`

---

## 2. LANGKAH 3 — Reproducibility Verification against Step 19

| Model | Platform | Computed Macro-F1 | Expected Step 19 Macro-F1 | Diff | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Model A | `chantest` | 0.819373 | 0.819373 | 0.00e+00 | **PASS** |
| Model A | `nanion` | 0.754762 | 0.754762 | 0.00e+00 | **PASS** |
| Model A | `manual` | 0.819373 | 0.819373 | 0.00e+00 | **PASS** |
| Model B | `chantest` | 0.811966 | 0.811966 | 0.00e+00 | **PASS** |
| Model B | `nanion` | 0.756614 | 0.756614 | 0.00e+00 | **PASS** |
| Model B | `manual` | 0.751515 | 0.751515 | 0.00e+00 | **PASS** |

---

## 3. LANGKAH 4 & 5 — Saved Probability CSV Audit & Integrity Guards

| File Name | Rows | Training Rows | Testing Rows | Prob Sum = 1.0 | Name Match | SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `logistic_multivariate_probs_chantest_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `83DABFD817830EE5FAAFFA8D025F3C4A525D73E0DAA597AE13CFF61A01B90734` |
| `logistic_multivariate_probs_nanion_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `5CF3E9ECFD188171E4B7865B67B6DE9721BA9B6CE0E3A992A626A589240B238E` |
| `logistic_multivariate_probs_manual_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `96A4B3EE87106EA1D8333CD65975C0FEFFF40BB0EDE2AF27B0DCC3B8056041C2` |
| `logistic_qnet_probs_chantest_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `C6FBCD92D6D43976AA05A18CF59ACE0826C99F378EE534968FDFE031BEAC25D3` |
| `logistic_qnet_probs_nanion_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `F89A6C826E8F115656C248D0360A66192D34BEB36FBE492B8F0F34E29A1DF903` |
| `logistic_qnet_probs_manual_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `31DCCCA7BF9ADFE74C5D70FF57D6636ECA7B2A7F99CF574CC5859A376C6BBAAE` |

---

## 4. SANITY FLAGS

NONE (All reproducibility diffs < 1e-6, exact drug-level probability sums = 1.0, clean integrity guards).

---
*Report generated automatically by `scripts/pipeline/run_step21a_save_lr_probs.py`.*
