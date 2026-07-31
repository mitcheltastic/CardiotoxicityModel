# Step 21B: Saving Drug-Level Probabilities for Random Forest Baseline Model

This document reports the extraction, reproducibility verification, and integrity audits of drug-level probability CSV files for the **Random Forest** baseline estimator extracted directly from the frozen Stacking Classifier artifact (`models/chantest_stacking_model_rev.joblib`).

---

## 1. LANGKAH 1 — Frozen Model SHA-256 Audit & RF Extraction

- **Model Artifact**: `models/chantest_stacking_model_rev.joblib`
- **SHA-256 Hash**: `A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2` (**PASS**)
- **Extracted Estimator**: `stack.named_estimators_['rf']`
- **Fitted Status (`n_features_in_`)**: `11` (**PASS**)
- **Criterion**: `entropy` | **n_estimators**: `400` | **random_state**: `42`

---

## 2. LANGKAH 2 & 3 — Reproducibility Verification

| Platform | Computed Macro-F1 | Expected Macro-F1 | Difference | Reproducibility Status |
| :--- | :---: | :---: | :---: | :---: |
| `chantest` | 0.877778 | 0.877778 | 0.00e+00 | **PASS** |
| `nanion` | 0.819373 | 0.819373 | 0.00e+00 | **PASS** |
| `manual` | 0.755556 | 0.755556 | 0.00e+00 | **PASS** |

---

## 3. LANGKAH 4 & 5 — Saved Probability CSV Audit & Integrity Guards

| File Name | Total Rows | Train Rows | Test Rows | Prob Sum = 1.0 | Drug Names Match | SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `rf_probs_chantest_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `89D441661C8FF29E097F493A8D0A22EA5AF44C7D49C83EED23F3D5862A264F7C` |
| `rf_probs_nanion_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `5B17BEE8A07B02E143F94B9D281BE31F4659242B681130973C0C383A0C51EFBD` |
| `rf_probs_manual_rev.csv` | 28 | 12 | 16 | **PASS** | **PASS** | `63E5492427CC4860FB23250339217677E5116160CE37EB9B6775BBA95C1C676B` |

---

## 4. SANITY FLAGS

NONE (All reproducibility diffs < 1e-6, exact drug-level probability sums = 1.0, clean integrity guards).

---
*Report generated automatically by `scripts/pipeline/run_step21b_save_rf_probs.py`.*
