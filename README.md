# In-Silico Cardiotoxicity (TdP) Risk Prediction

> ## ⚠️ STATUS DOKUMEN
> Angka dan hasil di README ini berasal dari dataset **sebelum revisi `_rev`** dan
> **sudah tidak berlaku**. Dataset telah diganti pada 2026-07-25 dan seluruh hasil
> dihitung ulang.
>
> **Sumber angka yang berlaku:** `docs/PAPER_NUMBERS.md`
> **Ringkasan naratif:** `docs/STORYLINE.md`
> **Figure resmi:** `results/` (tertanggal 2026-07-25)

A machine learning pipeline for predicting **Torsade de Pointes (TdP)** cardiotoxicity risk from 11 CiPA cardiac biomarkers. Built as part of an undergraduate thesis project.

## Overview

This project classifies drugs into three clinical risk tiers — **High**, **Intermediate**, and **Low** — using stacking ensembles of Random Forest, XGBoost, Logistic Regression, SVM, and deep learning (ANN) models. The pipeline addresses a critical **data leakage** issue discovered during development by enforcing drug-level grouped cross-validation (`StratifiedGroupKFold`).

> ⚠️ Tabel di bawah ini berisi angka dari dataset lama. Angka yang berlaku ada di
> `docs/PAPER_NUMBERS.md`.

### Key Results

| Metric | Value |
|:---|:---|
| **Test Macro-F1** | 0.828 |
| **Bootstrap Median F1 [95% CI]** | 0.819 [0.694, 0.878] |
| **High-Risk Recall** | 0.992 |
| **Critical Misses (High → Low)** | **0** |

The deployable model is a **Stacking Ensemble** (`RF + XGB`, meta-learner: `RF`) trained on the `chantest` dataset.

## Biomarker Features

The model uses the 11 biomarkers from the [CiPA initiative](https://cipaproject.org/):

`qNet` · `dvdtmax` · `vmax` · `vrest` · `APD50` · `APD90` · `max_dv` · `camax` · `carest` · `CaTD50` · `CaTD90`

## Project Structure

> ⚠️ Bagian di bawah ini merujuk artifak dari dataset lama (sebelum _rev) dan sudah tidak
> berlaku. Hasil yang berlaku ada di results/ dan docs/PAPER_NUMBERS.md.

```
TrainAI/
├── data/                         # Simulation datasets (train/test splits)
├── docs/                         # Audit reports, step-by-step validation, and paper numbers
├── figures/                      # Project figure output directory
├── hf_space/                     # Hugging Face Spaces Gradio web application & deployment
├── logs/                         # Execution and pipeline logs
├── models/                       # Trained model checkpoints & saved states
├── results/                      # ⭐ Official _rev publication figures & evaluation tables (2026-07-25)
├── scripts/                      # Modular execution scripts
│   ├── legacy/                   # Historical pipeline scripts
│   ├── pipeline/                 # Core training, CV, and evaluation scripts
│   └── reporting/                # Reporting and figure generation scripts
├── src/                          # Source modules and data loader utilities
├── artifacts/                    # (artifak lama, jangan dipakai untuk paper)
├── deploy_utils.py               # Utility wrappers for model deployment
├── plus_ultra.ipynb              # Notebook development workspace
└── README.md                     # Project README
```

> ⚠️ Angka pada bagian ini (F1 0.999 / 0.460, 612 konfigurasi, hasil SHAP) berasal dari
> dataset lama. Protokol yang dipakai sekarang dijelaskan di `docs/STORYLINE.md`.

## Methodology

1. **Data Validation** — Verified shapes, disjoint drug splits (12 train / 16 test drugs), and handled missing values.
2. **Leakage Discovery** — Row-level CV gave F1 ≈ 0.999; drug-level CV dropped to F1 ≈ 0.460. Fixed by switching to `StratifiedGroupKFold(n_splits=4)` grouped by drug.
3. **Ensemble Sweep** — Evaluated 612 configurations across 3 datasets × 7 base learners × 6 ensemble types.
4. **Bootstrap Validation** — 10,000-iteration drug-level bootstrap on the test set for robust confidence intervals.
5. **SHAP Explainability** — `KernelExplainer` confirmed `qNet` as the top predictor, aligning with CiPA literature.

## Usage

### Loading the Deployable Model

> **⚠️ Important:** `deploy_utils.py` must be importable when loading the model.

> ⚠️ Bagian di bawah ini merujuk artifak dari dataset lama (sebelum _rev) dan sudah tidak
> berlaku. Hasil yang berlaku ada di results/ dan docs/PAPER_NUMBERS.md.

```python
import deploy_utils  # Must be imported before loading
import joblib

bundle = joblib.load("artifacts/deployable_chantest_3class.joblib")

model  = bundle["model"]      # Fitted StackingClassifier
scaler = bundle["scaler"]     # Fitted StandardScaler
classes = bundle["class_map"] # {0: 'high', 1: 'intermediate', 2: 'low'}
```

### Making Predictions

```python
import numpy as np

# 11 biomarkers in order:
# qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90
raw_input = np.array([[4.07, 261.0, 38.8, -87.7, 196.5, 271.3, 261.0, 0.35, 0.07, 218.8, 384.0]])

scaled = scaler.transform(raw_input)
prediction = model.predict(scaled)
probabilities = model.predict_proba(scaled)

print(f"Risk: {classes[prediction[0]]}")
print(f"Probabilities: {dict(zip(classes.values(), probabilities[0]))}")
```

## Requirements

- Python 3.10+
- scikit-learn
- xgboost
- tensorflow / keras
- numpy, pandas
- shap
- joblib
- matplotlib, seaborn

## License

This project is part of an academic thesis. Please contact the author for usage permissions.
