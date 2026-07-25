# STEP 11 — BASE LEARNER REFERENCE PERFORMANCE

**Model Checkpoint SHA-256:** `A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2` (PASS)  
**Status Evaluasi:** Guard Reproduksi PASS (Chantest Row Macro-F1 = 0.688289, Drug Macro-F1 = 0.694444)  

---

## 1. Konfigurasi Estimator Beku

- **RandomForestClassifier (`stack.named_estimators_['rf']`):**
  - `n_features_in_`: 11
  - Hyperparameters: `{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 400, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}`
- **XGBClassifier (`stack.named_estimators_['xgb']`):**
  - `n_features_in_`: 11
  - Hyperparameters: `{'objective': 'multi:softprob', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.1, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 10, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 300, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': None, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None, 'num_class': 3}`

---

## 2. Performa Prediksi (Tiga Model, Tiga Platform)

### Tabel 1: Drug-Level Macro-F1 (6 Desimal)

| Model | Chantest | Nanion | Manual |
|---|---|---|---|
| RF | 0.877778 | 0.819373 | 0.755556 |
| XGB | 0.819373 | 0.754762 | 0.819373 |
| STACK | 0.694444 | 0.621032 | 0.754762 |

### Tabel 2: Jumlah Prediksi Benar / 16 Obat dan Under-Triage

| Model | Platform | Jumlah Benar / 16 | Jumlah Under-Triage | Row-Level Macro-F1 | Accuracy | Recall High | Recall Int | Recall Low |
|---|---|---|---|---|---|---|---|---|
| RF | chantest | 14/16 | 0 | 0.821534 | 0.875000 | 1.000000 | 0.714286 | 1.000000 |
| RF | nanion | 13/16 | 0 | 0.773452 | 0.812500 | 1.000000 | 0.714286 | 0.800000 |
| RF | manual | 12/16 | 1 | 0.737576 | 0.750000 | 1.000000 | 0.571429 | 0.800000 |
| XGB | chantest | 13/16 | 0 | 0.820091 | 0.812500 | 1.000000 | 0.714286 | 0.800000 |
| XGB | nanion | 12/16 | 0 | 0.753444 | 0.750000 | 1.000000 | 0.714286 | 0.600000 |
| XGB | manual | 13/16 | 0 | 0.739374 | 0.812500 | 1.000000 | 0.714286 | 0.800000 |
| STACK | chantest | 11/16 | 1 | 0.688289 | 0.687500 | 0.750000 | 0.714286 | 0.600000 |
| STACK | nanion | 10/16 | 1 | 0.632400 | 0.625000 | 0.750000 | 0.714286 | 0.400000 |
| STACK | manual | 12/16 | 0 | 0.692524 | 0.750000 | 1.000000 | 0.714286 | 0.600000 |

---

## 3. Selisih Berpasangan (STACK - Baseline)

| Platform | STACK - RF | Status vs RF | STACK - XGB | Status vs XGB |
|---|---|---|---|---|
| chantest | -0.183333 | STACK KALAH | -0.124929 | STACK KALAH |
| nanion | -0.198341 | STACK KALAH | -0.133730 | STACK KALAH |
| manual | -0.000794 | STACK KALAH | -0.064611 | STACK KALAH |

---

## 4. Paired Bootstrap Analysis (10.000 Iterasi, Seed 42)

**Aturan Resample Degenerate:**
We compute macro-average F1 solely over the classes PRESENT in the true labels of each resample, adjusting to avoid undefined division by zero.
Jumlah resample yang terdampak (missing class): 131 dari 10.000 iterasi di setiap platform.

| Platform | Perbandingan | Median Selisih | 95% Confidence Interval | Lebar CI | Memuat Nol? |
|---|---|---|---|---|---|
| chantest | STACK - RF | -0.187302 | [-0.448155, +0.000000] | 0.448155 | Ya |
| chantest | STACK - XGB | -0.123151 | [-0.357156, +0.000000] | 0.357156 | Ya |
| nanion | STACK - RF | -0.197739 | [-0.454772, +0.000000] | 0.454772 | Ya |
| nanion | STACK - XGB | -0.126623 | [-0.374185, +0.000000] | 0.374185 | Ya |
| manual | STACK - RF | +0.000000 | [-0.197902, +0.188889] | 0.386791 | Ya |
| manual | STACK - XGB | -0.061601 | [-0.241282, +0.000000] | 0.241282 | Ya |

---

## SANITY FLAGS

- `FLAG: chantest STACK - RF CI width (0.448155) exceeds 0.30`
- `FLAG: chantest STACK - XGB CI width (0.357156) exceeds 0.30`
- `FLAG: nanion STACK - RF CI width (0.454772) exceeds 0.30`
- `FLAG: nanion STACK - XGB CI width (0.374185) exceeds 0.30`
- `FLAG: manual STACK - RF CI width (0.386791) exceeds 0.30`
- `FLAG: On chantest, RF drug-level macro-F1 (0.877778) exceeds STACK (0.694444)`
- `FLAG: On chantest, XGB drug-level macro-F1 (0.819373) exceeds STACK (0.694444)`
- `FLAG: On nanion, RF drug-level macro-F1 (0.819373) exceeds STACK (0.621032)`
- `FLAG: On nanion, XGB drug-level macro-F1 (0.754762) exceeds STACK (0.621032)`
- `FLAG: On manual, RF drug-level macro-F1 (0.755556) exceeds STACK (0.754762)`
- `FLAG: On manual, XGB drug-level macro-F1 (0.819373) exceeds STACK (0.754762)`
- `FLAG: On chantest, RF has fewer under-triage (0) than STACK (1)`
- `FLAG: On chantest, XGB has fewer under-triage (0) than STACK (1)`
- `FLAG: On nanion, RF has fewer under-triage (0) than STACK (1)`
- `FLAG: On nanion, XGB has fewer under-triage (0) than STACK (1)`