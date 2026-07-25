# CANONICAL PAPER NUMBERS
One canonical value per quantity, each with its source file. Every entry states its evaluation population.

## A. DATA
- **Rows dropped before loading**: 26 rows were dropped at load time because at least one of the 11 biomarker cells contained the Excel error string '#VALUE!': 25 rows in manual_training_scaled_rev.csv (all mexiletine, columns APD50/APD90/max_dv) and 1 row in manual_testing_scaled_rev.csv (ibutilide, CaTD90). No other file contained any sentinel. Audit trail: logs/DROPPED_ROWS_REV.csv. (Source: docs/DATA_AUDIT_REV_PART2.md, docs/LOADER_VERIFICATION_REV.md)
- **Chantest rows/drugs**: Training: 24000/12, Testing: 32000/16 (Source: docs/LOADER_VERIFICATION_REV.md)
- **Nanion rows/drugs**: Training: 24000/12, Testing: 32000/16 (Source: docs/LOADER_VERIFICATION_REV.md)
- **Manual rows/drugs**: Training: 23975/12, Testing: 31999/16 (Source: docs/LOADER_VERIFICATION_REV.md)
- **Training drugs (12)**: bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, sotalol, terfenadine, verapamil (Population: Chantest Training Split)
- **Testing drugs (16)**: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib (Population: Held-out Test Split)

## B. MODEL
- **RF Hyperparameters**: {'criterion': 'entropy', 'max_depth': None, 'n_estimators': 400} (Source: TrainAI/models/chantest_stacking_metadata_rev.json)
- **XGB Hyperparameters**: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 300} (Source: TrainAI/models/chantest_stacking_metadata_rev.json)
- **Meta-learner**: Random Forest (StackingClassifier final_estimator)
- **Stack method**: predict_proba
- **Passthrough**: False
- **CV Splitter**: StratifiedGroupKFold (k=4, grouped by drug)
- **Seed**: 42
- **sklearn version**: 1.8.0
- **xgboost version**: 3.2.0
- **Model artifact SHA-256**: A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2 (Source: TrainAI/models/chantest_stacking_model_rev.joblib)

## C. CLASSIFICATION

### CHANTEST (Population: 16 Test Drugs)
- **Drug-level macro-F1**: Point Estimate = 0.694444, Bootstrap Median = 0.679739 [95% CI: 0.391534, 0.896825] (Source: TrainAI/results/bootstrap_distributions_rev.csv)
- **Row-level macro-F1**: 0.688289 (Source: docs/STEP3_STACKING_CHANTEST.md)
- **Per-class (High/Int/Low) Point Estimates (Drug-level)**:
  - Precision: High=0.600000, Int=0.625000, Low=1.000000
  - Recall: High=0.750000, Int=0.714286, Low=0.600000
  - F1: High=0.666667, Int=0.666667, Low=0.750000
- **Confusion Matrix (Drug-level, True/Pred)**: High(0) [3, 1, 0], Int(1) [2, 5, 0], Low(2) [0, 2, 3] (Source: TrainAI/results/cross_platform_probs_chantest_rev.csv)

### NANION (Population: 16 Test Drugs)
- **Drug-level macro-F1**: Point Estimate = 0.621032, Bootstrap Median = 0.595238 [95% CI: 0.311111, 0.851852] (Source: TrainAI/results/bootstrap_distributions_rev.csv)
- **Row-level macro-F1**: 0.632400 (Source: docs/STEP4_CROSS_PLATFORM.md)
- **Per-class (High/Int/Low) Point Estimates (Drug-level)**:
  - Precision: High=0.600000, Int=0.555556, Low=1.000000
  - Recall: High=0.750000, Int=0.714286, Low=0.400000
  - F1: High=0.666667, Int=0.625000, Low=0.571429
- **Confusion Matrix (Drug-level, True/Pred)**: High(0) [3, 1, 0], Int(1) [2, 5, 0], Low(2) [0, 3, 2] (Source: TrainAI/results/cross_platform_probs_nanion_rev.csv)

### MANUAL (Population: 16 Test Drugs)
- **Drug-level macro-F1**: Point Estimate = 0.754762, Bootstrap Median = 0.738889 [95% CI: 0.479347, 0.944056] (Source: TrainAI/results/bootstrap_distributions_rev.csv)
- **Row-level macro-F1**: 0.692524 (Source: docs/STEP4_CROSS_PLATFORM.md)
- **Per-class (High/Int/Low) Point Estimates (Drug-level)**:
  - Precision: High=0.666667, Int=0.714286, Low=1.000000
  - Recall: High=1.000000, Int=0.714286, Low=0.600000
  - F1: High=0.800000, Int=0.714286, Low=0.750000
- **Confusion Matrix (Drug-level, True/Pred)**: High(0) [4, 0, 0], Int(1) [2, 5, 0], Low(2) [0, 2, 3] (Source: TrainAI/results/cross_platform_probs_manual_rev.csv)

## D. CONFORMAL
- **12 LODO scores**: [np.float64(0.045365), np.float64(0.09428), np.float64(0.152855), np.float64(0.16368), np.float64(0.236885), np.float64(0.255295), np.float64(0.408085), np.float64(0.45394), np.float64(0.63058), np.float64(0.703345), np.float64(0.76283), np.float64(0.986695)] (Source: TrainAI/results/conformal_lodo_scores_rev.csv, Population: 12 Chantest Train Drugs)
- **Rank 12**: q_hat = 0.986695 (Alpha interval: [1/13, 2/13)) (Source: TrainAI/results/conformal_rank_sweep_rev.csv)
- **Rank 11**: q_hat = 0.762830 (Alpha interval: [2/13, 3/13)) (Source: TrainAI/results/conformal_rank_sweep_rev.csv)
- **Rank 10**: q_hat = 0.703345 (Alpha interval: [3/13, 4/13)) (Source: TrainAI/results/conformal_rank_sweep_rev.csv)
- **Adopted Operating Point**: Rank 11, q_hat = 0.762830, alpha = 2/13, nominal coverage = 11/13 = 0.846154 (Population: 16 Test Drugs)

### CHANTEST Conformal (Population: 16 Test Drugs)
- **Correct-outcome count & proportion (Rank 11)**: 11/16 (0.6875) (Source: computed from TrainAI/results/cross_platform_probs_chantest_rev.csv)
- **Outcome breakdown**: Correct = 11, Over-called = 5, Under-triaged = 0 (Source: computed from TrainAI/results/cross_platform_probs_chantest_rev.csv)
- **Average set size**: 1.1250 (Source: computed from TrainAI/results/cross_platform_probs_chantest_rev.csv)
- **Set size distribution**: size 1 = 14, size 2 = 2, size 3 = 0 (Source: computed from TrainAI/results/cross_platform_probs_chantest_rev.csv)
- **Coverage (Empirical vs Nominal)**: 12/16 (0.7500) vs 0.8462 (Signed Gap: -0.0962, 95% Wilson CI: [0.5050, 0.8982], Source: computed from TrainAI/results/cross_platform_probs_chantest_rev.csv & docs/STEP6_CONFORMAL_OPERATING_POINT.md)
- **Rule-of-three under-triage bound**: 0 of 16 under-triaged observed; approximate 95% upper bound = 0.187500 (18.75%) (Source: docs/STEP9_BOOTSTRAP.md)

### NANION Conformal (Population: 16 Test Drugs)
- **Correct-outcome count & proportion (Rank 11)**: 11/16 (0.6875) (Source: computed from TrainAI/results/cross_platform_probs_nanion_rev.csv)
- **Outcome breakdown**: Correct = 11, Over-called = 5, Under-triaged = 0 (Source: computed from TrainAI/results/cross_platform_probs_nanion_rev.csv)
- **Average set size**: 1.1250 (Source: computed from TrainAI/results/cross_platform_probs_nanion_rev.csv)
- **Set size distribution**: size 1 = 14, size 2 = 2, size 3 = 0 (Source: computed from TrainAI/results/cross_platform_probs_nanion_rev.csv)
- **Coverage (Empirical vs Nominal)**: 12/16 (0.7500) vs 0.8462 (Signed Gap: -0.0962, 95% Wilson CI: [0.5050, 0.8982], Source: computed from TrainAI/results/cross_platform_probs_nanion_rev.csv & docs/STEP6_CONFORMAL_OPERATING_POINT.md)
- **Rule-of-three under-triage bound**: 0 of 16 under-triaged observed; approximate 95% upper bound = 0.187500 (18.75%) (Source: docs/STEP9_BOOTSTRAP.md)

### MANUAL Conformal (Population: 16 Test Drugs)
- **Correct-outcome count & proportion (Rank 11)**: 11/16 (0.6875) (Source: computed from TrainAI/results/cross_platform_probs_manual_rev.csv)
- **Outcome breakdown**: Correct = 11, Over-called = 5, Under-triaged = 0 (Source: computed from TrainAI/results/cross_platform_probs_manual_rev.csv)
- **Average set size**: 1.3750 (Source: computed from TrainAI/results/cross_platform_probs_manual_rev.csv)
- **Set size distribution**: size 1 = 10, size 2 = 6, size 3 = 0 (Source: computed from TrainAI/results/cross_platform_probs_manual_rev.csv)
- **Coverage (Empirical vs Nominal)**: 14/16 (0.8750) vs 0.8462 (Signed Gap: +0.0288, 95% Wilson CI: [0.6398, 0.9650], Source: computed from TrainAI/results/cross_platform_probs_manual_rev.csv & docs/STEP6_CONFORMAL_OPERATING_POINT.md)
- **Rule-of-three under-triage bound**: 0 of 16 under-triaged observed; approximate 95% upper bound = 0.187500 (18.75%) (Source: docs/STEP9_BOOTSTRAP.md)

## E. BASELINES
- **CHANTEST Always-High Baseline**: Correct = 4/16 (Source: computed from TrainAI/results/cross_platform_probs_chantest_rev.csv)
- **CHANTEST Hard-Label Model**: Correct = 11/16 (Source: computed from TrainAI/results/cross_platform_probs_chantest_rev.csv)
- **NANION Always-High Baseline**: Correct = 4/16 (Source: computed from TrainAI/results/cross_platform_probs_nanion_rev.csv)
- **NANION Hard-Label Model**: Correct = 10/16 (Source: computed from TrainAI/results/cross_platform_probs_nanion_rev.csv)
- **MANUAL Always-High Baseline**: Correct = 4/16 (Source: computed from TrainAI/results/cross_platform_probs_manual_rev.csv)
- **MANUAL Hard-Label Model**: Correct = 12/16 (Source: computed from TrainAI/results/cross_platform_probs_manual_rev.csv)

## F. SHAP
- **Aggregated Ranking**: ['qNet', 'vrest', 'max_dv', 'APD50', 'APD90', 'vmax', 'dvdtmax', 'camax', 'CaTD50', 'CaTD90', 'carest'] (Source: TrainAI/results/shap_global_importance_rev.csv, Population: 16 Test Drugs)
- **qNet canonical mean|SHAP|**: 0.145534 (Source: TrainAI/results/shap_global_importance_rev.csv)
- **vrest permutation finding**: permuting vrest changes on average 1.0 of 16 drug-level predictions (zero on many seeds), while permuting qNet changes 3.0; mean drug-level macro-F1 drop 0.073 for vrest vs 0.108 for qNet. (Source: docs/STEP8_SHAP_DIAGNOSTIC.md)

## G. FIGURES
- **fig1_block_diagram.pdf**: Path = TrainAI/results/fig1_block_diagram.pdf, Dimensions = 7.16 x 3.70 inches, 37.7 KB, Raster: False
- **fig_shap_global.pdf**: Path = TrainAI/results/fig_shap_global.pdf, Dimensions = 3.40 x 3.40 inches, 11.4 KB, Raster: False

==========================================
=== SANITY FLAGS ===
==========================================
All values compiled successfully. Point estimates strictly isolated from medians.


## COVERAGE MISS RECONCILIATION
Recomputed independently from `results/cross_platform_probs_*_rev.csv` at Rank 11 (`q_hat = 0.762830`, threshold = `0.237170`).

### Platform: CHANTEST
| Drug Name | True Label | p_mean[true_label] | Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| astemizole | 1 | 0.960353 | 0.237170 | CLEARS |
| azimilide | 0 | 0.989995 | 0.237170 | CLEARS |
| clarithromycin | 1 | 0.984211 | 0.237170 | CLEARS |
| clozapine | 1 | 0.897620 | 0.237170 | CLEARS |
| disopyramide | 0 | 0.438730 | 0.237170 | CLEARS |
| domperidone | 1 | 0.003185 | 0.237170 | MISSES |
| droperidol | 1 | 0.941604 | 0.237170 | CLEARS |
| ibutilide | 0 | 0.911960 | 0.237170 | CLEARS |
| loratadine | 2 | 0.222000 | 0.237170 | MISSES |
| metoprolol | 2 | 0.734105 | 0.237170 | CLEARS |
| nifedipine | 2 | 1.000000 | 0.237170 | CLEARS |
| nitrendipine | 2 | 0.880770 | 0.237170 | CLEARS |
| pimozide | 1 | 0.046999 | 0.237170 | MISSES |
| risperidone | 1 | 0.978645 | 0.237170 | CLEARS |
| tamoxifen | 2 | 0.060545 | 0.237170 | MISSES |
| vandetanib | 0 | 0.927865 | 0.237170 | CLEARS |

**Recomputed Misses**: ['domperidone', 'loratadine', 'pimozide', 'tamoxifen'] (Total: 4 misses, Coverage: 12/16)

### Platform: NANION
| Drug Name | True Label | p_mean[true_label] | Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| astemizole | 1 | 0.960270 | 0.237170 | CLEARS |
| azimilide | 0 | 0.989990 | 0.237170 | CLEARS |
| clarithromycin | 1 | 0.830514 | 0.237170 | CLEARS |
| clozapine | 1 | 0.989152 | 0.237170 | CLEARS |
| disopyramide | 0 | 0.264455 | 0.237170 | CLEARS |
| domperidone | 1 | 0.000000 | 0.237170 | MISSES |
| droperidol | 1 | 0.918105 | 0.237170 | CLEARS |
| ibutilide | 0 | 0.835310 | 0.237170 | CLEARS |
| loratadine | 2 | 0.245285 | 0.237170 | CLEARS |
| metoprolol | 2 | 0.157835 | 0.237170 | MISSES |
| nifedipine | 2 | 1.000000 | 0.237170 | CLEARS |
| nitrendipine | 2 | 1.000000 | 0.237170 | CLEARS |
| pimozide | 1 | 0.044843 | 0.237170 | MISSES |
| risperidone | 1 | 0.980140 | 0.237170 | CLEARS |
| tamoxifen | 2 | 0.059030 | 0.237170 | MISSES |
| vandetanib | 0 | 0.971765 | 0.237170 | CLEARS |

**Recomputed Misses**: ['domperidone', 'metoprolol', 'pimozide', 'tamoxifen'] (Total: 4 misses, Coverage: 12/16)

### Platform: MANUAL
| Drug Name | True Label | p_mean[true_label] | Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| astemizole | 1 | 0.965555 | 0.237170 | CLEARS |
| azimilide | 0 | 0.929785 | 0.237170 | CLEARS |
| clarithromycin | 1 | 0.558811 | 0.237170 | CLEARS |
| clozapine | 1 | 0.915694 | 0.237170 | CLEARS |
| disopyramide | 0 | 0.823025 | 0.237170 | CLEARS |
| domperidone | 1 | 0.318370 | 0.237170 | CLEARS |
| droperidol | 1 | 0.855402 | 0.237170 | CLEARS |
| ibutilide | 0 | 0.794762 | 0.237170 | CLEARS |
| loratadine | 2 | 0.714895 | 0.237170 | CLEARS |
| metoprolol | 2 | 0.061365 | 0.237170 | MISSES |
| nifedipine | 2 | 1.000000 | 0.237170 | CLEARS |
| nitrendipine | 2 | 1.000000 | 0.237170 | CLEARS |
| pimozide | 1 | 0.293695 | 0.237170 | CLEARS |
| risperidone | 1 | 0.886972 | 0.237170 | CLEARS |
| tamoxifen | 2 | 0.147010 | 0.237170 | MISSES |
| vandetanib | 0 | 0.529700 | 0.237170 | CLEARS |

**Recomputed Misses**: ['metoprolol', 'tamoxifen'] (Total: 2 misses, Coverage: 14/16)

### Specific Drug Derivation (tamoxifen vs metoprolol)
- **Chantest - tamoxifen** (true low=2): `p_mean[2] = 0.060545` (< 0.237170) -> **MISSES**
- **Chantest - metoprolol** (true low=2): `p_mean[2] = 0.734105` (>= 0.237170) -> **CLEARS**
- **Nanion - tamoxifen** (true low=2): `p_mean[2] = 0.059030` (< 0.237170) -> **MISSES**
- **Nanion - metoprolol** (true low=2): `p_mean[2] = 0.157835` (< 0.237170) -> **MISSES**

### Document Comparison & Settlement
- **STEP6 (`docs/STEP6_CONFORMAL_OPERATING_POINT.md`)**: Recorded misses:
  - Chantest: `['domperidone', 'loratadine', 'pimozide', 'tamoxifen']` (4 misses)
  - Nanion: `['domperidone', 'metoprolol', 'pimozide', 'tamoxifen']` (4 misses)
  - Status: **100% MATCHES THE RECOMPUTATION**.
- **PAPER_NUMBERS (`docs/PAPER_NUMBERS.md`)**: Any prior draft claiming metoprolol missed on Chantest or loratadine missed on Nanion is **SUPERSEDED** by the numeric proof above.
- **Confirmed Final Counts & Coverage**:
  - Chantest Misses = 4/16 -> **Coverage: 12/16 (0.7500)**
  - Nanion Misses = 4/16 -> **Coverage: 12/16 (0.7500)**
  - Manual Misses = 2/16 -> **Coverage: 14/16 (0.8750)**

---

---

## H. BASELINE BASE LEARNER DAN CONFORMAL RF

Setiap angka pada seksi ini secara eksplisit menyebutkan file sumber dan populasi evaluasinya.

### H1. Performa Base Learner (16 Obat Uji, Model Beku, Tanpa Refit)
* **Sumber Data:** `docs/STEP11_BASE_LEARNER_REFERENCE.md` & `results/base_learner_reference_rev.csv`
* **Populasi Evaluasi:** 16 obat test held-out per platform (`chantest`, `nanion`, `manual`), model beku dari `models/chantest_stacking_model_rev.joblib`.
* **Metrik Drug-Level Macro-F1 & Under-Triage:**
  - **Chantest Platform (16 obat):**
    - RF Base Learner: Macro-F1 = `0.877778` (Benar: 14/16, Under-triage: `0`)
    - XGB Base Learner: Macro-F1 = `0.819373` (Benar: 13/16, Under-triage: `0`)
    - STACK Model: Macro-F1 = `0.694444` (Benar: 11/16, Under-triage: `1` [senyawa `disopyramide`])
  - **Nanion Platform (16 obat):**
    - RF Base Learner: Macro-F1 = `0.819373` (Benar: 13/16, Under-triage: `0`)
    - XGB Base Learner: Macro-F1 = `0.754762` (Benar: 12/16, Under-triage: `0`)
    - STACK Model: Macro-F1 = `0.621032` (Benar: 10/16, Under-triage: `1` [senyawa `disopyramide`])
  - **Manual Platform (16 obat):**
    - RF Base Learner: Macro-F1 = `0.755556` (Benar: 12/16, Under-triage: `1` [senyawa `clarithromycin`])
    - XGB Base Learner: Macro-F1 = `0.819373` (Benar: 13/16, Under-triage: `0`)
    - STACK Model: Macro-F1 = `0.754762` (Benar: 12/16, Under-triage: `0`)
* **Selisih Berpasangan Paired Bootstrap STACK minus Base Learner (10.000 Iterasi, Seed 42, Verbatim STEP11):**
  - **Chantest Platform:**
    - STACK - RF: Point Estimate Diff = `-0.183333` | Bootstrap Median = `-0.187302` (95% CI `[-0.448155, +0.000000]`, Lebar CI: `0.448155`, Memuat 0: YA)
    - STACK - XGB: Point Estimate Diff = `-0.124929` | Bootstrap Median = `-0.123151` (95% CI `[-0.357156, +0.000000]`, Lebar CI: `0.357156`, Memuat 0: YA)
  - **Nanion Platform:**
    - STACK - RF: Point Estimate Diff = `-0.198341` | Bootstrap Median = `-0.197739` (95% CI `[-0.454772, +0.000000]`, Lebar CI: `0.454772`, Memuat 0: YA)
    - STACK - XGB: Point Estimate Diff = `-0.133730` | Bootstrap Median = `-0.126623` (95% CI `[-0.374185, +0.000000]`, Lebar CI: `0.374185`, Memuat 0: YA)
  - **Manual Platform:**
    - STACK - RF: Point Estimate Diff = `-0.000794` | Bootstrap Median = `+0.000000` (95% CI `[-0.197902, +0.188889]`, Lebar CI: `0.386791`, Memuat 0: YA)
    - STACK - XGB: Point Estimate Diff = `-0.064611` | Bootstrap Median = `-0.061601` (95% CI `[-0.241282, +0.000000]`, Lebar CI: `0.241282`, Memuat 0: YA)

### H2. Diagnostik STACK vs Base Learner (Terverifikasi Sesuai STEP12)
* **Sumber Data:** `docs/STEP12_STACK_VS_BASE_DIAGNOSTIC.md`
* **Populasi Evaluasi:** 16 obat test held-out per platform.
* **Daftar Obat Terverifikasi (RF BENAR, STACK SALAH):**
  - **Chantest Platform (`3` obat):**
    - `disopyramide` (True HIGH): RF & XGB high, STACK int → **Kategori A** (Meta Menimpa Kedua Base Benar)
    - `loratadine` (True LOW): RF & XGB low, STACK int → **Kategori A** (Meta Menimpa Kedua Base Benar)
    - `tamoxifen` (True LOW): RF low, XGB int, STACK int → **Kategori B** (Meta Mengikuti Base Salah)
  - **Nanion Platform (`3` obat):**
    - `disopyramide` (True HIGH): RF & XGB high, STACK int → **Kategori A** (Meta Menimpa Kedua Base Benar)
    - `loratadine` (True LOW): RF & XGB low, STACK int → **Kategori A** (Meta Menimpa Kedua Base Benar)
    - `tamoxifen` (True LOW): RF low, XGB int, STACK int → **Kategori B** (Meta Mengikuti Base Salah)
  - **Manual Platform (`1` obat):**
    - `tamoxifen` (True LOW): RF low, XGB low, STACK int → **Kategori A** (Meta Menimpa Kedua Base Benar)
* **Konfirmasi Jumlah Obat Terverifikasi:** Jumlah obat di mana RF BENAR tapi STACK SALAH adalah **3 obat** pada Chantest, **3 obat** pada Nanion, dan **1 obat** pada Manual (`3 / 3 / 1`).
* **Temuan Karakteristik Meta-Learner STACK:**
  - Meta-learner menimpa KEDUA base learner yang sudah benar pada 2 senyawa (`disopyramide` dan `loratadine`) di Chantest dan Nanion.
  - Ketiga kesalahan khas STACK mengarah ke kelas **INTERMEDIATE**.
  - Senyawa `disopyramide` (True HIGH) diprediksi **HIGH secara benar oleh kedua base learner (RF dan XGB)** di ketiga platform, namun di-under-triage menjadi INTERMEDIATE oleh meta-learner STACK di Chantest dan Nanion.

### H3. Conformal RF (Terverifikasi)
* **Sumber Data:** `docs/STEP13C_RF_CONFORMAL_FIXED.md` & `results/conformal_rf_rev.csv`
* **Populasi Evaluasi:** 12 obat training chantest (kalibrasi LODO) & 16 obat test per platform.
* **12 Skor Non-Conformity $s_d$ LODO RF:** `cisapride` (`0.023518`), `dofetilide` (`0.025005`), `terfenadine` (`0.102810`), `mexiletine` (`0.102904`), `bepridil` (`0.104039`), `chlorpromazine` (`0.242225`), `quinidine` (`0.273810`), `ranolazine` (`0.322894`), `verapamil` (`0.368977`), `ondansetron` (`0.378064`), `sotalol` (`0.533987`), `diltiazem` (`0.904169`).
* **Titik Operasi Terpilih:** **Rank 11** ($\hat{q} = 0.533987$, Threshold $= 0.466013$, Nominal Coverage $= 11/13 pprox 0.846154$).
* **Performa Per Platform pada Rank 11:**
  - **Chantest:** Benar `14/16`, Over-called `2`, Under-triage `0`, Abstain `0`, Rata-rata Ukuran Set `1.0000`, Sebaran Set `{0:0, 1:16, 2:0, 3:0}`, Coverage Empiris `0.8750` vs Nominal `0.8462` (Selisih `+0.0288`).
  - **Nanion:** Benar `13/16`, Over-called `3`, Under-triage `0`, Abstain `0`, Rata-rata Ukuran Set `1.0000`, Sebaran Set `{0:0, 1:16, 2:0, 3:0}`, Coverage Empiris `0.8125` vs Nominal `0.8462` (Selisih `-0.0337`).
  - **Manual:** Benar `13/16`, Over-called `3`, Under-triage `0`, Abstain `0`, Rata-rata Ukuran Set `1.0625`, Sebaran Set `{0:0, 1:15, 2:1, 3:0}`, Coverage Empiris `0.8125` vs Nominal `0.8462` (Selisih `-0.0337`).
* **Batas Atas Rule-of-Three Under-Triage:** $3 / 16 = 0.187500$ ($18.75\%$).
* **Detail Senyawa `clarithromycin` di Manual:** Probabilitas RF ($p_{	ext{high}}=0.0$, $p_{	ext{int}}=0.476429$, $p_{	ext{low}}=0.523571$). Pada Rank 11 (Threshold $0.466013$), set conformal melebar menjadi `{intermediate, low}` (ukuran 2), aksi decision = `intermediate` (1), hasil **Benar (Correct)**. Under-triage dihapus sah lewat perluasan set, bukan aturan set kosong.

### H4. Perbandingan RF+Conformal vs STACK+Conformal
* **Sumber Data:** `docs/STEP14_CONFORMAL_COMPARISON.md` & `results/conformal_comparison_bootstrap_rev.csv`
* **Populasi Evaluasi:** 16 obat test held-out per platform, 10.000 iterasi paired bootstrap, seed 42.
* **Hasil Bootstrap Diff (RF+Conformal minus STACK+Conformal):**
  - **Chantest Platform:**
    - Selisih Proporsi Benar: Median = `+0.187500` (95% CI `[+0.000000, +0.375000]`, Memuat 0, Batas bawah = 0.0)
    - Selisih Rata-rata Ukuran Set: Median = `-0.125000` (95% CI `[-0.312500, +0.000000]`, Memuat 0, Batas atas = 0.0)
  - **Nanion Platform:**
    - Selisih Proporsi Benar: Median = `+0.125000` (95% CI `[+0.000000, +0.312500]`, Memuat 0, Batas bawah = 0.0)
    - Selisih Rata-rata Ukuran Set: Median = `-0.125000` (95% CI `[-0.312500, +0.000000]`, Memuat 0, Batas atas = 0.0)
  - **Manual Platform:**
    - Selisih Proporsi Benar: Median = `+0.125000` (95% CI `[+0.000000, +0.312500]`, Memuat 0, Batas bawah = 0.0)
    - Selisih Rata-rata Ukuran Set: Median = `-0.312500` (95% CI `[-0.562500, -0.125000]`, **TIDAK Memuat 0**)
