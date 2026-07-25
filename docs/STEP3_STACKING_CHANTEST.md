==========================================
=== STEP 1 — RECOVER PRIOR SEARCH SPACES ===
==========================================
Derived Search Spaces from `plus_ultra.ipynb` (Cell 10, Lines 1018-1035 & Cell 17, Lines 1583-1601):
Search Strategy: GridSearchCV with StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
Scoring Metric:  'f1_macro'

RandomForestClassifier Grid:
{
  "criterion": [
    "gini",
    "entropy"
  ],
  "n_estimators": [
    200,
    400
  ],
  "max_depth": [
    null,
    20
  ]
}

XGBClassifier Grid:
{
  "learning_rate": [
    0.001,
    0.1
  ],
  "max_depth": [
    6,
    10
  ],
  "n_estimators": [
    300,
    500
  ]
}

==========================================
=== STEP 2 — LOAD & ASSERT DATA ===
==========================================
X_train shape: (24000, 11) | Unique drugs: 12
X_test shape:  (32000, 11)  | Unique drugs: 16

Training Class Distribution:
  Row-level:  {'high': 8000, 'intermediate': 8000, 'low': 8000}
  Drug-level: {'high': 4, 'intermediate': 4, 'low': 4}

Testing Class Distribution:
  Row-level:  {'intermediate': 14000, 'low': 10000, 'high': 8000}
  Drug-level: {'intermediate': 7, 'low': 5, 'high': 4}

ASSERT: train_drugs INTERSECT test_drugs is empty: PASS
ASSERT: X_train columns match canonical 11 order: PASS
ASSERT: X_test columns match canonical 11 order:  PASS

==========================================
=== STEP 3 — VERIFY GROUPED SPLITTER (StratifiedGroupKFold) ===
==========================================

--- Fold 1 ---
  Train Fold Size: 18000 rows (9 drugs)
  Val Fold Size:   6000 rows (3 drugs)
  Train Drugs: ['bepridil', 'chlorpromazine', 'cisapride', 'diltiazem', 'dofetilide', 'ondansetron', 'ranolazine', 'sotalol', 'verapamil']
  Val Drugs:   ['mexiletine', 'quinidine', 'terfenadine']
  Train Class Dist: {'high': 6000, 'intermediate': 6000, 'low': 6000}
  Val Class Dist:   {'low': 2000, 'high': 2000, 'intermediate': 2000}
  ASSERT No Drug Overlap: PASS

--- Fold 2 ---
  Train Fold Size: 18000 rows (9 drugs)
  Val Fold Size:   6000 rows (3 drugs)
  Train Drugs: ['bepridil', 'chlorpromazine', 'diltiazem', 'dofetilide', 'mexiletine', 'ondansetron', 'quinidine', 'ranolazine', 'terfenadine']
  Val Drugs:   ['cisapride', 'sotalol', 'verapamil']
  Train Class Dist: {'high': 6000, 'intermediate': 6000, 'low': 6000}
  Val Class Dist:   {'intermediate': 2000, 'high': 2000, 'low': 2000}
  ASSERT No Drug Overlap: PASS

--- Fold 3 ---
  Train Fold Size: 18000 rows (9 drugs)
  Val Fold Size:   6000 rows (3 drugs)
  Train Drugs: ['cisapride', 'dofetilide', 'mexiletine', 'ondansetron', 'quinidine', 'ranolazine', 'sotalol', 'terfenadine', 'verapamil']
  Val Drugs:   ['bepridil', 'chlorpromazine', 'diltiazem']
  Train Class Dist: {'intermediate': 6000, 'high': 6000, 'low': 6000}
  Val Class Dist:   {'high': 2000, 'intermediate': 2000, 'low': 2000}
  ASSERT No Drug Overlap: PASS

--- Fold 4 ---
  Train Fold Size: 18000 rows (9 drugs)
  Val Fold Size:   6000 rows (3 drugs)
  Train Drugs: ['bepridil', 'chlorpromazine', 'cisapride', 'diltiazem', 'mexiletine', 'quinidine', 'sotalol', 'terfenadine', 'verapamil']
  Val Drugs:   ['dofetilide', 'ondansetron', 'ranolazine']
  Train Class Dist: {'high': 6000, 'intermediate': 6000, 'low': 6000}
  Val Class Dist:   {'high': 2000, 'intermediate': 2000, 'low': 2000}
  ASSERT No Drug Overlap: PASS

==========================================
=== STEP 4 — TUNE BASE LEARNERS ===
==========================================

1. Tuning RandomForestClassifier...
RF Best Params: {'criterion': 'entropy', 'max_depth': None, 'n_estimators': 400}
RF Best CV Macro-F1: 0.790324
RF Per-Fold CV Macro-F1: [np.float64(0.971159), np.float64(0.705253), np.float64(0.564481), np.float64(0.920402)]

2. Tuning XGBClassifier...
XGB Best Params: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 300}
XGB Best CV Macro-F1: 0.807752
XGB Per-Fold CV Macro-F1: [np.float64(0.971789), np.float64(0.740393), np.float64(0.597456), np.float64(0.921369)]

==========================================
=== STEP 5 — FIT THE STACKING ENSEMBLE ===
==========================================
scikit-learn version: 1.8.0
xgboost version:      3.2.0

Internal CV Grouping Implementation Note:
In scikit-learn 1.8.0, StackingClassifier internal CV grouping is guaranteed by pre-generating
the 4 grouped fold indices `fold_splits = list(sgkf.split(X_train, y_train, groups=groups_train))`
and passing `cv=fold_splits` to StackingClassifier. This ensures out-of-fold probability predictions
are computed with 100% whole-drug holdout group isolation.
Fitted StackingClassifier on chantest training set in 34.30 seconds.

==========================================
=== STEP 6 — EVALUATE ON CHANTEST TEST (16 HELD-OUT DRUGS) ===
==========================================

A. ROW-LEVEL METRICS (32,000 test rows):
  Macro-F1: 0.688289
  Accuracy: 0.682344
  Per-Class Metrics:
    Class 0 (high        ): Precision=0.611978, Recall=0.823875, F1=0.702291
    Class 1 (intermediate): Precision=0.622178, Recall=0.698929, F1=0.658324
    Class 2 (low         ): Precision=0.992004, Recall=0.545900, F1=0.704251
  Confusion Matrix (rows=true, cols=pred):
[[6591 1402    7]
 [4178 9785   37]
 [   1 4540 5459]]

B. DRUG-LEVEL METRICS (16 held-out drugs):
  Drug-Level Macro-F1: 0.694444
  Drug-Level Accuracy: 0.687500
  Per-Class Recall:
    Class 0 (high        ): Recall=0.750000, Precision=0.600000, F1=0.666667
    Class 1 (intermediate): Recall=0.714286, Precision=0.625000, F1=0.666667
    Class 2 (low         ): Recall=0.600000, Precision=1.000000, F1=0.750000

16-ROW DRUG-LEVEL PREDICTION TABLE:
Drug Name        | True Label   | Pred Label   | Mean P(High) | Mean P(Int)  | Mean P(Low)  | Match
-----------------------------------------------------------------------------------------------
astemizole       | intermediate | intermediate | 0.039572     | 0.960353     | 0.000075     | YES
azimilide        | high         | high         | 0.989995     | 0.000000     | 0.010005     | YES
clarithromycin   | intermediate | intermediate | 0.015209     | 0.984211     | 0.000580     | YES
clozapine        | intermediate | intermediate | 0.078595     | 0.897620     | 0.023785     | YES
disopyramide     | high         | intermediate | 0.438730     | 0.495815     | 0.065455     | NO (MISSED)
domperidone      | intermediate | high         | 0.986030     | 0.003185     | 0.010785     | NO (MISSED)
droperidol       | intermediate | intermediate | 0.057251     | 0.941604     | 0.001145     | YES
ibutilide        | high         | high         | 0.911960     | 0.048745     | 0.039295     | YES
loratadine       | low          | intermediate | 0.148550     | 0.629450     | 0.222000     | NO (MISSED)
metoprolol       | low          | low          | 0.003472     | 0.262423     | 0.734105     | YES
nifedipine       | low          | low          | 0.000000     | 0.000000     | 1.000000     | YES
nitrendipine     | low          | low          | 0.029895     | 0.089335     | 0.880770     | YES
pimozide         | intermediate | high         | 0.944341     | 0.046999     | 0.008660     | NO (MISSED)
risperidone      | intermediate | intermediate | 0.006530     | 0.978645     | 0.014825     | YES
tamoxifen        | low          | intermediate | 0.008780     | 0.930675     | 0.060545     | NO (MISSED)
vandetanib       | high         | high         | 0.927865     | 0.052095     | 0.020040     | YES

PRIMARY METRIC DECLARATION:
Drug-Level Macro-F1 is the paper's primary metric because preclinical cardiotoxicity risk assessment
(e.g., CiPA/FDA safety pharmacology guidelines) evaluates risk per compound/drug entity, not per single
simulation replicate. Replicate rows within a drug are non-independent samples of the same underlying compound.

==========================================
=== STEP 7 — PERSIST THE FROZEN MODEL ===
==========================================
Saved Model File:    d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\models\chantest_stacking_model_rev.joblib | SHA-256: a93b9868fde30c7c93b2fb86ec5c1e594c0aa664f8f700c96bb7d9e6e7f2a5d2
Saved Metadata File: d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\models\chantest_stacking_metadata_rev.json  | SHA-256: da0f2a43929e2bff51230e0466513af14ab2c9635ffb8ef3b49ba5d81849ed01

==========================================
=== SANITY FLAGS ===
==========================================
  - SANITY FLAG: Drug-level Macro-F1 (0.694444) differs noticeably from old pre-_rev benchmark (0.828).
## VERIFICATION

### 1. RUN INTEGRITY
**Metadata File (models/chantest_stacking_metadata_rev.json) Contents:**
`json
{
  "model_name": "chantest_stacking_ensemble_rev",
  "timestamp": "2026-07-25T09:43:05.320278",
  "seed": 42,
  "sklearn_version": "1.8.0",
  "xgboost_version": "3.2.0",
  "label_map": {
    "high": 0,
    "intermediate": 1,
    "low": 2
  },
  "canonical_biomarkers": [
    "qNet",
    "dvdtmax",
    "vmax",
    "vrest",
    "APD50",
    "APD90",
    "max_dv",
    "camax",
    "carest",
    "CaTD50",
    "CaTD90"
  ],
  "tuned_hyperparameters": {
    "rf": {
      "criterion": "entropy",
      "max_depth": null,
      "n_estimators": 400
    },
    "xgb": {
      "learning_rate": 0.1,
      "max_depth": 10,
      "n_estimators": 300
    },
    "meta_rf": {
      "random_state": 42
    }
  },
  "cross_validation": {
    "splitter": "StratifiedGroupKFold",
    "n_splits": 4,
    "rf_best_cv_macro_f1": 0.790323807415431,
    "rf_fold_scores": [
      0.9711592956332437,
      0.7052528864683562,
      0.5644808360558785,
      0.9204022115042455
    ],
    "xgb_best_cv_macro_f1": 0.8077516997015638,
    "xgb_fold_scores": [
      0.9717885639534011,
      0.7403933375717463,
      0.5974560246346657,
      0.921368872646442
    ]
  },
  "chantest_test_evaluation": {
    "row_level": {
      "macro_f1": 0.688288587457072,
      "accuracy": 0.68234375,
      "confusion_matrix": [
        [6591, 1402, 7],
        [4178, 9785, 37],
        [1, 4540, 5459]
      ]
    },
    "drug_level": {
      "macro_f1": 0.6944444444444443,
      "accuracy": 0.6875,
      "confusion_matrix": [
        [3, 1, 0],
        [2, 5, 0],
        [0, 2, 3]
      ]
    }
  }
}
`

**File Stats:**
- chantest_stacking_metadata_rev.json: 1915 bytes, Last Modified: 2026-07-25 09:43:05 AM
  - SHA-256: DA0F2A43929E2BFF51230E0466513AF14AB2C9635FFB8EF3B49BA5D81849ED01
- chantest_stacking_model_rev.joblib: 8690060 bytes, Last Modified: 2026-07-25 09:43:05 AM
  - SHA-256: A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2

**Timestamp Analysis:**
The timestamps on both files are identical down to the second (9:43:05 AM). This is consistent with ONE single completed run, and neither file predates the other.
*(Note: docs/STEP3_STACKING_CHANTEST.md in its original form is documented prior to this section).*

### 2. STACK INTERNAL CV GROUPING
- **Installed sklearn version**: 1.8.0
- **How groups were obtained**: StackingClassifier internal CV grouping is guaranteed by pre-generating the 4 grouped fold indices using StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42) and passing the list of splits explicitly to cv=fold_splits.
- **Code from scripts/pipeline/run_step3_stacking.py**:
`python
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
fold_splits = list(sgkf.split(X_train, y_train, groups=groups_train))
# ...
stack_model = StackingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb)],
    final_estimator=meta_rf,
    stack_method='predict_proba',
    passthrough=False,
    cv=fold_splits
)
`
- **Were the meta-learner's out-of-fold predictions generated with folds that never split a single drug?**: YES. StratifiedGroupKFold guarantees no drug overlap between train and validation indices, and passing the pre-computed splits to cv ensures the StackingClassifier strictly adheres to these group boundaries.

### 3. SEARCH SPACE PROVENANCE
- **RF Grid**: {'criterion': ['gini', 'entropy'], 'n_estimators': [200, 400], 'max_depth': [None, 20]}
- **XGB Grid**: {'learning_rate': [0.001, 0.1], 'max_depth': [6, 10], 'n_estimators': [300, 500]}
- **Source**: These were directly derived from plus_ultra.ipynb (Cell 10, Lines 1018-1035 & Cell 17, Lines 1583-1601).
- **Per-fold CV Scores**:
  - RF: [0.971159, 0.705253, 0.564481, 0.920402]
  - XGB: [0.971789, 0.740393, 0.597456, 0.921369]

### 4. LABEL THE METRICS PROPERLY
| Metric | Value | Evaluation Population |
| :--- | :--- | :--- |
| (a) RF CV macro-F1 | 0.790324 | Out-of-fold CV over the 12 training drugs |
| (b) XGB CV macro-F1 | 0.807752 | Out-of-fold CV over the 12 training drugs |
| (c) Stack ROW-level macro-F1 | 0.688289 | 32,000 independent rows over the 16 held-out test drugs |
| (d) Stack DRUG-level macro-F1 | 0.694444 | 16 held-out test drugs aggregated |

### 5. WHERE THE PERFORMANCE ACTUALLY WENT
**Drug-Level Per-Class Results:**
- Class 0 (high): Precision=0.600000, Recall=0.750000, F1=0.666667
- Class 1 (intermediate): Precision=0.625000, Recall=0.714286, F1=0.666667
- Class 2 (low): Precision=1.000000, Recall=0.600000, F1=0.750000

**Confusion Matrix (True \ Pred):**
`
[[3 1 0]
 [2 5 0]
 [0 2 3]]
`

**16-Row Table:**
`
Drug Name        | True Label   | Pred Label   | Mean P(High) | Mean P(Int)  | Mean P(Low)  | Match
-----------------------------------------------------------------------------------------------
astemizole       | intermediate | intermediate | 0.039572     | 0.960353     | 0.000075     | YES
azimilide        | high         | high         | 0.989995     | 0.000000     | 0.010005     | YES
clarithromycin   | intermediate | intermediate | 0.015209     | 0.984211     | 0.000580     | YES
clozapine        | intermediate | intermediate | 0.078595     | 0.897620     | 0.023785     | YES
disopyramide     | high         | intermediate | 0.438730     | 0.495815     | 0.065455     | NO (MISSED)
domperidone      | intermediate | high         | 0.986030     | 0.003185     | 0.010785     | NO (MISSED)
droperidol       | intermediate | intermediate | 0.057251     | 0.941604     | 0.001145     | YES
ibutilide        | high         | high         | 0.911960     | 0.048745     | 0.039295     | YES
loratadine       | low          | intermediate | 0.148550     | 0.629450     | 0.222000     | NO (MISSED)
metoprolol       | low          | low          | 0.003472     | 0.262423     | 0.734105     | YES
nifedipine       | low          | low          | 0.000000     | 0.000000     | 1.000000     | YES
nitrendipine     | low          | low          | 0.029895     | 0.089335     | 0.880770     | YES
pimozide         | intermediate | high         | 0.944341     | 0.046999     | 0.008660     | NO (MISSED)
risperidone      | intermediate | intermediate | 0.006530     | 0.978645     | 0.014825     | YES
tamoxifen        | low          | intermediate | 0.008780     | 0.930675     | 0.060545     | NO (MISSED)
vandetanib       | high         | high         | 0.927865     | 0.052095     | 0.020040     | YES
`

**Loss Analysis:**
The low class lost the most in recall (60%, 2/5 errors), but intermediate lost the most precision (true low and true high both incorrectly predicted as intermediate). 
Errors (Counts: 5 out of 16 drugs missed):
- disopyramide: true high, pred intermediate (Under-called)
- domperidone: true intermediate, pred high (Over-called)
- loratadine: true low, pred intermediate (Over-called)
- pimozide: true intermediate, pred high (Over-called)
- 	amoxifen: true low, pred intermediate (Over-called)

### 6. CORRECT THE STALE NARRATIVE
The sentence claiming "fixing data leakage" or "inflated" was *not* written in STEP3_STACKING_CHANTEST.md. The only claim present in the document is:
"SANITY FLAG: Drug-level Macro-F1 (0.694444) differs noticeably from old pre-_rev benchmark (0.828)."

Therefore, there is no stale claim to replace in the document itself. To clarify the narrative nonetheless, here is the corrected context:
"The 12/16 grouped-CV protocol is unchanged from the previous run; the only change is the input data (control-normalized _rev inputs, no scaler). RF and XGB are scale-invariant, so removing StandardScaler is not itself expected to change performance. The cause of the difference has not been established and is not asserted here."
