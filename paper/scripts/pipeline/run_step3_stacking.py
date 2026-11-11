import os
import sys
import json
import time
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd

# ASSERT NO SCALERS IMPORTED OR USED
scaler_modules = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'QuantileTransformer']
for sm in scaler_modules:
    if sm in globals() or sm in locals():
        raise RuntimeError(f"ERROR: Scaler {sm} detected in script environment!")

# Add TrainAI to path
trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS, LABEL_MAP

import sklearn
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import joblib

SEED = 42

output_lines = []
def log(msg=""):
    output_lines.append(msg)

log("==========================================")
log("=== STEP 1 — RECOVER PRIOR SEARCH SPACES ===")
log("==========================================")
log("Derived Search Spaces from `plus_ultra.ipynb` (Cell 10, Lines 1018-1035 & Cell 17, Lines 1583-1601):")
log("Search Strategy: GridSearchCV with StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)")
log("Scoring Metric:  'f1_macro'")
log("\nRandomForestClassifier Grid:")
rf_grid = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [200, 400],
    'max_depth': [None, 20]
}
log(json.dumps(rf_grid, indent=2))

log("\nXGBClassifier Grid:")
xgb_grid = {
    'learning_rate': [0.001, 0.1],
    'max_depth': [6, 10],
    'n_estimators': [300, 500]
}
log(json.dumps(xgb_grid, indent=2))

log("\n==========================================")
log("=== STEP 2 — LOAD & ASSERT DATA ===")
log("==========================================")
X_train, y_train, groups_train, r_train = load_rev('chantest', 'training', data_dir=os.path.join(trainai_dir, 'data'))
X_test,  y_test,  groups_test,  r_test  = load_rev('chantest', 'testing',  data_dir=os.path.join(trainai_dir, 'data'))

log(f"X_train shape: {X_train.shape} | Unique drugs: {groups_train.nunique()}")
log(f"X_test shape:  {X_test.shape}  | Unique drugs: {groups_test.nunique()}")

# Label inverse map for reporting
inv_label_map = {v: k for k, v in LABEL_MAP.items()}

train_class_counts_row = pd.Series(y_train).map(inv_label_map).value_counts().to_dict()
test_class_counts_row  = pd.Series(y_test).map(inv_label_map).value_counts().to_dict()

# Drug-level labels
df_train_tmp = pd.DataFrame({'drug': groups_train, 'label': y_train})
df_test_tmp  = pd.DataFrame({'drug': groups_test,  'label': y_test})
train_class_counts_drug = df_train_tmp.groupby('drug')['label'].first().map(inv_label_map).value_counts().to_dict()
test_class_counts_drug  = df_test_tmp.groupby('drug')['label'].first().map(inv_label_map).value_counts().to_dict()

log(f"\nTraining Class Distribution:")
log(f"  Row-level:  {train_class_counts_row}")
log(f"  Drug-level: {train_class_counts_drug}")
log(f"\nTesting Class Distribution:")
log(f"  Row-level:  {test_class_counts_row}")
log(f"  Drug-level: {test_class_counts_drug}")

# ASSERTIONS
train_drugs = set(groups_train.unique())
test_drugs  = set(groups_test.unique())
overlap = train_drugs.intersection(test_drugs)

log(f"\nASSERT: train_drugs INTERSECT test_drugs is empty: {'PASS' if len(overlap)==0 else 'FAIL'}")
if len(overlap) > 0:
    raise AssertionError(f"ERROR: Overlapping drugs found between train and test: {overlap}")

log(f"ASSERT: X_train columns match canonical 11 order: {'PASS' if list(X_train.columns) == CANONICAL_BIOMARKERS else 'FAIL'}")
log(f"ASSERT: X_test columns match canonical 11 order:  {'PASS' if list(X_test.columns) == CANONICAL_BIOMARKERS else 'FAIL'}")

log("\n==========================================")
log("=== STEP 3 — VERIFY GROUPED SPLITTER (StratifiedGroupKFold) ===")
log("==========================================")
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)

fold_splits = list(sgkf.split(X_train, y_train, groups=groups_train))

for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits, 1):
    tr_drugs = set(groups_train.iloc[tr_idx].unique())
    val_drugs = set(groups_train.iloc[val_idx].unique())
    fold_overlap = tr_drugs.intersection(val_drugs)
    
    tr_class_dist = pd.Series(y_train.iloc[tr_idx]).map(inv_label_map).value_counts().to_dict()
    val_class_dist = pd.Series(y_train.iloc[val_idx]).map(inv_label_map).value_counts().to_dict()
    
    log(f"\n--- Fold {fold_idx} ---")
    log(f"  Train Fold Size: {len(tr_idx)} rows ({len(tr_drugs)} drugs)")
    log(f"  Val Fold Size:   {len(val_idx)} rows ({len(val_drugs)} drugs)")
    log(f"  Train Drugs: {sorted(list(tr_drugs))}")
    log(f"  Val Drugs:   {sorted(list(val_drugs))}")
    log(f"  Train Class Dist: {tr_class_dist}")
    log(f"  Val Class Dist:   {val_class_dist}")
    log(f"  ASSERT No Drug Overlap: {'PASS' if len(fold_overlap)==0 else 'FAIL'}")
    
    if len(fold_overlap) > 0:
        raise AssertionError(f"ERROR: Fold {fold_idx} has drug overlap: {fold_overlap}")

log("\n==========================================")
log("=== STEP 4 — TUNE BASE LEARNERS ===")
log("==========================================")

# 1. RF Tuning
log("\n1. Tuning RandomForestClassifier...")
rf_base = RandomForestClassifier(random_state=SEED)
rf_search = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_grid,
    cv=sgkf,
    scoring='f1_macro',
    n_jobs=-1,
    refit=True
)
rf_search.fit(X_train, y_train, groups=groups_train)

log(f"RF Best Params: {rf_search.best_params_}")
log(f"RF Best CV Macro-F1: {rf_search.best_score_:.6f}")

# Extract per-fold scores for best RF params
best_rf_idx = rf_search.best_index_
rf_fold_scores = [rf_search.cv_results_[f'split{i}_test_score'][best_rf_idx] for i in range(4)]
log(f"RF Per-Fold CV Macro-F1: {[round(s, 6) for s in rf_fold_scores]}")

# 2. XGBoost Tuning
log("\n2. Tuning XGBClassifier...")
xgb_base = xgb.XGBClassifier(
    random_state=SEED,
    tree_method='hist',
    objective='multi:softprob',
    num_class=3
)
xgb_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_grid,
    cv=sgkf,
    scoring='f1_macro',
    n_jobs=-1,
    refit=True
)
xgb_search.fit(X_train, y_train, groups=groups_train)

log(f"XGB Best Params: {xgb_search.best_params_}")
log(f"XGB Best CV Macro-F1: {xgb_search.best_score_:.6f}")

best_xgb_idx = xgb_search.best_index_
xgb_fold_scores = [xgb_search.cv_results_[f'split{i}_test_score'][best_xgb_idx] for i in range(4)]
log(f"XGB Per-Fold CV Macro-F1: {[round(s, 6) for s in xgb_fold_scores]}")

log("\n==========================================")
log("=== STEP 5 — FIT THE STACKING ENSEMBLE ===")
log("==========================================")
log(f"scikit-learn version: {sklearn.__version__}")
log(f"xgboost version:      {xgb.__version__}")

log("\nInternal CV Grouping Implementation Note:")
log("In scikit-learn 1.8.0, StackingClassifier internal CV grouping is guaranteed by pre-generating")
log("the 4 grouped fold indices `fold_splits = list(sgkf.split(X_train, y_train, groups=groups_train))`")
log("and passing `cv=fold_splits` to StackingClassifier. This ensures out-of-fold probability predictions")
log("are computed with 100% whole-drug holdout group isolation.")

best_rf = rf_search.best_estimator_
best_xgb = xgb_search.best_estimator_

meta_rf = RandomForestClassifier(random_state=SEED)

stack_model = StackingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb)],
    final_estimator=meta_rf,
    stack_method='predict_proba',
    passthrough=False,
    cv=fold_splits
)

t0_stack = time.time()
stack_model.fit(X_train, y_train)
t1_stack = time.time()
log(f"Fitted StackingClassifier on chantest training set in {t1_stack - t0_stack:.2f} seconds.")

log("\n==========================================")
log("=== STEP 6 — EVALUATE ON CHANTEST TEST (16 HELD-OUT DRUGS) ===")
log("==========================================")

# Predict probabilities and hard labels on test set
y_test_proba = stack_model.predict_proba(X_test)
y_test_pred_row = np.argmax(y_test_proba, axis=1)

# A. ROW-LEVEL EVALUATION
row_macro_f1 = f1_score(y_test, y_test_pred_row, average='macro')
row_acc = accuracy_score(y_test, y_test_pred_row)
row_p, row_r, row_f1, _ = precision_recall_fscore_support(y_test, y_test_pred_row, labels=[0, 1, 2])
row_cm = confusion_matrix(y_test, y_test_pred_row, labels=[0, 1, 2])

log("\nA. ROW-LEVEL METRICS (32,000 test rows):")
log(f"  Macro-F1: {row_macro_f1:.6f}")
log(f"  Accuracy: {row_acc:.6f}")
log("  Per-Class Metrics:")
for cls_idx, cls_name in inv_label_map.items():
    log(f"    Class {cls_idx} ({cls_name:<12}): Precision={row_p[cls_idx]:.6f}, Recall={row_r[cls_idx]:.6f}, F1={row_f1[cls_idx]:.6f}")
log(f"  Confusion Matrix (rows=true, cols=pred):\n{row_cm}")

# B. DRUG-LEVEL EVALUATION
# Average predicted probabilities within each drug
df_eval = pd.DataFrame(y_test_proba, columns=['prob_high', 'prob_int', 'prob_low'])
df_eval['drug_name'] = groups_test.values
df_eval['true_label'] = y_test.values

drug_summary = df_eval.groupby('drug_name').agg({
    'true_label': 'first',
    'prob_high': 'mean',
    'prob_int': 'mean',
    'prob_low': 'mean'
}).reset_index()

# Drug-level argmax prediction
drug_summary['pred_label'] = np.argmax(
    drug_summary[['prob_high', 'prob_int', 'prob_low']].values, axis=1
)

drug_y_true = drug_summary['true_label'].values
drug_y_pred = drug_summary['pred_label'].values

drug_macro_f1 = f1_score(drug_y_true, drug_y_pred, average='macro')
drug_acc = accuracy_score(drug_y_true, drug_y_pred)
drug_p, drug_r, drug_f1, _ = precision_recall_fscore_support(drug_y_true, drug_y_pred, labels=[0, 1, 2])
drug_cm = confusion_matrix(drug_y_true, drug_y_pred, labels=[0, 1, 2])

log("\nB. DRUG-LEVEL METRICS (16 held-out drugs):")
log(f"  Drug-Level Macro-F1: {drug_macro_f1:.6f}")
log(f"  Drug-Level Accuracy: {drug_acc:.6f}")
log("  Per-Class Recall:")
for cls_idx, cls_name in inv_label_map.items():
    log(f"    Class {cls_idx} ({cls_name:<12}): Recall={drug_r[cls_idx]:.6f}, Precision={drug_p[cls_idx]:.6f}, F1={drug_f1[cls_idx]:.6f}")

log("\n16-ROW DRUG-LEVEL PREDICTION TABLE:")
log(f"{'Drug Name':<16} | {'True Label':<12} | {'Pred Label':<12} | {'Mean P(High)':<12} | {'Mean P(Int)':<12} | {'Mean P(Low)':<12} | Match")
log("-" * 95)
for _, row in drug_summary.iterrows():
    dr = row['drug_name']
    t_str = inv_label_map[row['true_label']]
    p_str = inv_label_map[row['pred_label']]
    match_str = "YES" if row['true_label'] == row['pred_label'] else "NO (MISSED)"
    log(f"{dr:<16} | {t_str:<12} | {p_str:<12} | {row['prob_high']:<12.6f} | {row['prob_int']:<12.6f} | {row['prob_low']:<12.6f} | {match_str}")

log("\nPRIMARY METRIC DECLARATION:")
log("Drug-Level Macro-F1 is the paper's primary metric because preclinical cardiotoxicity risk assessment")
log("(e.g., CiPA/FDA safety pharmacology guidelines) evaluates risk per compound/drug entity, not per single")
log("simulation replicate. Replicate rows within a drug are non-independent samples of the same underlying compound.")

log("\n==========================================")
log("=== STEP 7 — PERSIST THE FROZEN MODEL ===")
log("==========================================")

models_dir = os.path.join(trainai_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'chantest_stacking_model_rev.joblib')
meta_path  = os.path.join(models_dir, 'chantest_stacking_metadata_rev.json')

joblib.dump(stack_model, model_path)

metadata = {
    'model_name': 'chantest_stacking_ensemble_rev',
    'timestamp': datetime.now().isoformat(),
    'seed': SEED,
    'sklearn_version': sklearn.__version__,
    'xgboost_version': xgb.__version__,
    'label_map': LABEL_MAP,
    'canonical_biomarkers': CANONICAL_BIOMARKERS,
    'tuned_hyperparameters': {
        'rf': rf_search.best_params_,
        'xgb': xgb_search.best_params_,
        'meta_rf': {'random_state': SEED}
    },
    'cross_validation': {
        'splitter': 'StratifiedGroupKFold',
        'n_splits': 4,
        'rf_best_cv_macro_f1': float(rf_search.best_score_),
        'rf_fold_scores': [float(s) for s in rf_fold_scores],
        'xgb_best_cv_macro_f1': float(xgb_search.best_score_),
        'xgb_fold_scores': [float(s) for s in xgb_fold_scores]
    },
    'chantest_test_evaluation': {
        'row_level': {
            'macro_f1': float(row_macro_f1),
            'accuracy': float(row_acc),
            'confusion_matrix': row_cm.tolist()
        },
        'drug_level': {
            'macro_f1': float(drug_macro_f1),
            'accuracy': float(drug_acc),
            'confusion_matrix': drug_cm.tolist()
        }
    }
}

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

model_hash = compute_sha256(model_path)
meta_hash  = compute_sha256(meta_path)

log(f"Saved Model File:    {model_path} | SHA-256: {model_hash}")
log(f"Saved Metadata File: {meta_path}  | SHA-256: {meta_hash}")

log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
flags = []
if abs(drug_macro_f1 - 0.828) > 0.05:
    flags.append(f"Drug-level Macro-F1 ({drug_macro_f1:.6f}) differs noticeably from old pre-_rev benchmark (0.828).")
if any(r == 0 for r in drug_r):
    flags.append("Zero recall detected for at least one class in drug-level evaluation.")
if any(r == 0 for r in row_r):
    flags.append("Zero recall detected for at least one class in row-level evaluation.")

if flags:
    for flg in flags:
        log(f"  - SANITY FLAG: {flg}")
else:
    log("  - No sanity flags: Model training and evaluation proceeded smoothly.")

full_text = "\n".join(output_lines)
print(full_text)

# Write to docs/STEP3_STACKING_CHANTEST.md
doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP3_STACKING_CHANTEST.md')
doc_path_trainai = os.path.join(trainai_dir, 'docs', 'STEP3_STACKING_CHANTEST.md')

os.makedirs(os.path.dirname(doc_path_repo), exist_ok=True)
os.makedirs(os.path.dirname(doc_path_trainai), exist_ok=True)

with open(doc_path_repo, 'w', encoding='utf-8') as f:
    f.write(full_text)

with open(doc_path_trainai, 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"\nWritten report to:\n  1. {doc_path_repo}\n  2. {doc_path_trainai}")
