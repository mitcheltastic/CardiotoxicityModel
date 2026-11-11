import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add TrainAI to path
trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev

output_lines = []
def log(msg=""):
    output_lines.append(msg)
    print(msg)

# ASSERT NO SCALERS IMPORTED OR USED
scaler_modules = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'QuantileTransformer']
for sm in scaler_modules:
    if sm in globals() or sm in locals():
        log(f"ERROR: Scaler {sm} detected in script environment!")
        sys.exit(1)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()

model_path = os.path.join(trainai_dir, 'models', 'chantest_stacking_model_rev.joblib')
meta_path  = os.path.join(trainai_dir, 'models', 'chantest_stacking_metadata_rev.json')

log("Loading model...")
if not os.path.exists(model_path):
    log(f"ERROR: Model file not found at {model_path}")
    sys.exit(1)

model_hash = compute_sha256(model_path)
EXPECTED_HASH = "A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2"
log(f"Loaded Model SHA-256: {model_hash}")
log(f"Expected SHA-256:     {EXPECTED_HASH}")

if model_hash == EXPECTED_HASH:
    log("Hash Check: PASS")
else:
    log("Hash Check: FAIL")
    log("ERROR: The artifact changed since Step 3.")
    sys.exit(1)

stack_model = joblib.load(model_path)

log("\nLoading metadata...")
with open(meta_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

canonical_biomarkers = metadata['canonical_biomarkers']
label_map = metadata['label_map']
inv_label_map = {v: k for k, v in label_map.items()}

log("\n==========================================")
log("=== STEP 0 — SEVERITY MAP ===")
log("==========================================")
SEVERITY = {0: 2, 1: 1, 2: 0}   # high=2 (most severe), intermediate=1, low=0 (least severe)
log(f"SEVERITY = {SEVERITY}")
log("severity(pred) > severity(true)  -> OVER-CALLED   (safe direction)")
log("severity(pred) < severity(true)  -> UNDER-TRIAGED (unsafe direction)")

def get_error_direction(true_label, pred_label):
    st = SEVERITY[true_label]
    sp = SEVERITY[pred_label]
    if sp > st:
        return "OVER-CALLED"
    elif sp < st:
        return "UNDER-TRIAGED"
    return "MATCH"

def evaluate_predictions(y_true, y_proba, groups):
    y_pred_row = np.argmax(y_proba, axis=1)
    
    row_macro_f1 = f1_score(y_true, y_pred_row, average='macro')
    row_acc = accuracy_score(y_true, y_pred_row)
    
    df_eval = pd.DataFrame(y_proba, columns=['prob_high', 'prob_int', 'prob_low'])
    df_eval['drug_name'] = groups.values
    df_eval['true_label'] = y_true.values
    
    drug_summary = df_eval.groupby('drug_name').agg({
        'true_label': 'first',
        'prob_high': 'mean',
        'prob_int': 'mean',
        'prob_low': 'mean'
    }).reset_index()
    
    drug_summary['pred_label'] = np.argmax(
        drug_summary[['prob_high', 'prob_int', 'prob_low']].values, axis=1
    )
    
    drug_y_true = drug_summary['true_label'].values
    drug_y_pred = drug_summary['pred_label'].values
    
    drug_macro_f1 = f1_score(drug_y_true, drug_y_pred, average='macro')
    drug_acc = accuracy_score(drug_y_true, drug_y_pred)
    drug_p, drug_r, drug_f1, _ = precision_recall_fscore_support(drug_y_true, drug_y_pred, labels=[0, 1, 2], zero_division=0)
    drug_cm = confusion_matrix(drug_y_true, drug_y_pred, labels=[0, 1, 2])
    
    return {
        'row_macro_f1': row_macro_f1,
        'row_acc': row_acc,
        'drug_macro_f1': drug_macro_f1,
        'drug_acc': drug_acc,
        'drug_p': drug_p,
        'drug_r': drug_r,
        'drug_f1': drug_f1,
        'drug_cm': drug_cm,
        'drug_summary': drug_summary
    }

log("\n==========================================")
log("=== STEP 1 — REPRODUCTION GUARD ===")
log("==========================================")
X_test_chan, y_test_chan, groups_test_chan, _ = load_rev('chantest', 'testing', data_dir=os.path.join(trainai_dir, 'data'))

# Assert feature order
predict_features = list(X_test_chan.columns)
log(f"Predict time features: {predict_features}")
log(f"Canonical features:    {canonical_biomarkers}")
if predict_features == canonical_biomarkers:
    log("Feature Order Check: PASS")
else:
    log("Feature Order Check: FAIL")
    log("ERROR: Feature order mismatch")
    sys.exit(1)

y_proba_chan = stack_model.predict_proba(X_test_chan)
res_chan = evaluate_predictions(y_test_chan, y_proba_chan, groups_test_chan)

log(f"Expected Drug-Level Macro-F1: 0.694444")
log(f"Actual Drug-Level Macro-F1:   {res_chan['drug_macro_f1']:.6f}")
if abs(res_chan['drug_macro_f1'] - 0.694444) < 1e-5:
    log("Drug-Level F1 Check: PASS")
else:
    log("Drug-Level F1 Check: FAIL")
    log("ERROR: frozen model does not reproduce Step 3 results")
    sys.exit(1)

log(f"Expected Row-Level Macro-F1: 0.688289")
log(f"Actual Row-Level Macro-F1:   {res_chan['row_macro_f1']:.6f}")
if abs(res_chan['row_macro_f1'] - 0.688289) < 1e-5:
    log("Row-Level F1 Check: PASS")
else:
    log("Row-Level F1 Check: FAIL")
    log("ERROR: frozen model does not reproduce Step 3 results")
    sys.exit(1)

log("\n==========================================")
log("=== STEP 2 — PRIMARY: 16 HELD-OUT DRUGS, ALL THREE PLATFORMS ===")
log("==========================================")

platforms = ['chantest', 'nanion', 'manual']
test_results = {}
sanity_flags = []
under_triaged_drugs = {}

for plat in platforms:
    log(f"\n--- Platform: {plat.upper()} (TESTING) ---")
    if plat == 'chantest':
        X, y, groups = X_test_chan, y_test_chan, groups_test_chan
        res = res_chan
    else:
        X, y, groups, _ = load_rev(plat, 'testing', data_dir=os.path.join(trainai_dir, 'data'))
        if list(X.columns) != canonical_biomarkers:
            log(f"ERROR: Feature mismatch on {plat}")
            sys.exit(1)
        y_proba = stack_model.predict_proba(X)
        res = evaluate_predictions(y, y_proba, groups)
    
    test_results[plat] = res
    
    log(f"ROW-LEVEL: Macro-F1={res['row_macro_f1']:.6f}, Accuracy={res['row_acc']:.6f}")
    log(f"DRUG-LEVEL: Macro-F1={res['drug_macro_f1']:.6f}, Accuracy={res['drug_acc']:.6f}")
    
    if abs(res['drug_macro_f1'] - res_chan['drug_macro_f1']) > 0.05 and plat != 'chantest':
        sanity_flags.append(f"{plat.upper()} macro-F1 ({res['drug_macro_f1']:.6f}) differs sharply from chantest ({res_chan['drug_macro_f1']:.6f})")

    log("Per-Class Metrics:")
    for cls_idx, cls_name in inv_label_map.items():
        log(f"  Class {cls_idx} ({cls_name:<12}): Precision={res['drug_p'][cls_idx]:.6f}, Recall={res['drug_r'][cls_idx]:.6f}, F1={res['drug_f1'][cls_idx]:.6f}")
        if res['drug_r'][cls_idx] == 0:
            sanity_flags.append(f"{plat.upper()} testing has 0 recall for class {cls_name}")
            
    log(f"Confusion Matrix (rows=true, cols=pred):\n{res['drug_cm']}")
    
    log("\n16-ROW DRUG-LEVEL PREDICTION TABLE:")
    log(f"{'Drug Name':<16} | {'True Label':<12} | {'Pred Label':<12} | {'P(High)':<10} | {'P(Int)':<10} | {'P(Low)':<10} | Error Direction")
    log("-" * 95)
    
    ds = res['drug_summary']
    under_triage_count = 0
    errors_list = []
    
    for _, row in ds.iterrows():
        dr = row['drug_name']
        t_lbl = row['true_label']
        p_lbl = row['pred_label']
        
        t_str = inv_label_map[t_lbl]
        p_str = inv_label_map[p_lbl]
        
        err_dir = get_error_direction(t_lbl, p_lbl)
        
        probs = sorted([row['prob_high'], row['prob_int'], row['prob_low']], reverse=True)
        if (probs[0] - probs[1]) <= 0.05:
            sanity_flags.append(f"{plat.upper()} testing: {dr} top two probs are within 0.05 ({probs[0]:.4f}, {probs[1]:.4f})")
        
        if err_dir == "UNDER-TRIAGED":
            under_triage_count += 1
            if dr not in under_triaged_drugs:
                under_triaged_drugs[dr] = []
            under_triaged_drugs[dr].append(plat)
        
        if err_dir != "MATCH":
            errors_list.append(f"{dr}: True {t_str} -> Pred {p_str} ({err_dir})")
            
        log(f"{dr:<16} | {t_str:<12} | {p_str:<12} | {row['prob_high']:<10.6f} | {row['prob_int']:<10.6f} | {row['prob_low']:<10.6f} | {err_dir}")
        
    log(f"\nErrors: {len(errors_list)}")
    for e in errors_list:
        log(f"  - {e}")
    log(f"Explicit count of UNDER-TRIAGED drugs: {under_triage_count}")

log("\nCROSS-PLATFORM COMPARISON (Drug-Level Macro-F1):")
log(f"CHANTEST: {test_results['chantest']['drug_macro_f1']:.6f}")
log(f"NANION:   {test_results['nanion']['drug_macro_f1']:.6f}")
log(f"MANUAL:   {test_results['manual']['drug_macro_f1']:.6f}")
log("This is the paper's cross-platform result, evaluated on drugs the model never saw during training.")


log("\n==========================================")
log("=== STEP 3 — SECONDARY ===")
log("==========================================")
log("SEEN DRUGS — platform transfer only, not held-out generalization.")

train_results = {}
for plat in ['nanion', 'manual']:
    X, y, groups, _ = load_rev(plat, 'training', data_dir=os.path.join(trainai_dir, 'data'))
    y_proba = stack_model.predict_proba(X)
    res = evaluate_predictions(y, y_proba, groups)
    train_results[plat] = res
    log(f"{plat.upper()} (Seen Drugs) Drug-Level Macro-F1: {res['drug_macro_f1']:.6f}")


log("\n==========================================")
log("=== STEP 4 — DISOPYRAMIDE FOCUS (AND OTHER UNDER-TRIAGES) ===")
log("==========================================")

def extract_drug_info(ds, drug):
    row = ds[ds['drug_name'] == drug]
    if row.empty:
        return None
    return row.iloc[0]

all_ut_drugs = list(under_triaged_drugs.keys())
if 'disopyramide' not in all_ut_drugs:
    all_ut_drugs.insert(0, 'disopyramide')

for drug in all_ut_drugs:
    log(f"\n--- DRUG: {drug.upper()} ---")
    for plat in platforms:
        ds = test_results[plat]['drug_summary']
        row = extract_drug_info(ds, drug)
        if row is not None:
            t_lbl = row['true_label']
            p_lbl = row['pred_label']
            err_dir = get_error_direction(t_lbl, p_lbl)
            diff = row['prob_int'] - row['prob_high']
            
            p_str = inv_label_map[p_lbl]
            log(f"Platform: {plat.upper()}")
            log(f"  P(High): {row['prob_high']:.6f} | P(Int): {row['prob_int']:.6f} | P(Low): {row['prob_low']:.6f}")
            log(f"  Pred: {p_str} | Status: {err_dir}")
            log(f"  Difference (P(Int) - P(High)): {diff:.6f}")
            
            if err_dir == "UNDER-TRIAGED":
                sanity_flags.append(f"{drug} is under-triaged on {plat}")


log("\n==========================================")
log("=== STEP 5 — PERSIST FOR THE CONFORMAL STEP ===")
log("==========================================")
results_dir = os.path.join(trainai_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# chantest training
X_train_chan, y_train_chan, groups_train_chan, _ = load_rev('chantest', 'training', data_dir=os.path.join(trainai_dir, 'data'))
y_proba_tr_chan = stack_model.predict_proba(X_train_chan)
res_tr_chan = evaluate_predictions(y_train_chan, y_proba_tr_chan, groups_train_chan)
train_results['chantest'] = res_tr_chan

saved_files = []
for plat in platforms:
    ds_test = test_results[plat]['drug_summary'].copy()
    ds_test['split'] = 'testing'
    
    ds_train = train_results[plat]['drug_summary'].copy()
    ds_train['split'] = 'training'
    
    ds_all = pd.concat([ds_train, ds_test], ignore_index=True)
    ds_all['platform'] = plat
    
    ds_all = ds_all[['platform', 'drug_name', 'true_label', 'prob_high', 'prob_int', 'prob_low', 'pred_label', 'split']]
    
    csv_path = os.path.join(results_dir, f'cross_platform_probs_{plat}_rev.csv')
    ds_all.to_csv(csv_path, index=False)
    
    csv_hash = compute_sha256(csv_path)
    saved_files.append((csv_path, len(ds_all), csv_hash))

for path, rows, hsh in saved_files:
    log(f"Saved: {path} | Rows: {rows} | SHA-256: {hsh}")

log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
unique_flags = []
for f in sanity_flags:
    if f not in unique_flags:
        unique_flags.append(f)

if unique_flags:
    for f in unique_flags:
        log(f"  - FLAG: {f}")
else:
    log("  - No sanity flags raised.")

doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP4_CROSS_PLATFORM.md')
os.makedirs(os.path.dirname(doc_path_repo), exist_ok=True)
with open(doc_path_repo, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))
log(f"\nWritten output to {doc_path_repo}")
