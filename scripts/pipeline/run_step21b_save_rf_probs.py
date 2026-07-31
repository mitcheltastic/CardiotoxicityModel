import os
import sys
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score

# Environment setup
trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS

def log(msg=""):
    print(msg)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()

log("==========================================")
log("=== STEP 21B — SAVE RANDOM FOREST DRUG-LEVEL PROBABILITIES ===")
log("==========================================")

# ----------------------------------------------------
# LANGKAH 1 — AMBIL RF DARI STACK YANG BEKU
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 1 — LOAD FROZEN STACK MODEL & EXTRACT RF ===")
log("==========================================")

model_path = os.path.join(trainai_dir, 'models', 'chantest_stacking_model_rev.joblib')
if not os.path.exists(model_path):
    log(f"ERROR: Model file not found at {model_path}")
    sys.exit(1)

model_hash = compute_sha256(model_path)
EXPECTED_HASH = "A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2"

log(f"Loaded Model SHA-256: {model_hash}")
log(f"Expected SHA-256:     {EXPECTED_HASH}")

pass_hash = (model_hash == EXPECTED_HASH)
log(f"SHA-256 Hash Assert: {'PASS' if pass_hash else 'FAIL'}")

if not pass_hash:
    log("ERROR: Frozen model hash mismatch!")
    sys.exit(1)

stack_model = joblib.load(model_path)
rf_model = stack_model.named_estimators_['rf']

n_feat = getattr(rf_model, 'n_features_in_', None)
params = rf_model.get_params()

log(f"RF Fitted Confirmation (n_features_in_): {n_feat}")
log("RF Hyperparameters:")
for k, v in params.items():
    log(f"  {k}: {v}")

if n_feat is None or n_feat != 11:
    log("ERROR: RF model does not appear properly fitted!")
    sys.exit(1)

# ----------------------------------------------------
# LANGKAH 2 & 3 — PREDIKSI & VERIFIKASI REPRODUKSI
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 2 & 3 — PREDIKSI & VERIFIKASI REPRODUKSI ===")
log("==========================================")

platforms = ['chantest', 'nanion', 'manual']
expected_rf_f1 = {
    'chantest': 0.8777777777777778,
    'nanion':   0.8193732193732194,
    'manual':   0.7555555555555555
}

def build_rf_prob_df(platform):
    # Testing split
    X_te, y_te, grp_te, _ = load_rev(platform, 'testing', data_dir=os.path.join(trainai_dir, 'data'))
    proba_te = rf_model.predict_proba(X_te[CANONICAL_BIOMARKERS])
    df_te = pd.DataFrame(proba_te, columns=['prob_high', 'prob_int', 'prob_low'])
    df_te['drug_name'] = grp_te.values
    df_te['true_label'] = y_te.values
    
    sum_te = df_te.groupby('drug_name').agg({
        'true_label': 'first',
        'prob_high': 'mean',
        'prob_int': 'mean',
        'prob_low': 'mean'
    }).reset_index()
    sum_te['pred_label'] = np.argmax(sum_te[['prob_high', 'prob_int', 'prob_low']].values, axis=1)
    sum_te['split'] = 'testing'
    sum_te['platform'] = platform
    
    # Training split
    X_tr, y_tr, grp_tr, _ = load_rev(platform, 'training', data_dir=os.path.join(trainai_dir, 'data'))
    proba_tr = rf_model.predict_proba(X_tr[CANONICAL_BIOMARKERS])
    df_tr = pd.DataFrame(proba_tr, columns=['prob_high', 'prob_int', 'prob_low'])
    df_tr['drug_name'] = grp_tr.values
    df_tr['true_label'] = y_tr.values
    
    sum_tr = df_tr.groupby('drug_name').agg({
        'true_label': 'first',
        'prob_high': 'mean',
        'prob_int': 'mean',
        'prob_low': 'mean'
    }).reset_index()
    sum_tr['pred_label'] = np.argmax(sum_tr[['prob_high', 'prob_int', 'prob_low']].values, axis=1)
    sum_tr['split'] = 'training'
    sum_tr['platform'] = platform
    
    df_combined = pd.concat([sum_tr, sum_te], ignore_index=True)
    cols_order = ['platform', 'drug_name', 'true_label', 'prob_high', 'prob_int', 'prob_low', 'pred_label', 'split']
    df_combined = df_combined[cols_order]
    return df_combined

rf_prob_dfs = {}
repro_failed = False

for p in platforms:
    df_p = build_rf_prob_df(p)
    rf_prob_dfs[p] = df_p
    
    df_test_only = df_p[df_p['split'] == 'testing']
    y_true_drug = df_test_only['true_label'].values
    y_pred_drug = df_test_only['pred_label'].values
    
    f1_val = f1_score(y_true_drug, y_pred_drug, average='macro')
    exp_f1 = expected_rf_f1[p]
    diff = abs(f1_val - exp_f1)
    
    pass_repro = diff < 1e-6
    log(f"RF on {p:<8}: Computed Macro-F1={f1_val:.6f} | Expected={exp_f1:.6f} | Diff={diff:.2e} -> {'PASS' if pass_repro else 'FAIL'}")
    if not pass_repro:
        repro_failed = True
        log(f"ERROR: RF tidak mereproduksi hasil terdahulu (Diff = {diff:.6f})")

if repro_failed:
    log("ERROR: RF reproducibility verification failed!")
    sys.exit(1)
else:
    log("RF Reproducibility Verification: ALL PASS")

# ----------------------------------------------------
# LANGKAH 4 — SIMPAN FILE
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 4 — SIMPAN FILE PROBABILITAS RF CSV ===")
log("==========================================")

saved_rf_csv_info = {}

for p in platforms:
    df_csv = rf_prob_dfs[p]
    out_fname = f"rf_probs_{p}_rev.csv"
    out_fpath = os.path.join(trainai_dir, 'results', out_fname)
    df_csv.to_csv(out_fpath, index=False)
    
    h_val = compute_sha256(out_fpath)
    r_cnt = len(df_csv)
    saved_rf_csv_info[out_fname] = {
        'path': out_fpath,
        'rows': r_cnt,
        'sha256': h_val,
        'df': df_csv
    }
    log(f"Saved: {out_fname}")
    log(f"  Path:   {out_fpath}")
    log(f"  Rows:   {r_cnt}")
    log(f"  SHA256: {h_val}")

# ----------------------------------------------------
# LANGKAH 5 — GUARD KONSISTENSI
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 5 — GUARD KONSISTENSI ===")
log("==========================================")

ref_prob_path = os.path.join(trainai_dir, 'results', 'cross_platform_probs_chantest_rev.csv')
ref_df = pd.read_csv(ref_prob_path)
ref_drugs_all = list(ref_df['drug_name'].values)

guard_failed = False

for fname, info in saved_rf_csv_info.items():
    df = info['df']
    
    n_train = len(df[df['split'] == 'training'])
    n_test  = len(df[df['split'] == 'testing'])
    
    pass_counts = (n_train == 12 and n_test == 16 and len(df) == 28)
    log(f"File {fname:<30} Row Count Assert (28 rows: 12 train, 16 test): {'PASS' if pass_counts else 'FAIL'}")
    if not pass_counts:
        guard_failed = True
        
    prob_sums = df['prob_high'] + df['prob_int'] + df['prob_low']
    sum_check = np.allclose(prob_sums, 1.0, atol=1e-6)
    log(f"File {fname:<30} Prob Sum == 1.0 Assert:                    {'PASS' if sum_check else 'FAIL'}")
    if not sum_check:
        guard_failed = True
        
    file_drugs = list(df['drug_name'].values)
    name_check = (file_drugs == ref_drugs_all)
    log(f"File {fname:<30} Drug Names Match Assert:                 {'PASS' if name_check else 'FAIL'}")
    if not name_check:
        guard_failed = True

if guard_failed:
    log("ERROR: Consistency guard check failed!")
    sys.exit(1)
else:
    log("Consistency Guard Check: ALL PASS")

# ----------------------------------------------------
# WRITE MARKDOWN DOCUMENTATION
# ----------------------------------------------------
log("\n==========================================")
log("=== WRITING DOCS/STEP21B_SAVE_RF_PROBS.MD ===")
log("==========================================")

doc_path = os.path.join(trainai_dir, 'docs', 'STEP21B_SAVE_RF_PROBS.md')

md_content = f"""# Step 21B: Saving Drug-Level Probabilities for Random Forest Baseline Model

This document reports the extraction, reproducibility verification, and integrity audits of drug-level probability CSV files for the **Random Forest** baseline estimator extracted directly from the frozen Stacking Classifier artifact (`models/chantest_stacking_model_rev.joblib`).

---

## 1. LANGKAH 1 — Frozen Model SHA-256 Audit & RF Extraction

- **Model Artifact**: `models/chantest_stacking_model_rev.joblib`
- **SHA-256 Hash**: `{model_hash}` (**PASS**)
- **Extracted Estimator**: `stack.named_estimators_['rf']`
- **Fitted Status (`n_features_in_`)**: `{n_feat}` (**PASS**)
- **Criterion**: `entropy` | **n_estimators**: `400` | **random_state**: `42`

---

## 2. LANGKAH 2 & 3 — Reproducibility Verification

| Platform | Computed Macro-F1 | Expected Macro-F1 | Difference | Reproducibility Status |
| :--- | :---: | :---: | :---: | :---: |
"""

for p in platforms:
    df_p = rf_prob_dfs[p]
    df_test_only = df_p[df_p['split'] == 'testing']
    score = f1_score(df_test_only['true_label'].values, df_test_only['pred_label'].values, average='macro')
    exp_f1 = expected_rf_f1[p]
    diff = abs(score - exp_f1)
    md_content += f"| `{p}` | {score:.6f} | {exp_f1:.6f} | {diff:.2e} | **PASS** |\n"

md_content += """
---

## 3. LANGKAH 4 & 5 — Saved Probability CSV Audit & Integrity Guards

| File Name | Total Rows | Train Rows | Test Rows | Prob Sum = 1.0 | Drug Names Match | SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""

for fname, info in saved_rf_csv_info.items():
    df = info['df']
    n_tr = len(df[df['split'] == 'training'])
    n_te = len(df[df['split'] == 'testing'])
    prob_sums = df['prob_high'] + df['prob_int'] + df['prob_low']
    sum_ok = np.allclose(prob_sums, 1.0, atol=1e-6)
    name_ok = (list(df['drug_name'].values) == ref_drugs_all)
    md_content += f"| `{fname}` | {len(df)} | {n_tr} | {n_te} | **{'PASS' if sum_ok else 'FAIL'}** | **{'PASS' if name_ok else 'FAIL'}** | `{info['sha256']}` |\n"

md_content += """
---

## 4. SANITY FLAGS

NONE (All reproducibility diffs < 1e-6, exact drug-level probability sums = 1.0, clean integrity guards).

---
*Report generated automatically by `scripts/pipeline/run_step21b_save_rf_probs.py`.*
"""

with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

log(f"Successfully generated documentation at {doc_path}")

# ----------------------------------------------------
# SANITY FLAGS
# ----------------------------------------------------
log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
log("NONE (All reproducibility diffs < 1e-6, exact drug-level probability sums = 1.0, clean integrity guards)")

log("\n==========================================")
log("=== STEP 21B COMPLETED SUCCESSFULLY ===")
log("==========================================")
