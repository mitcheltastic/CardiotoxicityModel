import os
import sys
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Environment setup
trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS

log_messages = []
def log(msg=""):
    log_messages.append(str(msg))
    print(msg)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()

SEED = 42

log("==========================================")
log("=== STEP 21A — SAVE LR DRUG-LEVEL PROBABILITIES ===")
log("==========================================")

# ----------------------------------------------------
# LANGKAH 1 — CEK APAKAH MODELNYA TERSIMPAN
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 1 — SCAN SAVED LR MODELS ===")
log("==========================================")

models_dir = os.path.join(trainai_dir, 'models')
existing_files = os.listdir(models_dir) if os.path.exists(models_dir) else []
lr_model_files = [f for f in existing_files if f.endswith('.joblib') and ('logistic' in f.lower() or 'lr' in f.lower())]

log(f"Files found in models/ directory: {existing_files}")
log(f"LR model files found (*logistic*.joblib or *lr*.joblib): {lr_model_files}")

model_a = None
model_b = None

if len(lr_model_files) >= 2:
    log("Found pre-existing LR models. Loading without retraining...")
    path_a = os.path.join(models_dir, 'logistic_multivariate_rev.joblib')
    path_b = os.path.join(models_dir, 'logistic_qnet_rev.joblib')
    if os.path.exists(path_a) and os.path.exists(path_b):
        model_a = joblib.load(path_a)
        model_b = joblib.load(path_b)
        log(f"Loaded Model A from {path_a} (SHA256: {compute_sha256(path_a)})")
        log(f"Loaded Model B from {path_b} (SHA256: {compute_sha256(path_b)})")
    else:
        log("Expected specific LR model files not found. Proceeding to Step 2 fit...")

# ----------------------------------------------------
# LANGKAH 2 — LATIH ULANG DENGAN KONFIGURASI PERSIS SAMA (JIKA TIDAK TERSIMPAN)
# ----------------------------------------------------
if model_a is None or model_b is None:
    log("\n==========================================")
    log("=== LANGKAH 2 — FIT MODELS WITH C=0.01 ===")
    log("==========================================")
    
    # Load chantest training data
    X_tr_chan, y_tr_chan, grp_tr_chan, _ = load_rev('chantest', 'training', data_dir=os.path.join(trainai_dir, 'data'))
    
    # Model A: Multivariate LR on 11 biomarkers with C=0.01, max_iter=5000, random_state=42
    log("Training Model A (Multivariate LR, 11 biomarkers, C=0.01, max_iter=5000, seed=42)...")
    model_a = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED, solver='lbfgs')
    model_a.fit(X_tr_chan[CANONICAL_BIOMARKERS], y_tr_chan)
    
    # Model B: Single-biomarker LR on qNet only with C=0.01, max_iter=5000, random_state=42
    log("Training Model B (qNet-only LR, 1 feature, C=0.01, max_iter=5000, seed=42)...")
    model_b = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED, solver='lbfgs')
    model_b.fit(X_tr_chan[['qNet']], y_tr_chan)
    
    # Save models
    path_a = os.path.join(models_dir, 'logistic_multivariate_rev.joblib')
    path_b = os.path.join(models_dir, 'logistic_qnet_rev.joblib')
    
    joblib.dump(model_a, path_a)
    joblib.dump(model_b, path_b)
    
    hash_a = compute_sha256(path_a)
    hash_b = compute_sha256(path_b)
    
    log(f"Saved Model A to: {path_a}")
    log(f"  SHA-256: {hash_a}")
    log(f"Saved Model B to: {path_b}")
    log(f"  SHA-256: {hash_b}")

# ----------------------------------------------------
# LANGKAH 3 — VERIFIKASI REPRODUKSI
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 3 — VERIFIKASI REPRODUKSI STEP 19 ===")
log("==========================================")

expected_scores = {
    'Model A': {
        'chantest': 0.8193732193732194,
        'nanion':   0.7547619047619047,
        'manual':   0.8193732193732194
    },
    'Model B': {
        'chantest': 0.8119658119658121,
        'nanion':   0.7566137566137566,
        'manual':   0.7515151515151516
    }
}

platforms = ['chantest', 'nanion', 'manual']

def evaluate_drug_level(model, features, platform):
    X_te, y_te, grp_te, _ = load_rev(platform, 'testing', data_dir=os.path.join(trainai_dir, 'data'))
    proba_rows = model.predict_proba(X_te[features])
    
    df_eval = pd.DataFrame(proba_rows, columns=['prob_high', 'prob_int', 'prob_low'])
    df_eval['drug_name'] = grp_te.values
    df_eval['true_label'] = y_te.values
    
    drug_summary = df_eval.groupby('drug_name').agg({
        'true_label': 'first',
        'prob_high': 'mean',
        'prob_int': 'mean',
        'prob_low': 'mean'
    }).reset_index()
    
    pred_drug = np.argmax(drug_summary[['prob_high', 'prob_int', 'prob_low']].values, axis=1)
    y_true_drug = drug_summary['true_label'].values
    
    score = f1_score(y_true_drug, pred_drug, average='macro')
    return score, drug_summary

repro_failed = False

for m_name, m_obj, feats in [('Model A', model_a, CANONICAL_BIOMARKERS), ('Model B', model_b, ['qNet'])]:
    for p in platforms:
        score, _ = evaluate_drug_level(m_obj, feats, p)
        exp_score = expected_scores[m_name][p]
        diff = abs(score - exp_score)
        pass_repro = diff < 1e-6
        log(f"{m_name} on {p:<8}: Computed={score:.6f} | Expected={exp_score:.6f} | Diff={diff:.2e} -> {'PASS' if pass_repro else 'FAIL'}")
        if not pass_repro:
            repro_failed = True
            log(f"ERROR: model tidak mereproduksi hasil Step 19 (Diff = {diff:.6f})")

if repro_failed:
    log("ERROR: Reproducibility check failed! Stopping.")
    sys.exit(1)
else:
    log("Step 19 Reproducibility Verification: ALL PASS")

# ----------------------------------------------------
# LANGKAH 4 — SIMPAN PROBABILITAS
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 4 — SIMPAN FILE PROBABILITAS CSV ===")
log("==========================================")

def build_full_prob_csv(model, features, platform):
    # Testing split
    X_te, y_te, grp_te, _ = load_rev(platform, 'testing', data_dir=os.path.join(trainai_dir, 'data'))
    proba_te = model.predict_proba(X_te[features])
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
    
    # Training split
    X_tr, y_tr, grp_tr, _ = load_rev(platform, 'training', data_dir=os.path.join(trainai_dir, 'data'))
    proba_tr = model.predict_proba(X_tr[features])
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
    
    # Combine training (12 drugs) and testing (16 drugs)
    sum_tr['platform'] = platform
    sum_te['platform'] = platform
    
    df_combined = pd.concat([sum_tr, sum_te], ignore_index=True)
    cols_order = ['platform', 'drug_name', 'true_label', 'prob_high', 'prob_int', 'prob_low', 'pred_label', 'split']
    df_combined = df_combined[cols_order]
    return df_combined

saved_csv_info = {}

for m_key, m_obj, feats, fname_prefix in [
    ('logistic_multivariate', model_a, CANONICAL_BIOMARKERS, 'logistic_multivariate_probs'),
    ('logistic_qnet', model_b, ['qNet'], 'logistic_qnet_probs')
]:
    for p in platforms:
        df_csv = build_full_prob_csv(m_obj, feats, p)
        out_fname = f"{fname_prefix}_{p}_rev.csv"
        out_fpath = os.path.join(trainai_dir, 'results', out_fname)
        df_csv.to_csv(out_fpath, index=False)
        
        h_val = compute_sha256(out_fpath)
        r_cnt = len(df_csv)
        saved_csv_info[out_fname] = {
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

# Load reference drug names order from cross_platform_probs_chantest_rev.csv
ref_prob_path = os.path.join(trainai_dir, 'results', 'cross_platform_probs_chantest_rev.csv')
ref_df = pd.read_csv(ref_prob_path)
ref_drugs_tr = list(ref_df[ref_df['split'] == 'training']['drug_name'].values)
ref_drugs_te = list(ref_df[ref_df['split'] == 'testing']['drug_name'].values)
ref_drugs_all = list(ref_df['drug_name'].values)

guard_failed = False

for fname, info in saved_csv_info.items():
    df = info['df']
    
    n_train = len(df[df['split'] == 'training'])
    n_test  = len(df[df['split'] == 'testing'])
    
    pass_counts = (n_train == 12 and n_test == 16)
    log(f"File {fname:<45} Row Count Assert (12 train, 16 test): {'PASS' if pass_counts else 'FAIL'}")
    if not pass_counts:
        guard_failed = True
        
    prob_sums = df['prob_high'] + df['prob_int'] + df['prob_low']
    sum_check = np.allclose(prob_sums, 1.0, atol=1e-6)
    log(f"File {fname:<45} Prob Sum == 1.0 Assert:       {'PASS' if sum_check else 'FAIL'}")
    if not sum_check:
        guard_failed = True
        
    file_drugs = list(df['drug_name'].values)
    name_check = (file_drugs == ref_drugs_all)
    log(f"File {fname:<45} Drug Names Match Assert:    {'PASS' if name_check else 'FAIL'}")
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
log("=== WRITING DOCS/STEP21A_SAVE_LR_PROBS.MD ===")
log("==========================================")

doc_path = os.path.join(trainai_dir, 'docs', 'STEP21A_SAVE_LR_PROBS.md')

md_content = f"""# Step 21A: Saving Drug-Level Probabilities for Logistic Regression Baseline Models

This document details the generation, verification, and integrity audits of drug-level probability CSV files for both baseline Logistic Regression models (**Model A: Multivariate** and **Model B: qNet-only**).

---

## 1. LANGKAH 1 & 2 — Model Training & Persistence

- **Model A (Multivariate LR, 11 biomarkers)**: `LogisticRegression(C=0.01, max_iter=5000, random_state=42, solver='lbfgs')`
  - Saved to: `models/logistic_multivariate_rev.joblib`
  - SHA-256: `{compute_sha256(os.path.join(models_dir, 'logistic_multivariate_rev.joblib'))}`
- **Model B (qNet-only LR, 1 feature)**: `LogisticRegression(C=0.01, max_iter=5000, random_state=42, solver='lbfgs')`
  - Saved to: `models/logistic_qnet_rev.joblib`
  - SHA-256: `{compute_sha256(os.path.join(models_dir, 'logistic_qnet_rev.joblib'))}`

---

## 2. LANGKAH 3 — Reproducibility Verification against Step 19

| Model | Platform | Computed Macro-F1 | Expected Step 19 Macro-F1 | Diff | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""

for m_name, m_obj, feats in [('Model A', model_a, CANONICAL_BIOMARKERS), ('Model B', model_b, ['qNet'])]:
    for p in platforms:
        score, _ = evaluate_drug_level(m_obj, feats, p)
        exp_score = expected_scores[m_name][p]
        diff = abs(score - exp_score)
        md_content += f"| {m_name} | `{p}` | {score:.6f} | {exp_score:.6f} | {diff:.2e} | **PASS** |\n"

md_content += """
---

## 3. LANGKAH 4 & 5 — Saved Probability CSV Audit & Integrity Guards

| File Name | Rows | Training Rows | Testing Rows | Prob Sum = 1.0 | Name Match | SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""

for fname, info in saved_csv_info.items():
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
*Report generated automatically by `scripts/pipeline/run_step21a_save_lr_probs.py`.*
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
log("=== STEP 21A COMPLETED SUCCESSFULLY ===")
log("==========================================")
