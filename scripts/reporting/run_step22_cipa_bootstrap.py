import os
import sys
import hashlib
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

# Environment setup
TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import load_rev

RESULTS_DIR = os.path.join(TRAINAI_DIR, 'results')
DOCS_DIR = os.path.join(TRAINAI_DIR, 'docs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

EXPECTED_STACK_HASH = "A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2"

# ----------------------------------------------------
# 1. HASH ASSERTION & MODEL LOADING
# ----------------------------------------------------
stack_path = os.path.join(TRAINAI_DIR, 'models', 'chantest_stacking_model_rev.joblib')
if not os.path.exists(stack_path):
    print(f"ERROR: Model file not found at {stack_path}")
    sys.exit(1)

with open(stack_path, 'rb') as f:
    stack_bytes = f.read()
actual_hash = hashlib.sha256(stack_bytes).hexdigest().upper()

print(f"Checking Stacking model hash: {actual_hash}")
if actual_hash != EXPECTED_STACK_HASH:
    print(f"ERROR: Stack model hash mismatch! Expected {EXPECTED_STACK_HASH}, got {actual_hash}")
    sys.exit(1)
print("Stack model hash assertion: PASS")

stack_model = joblib.load(stack_path)
rf_model = stack_model.named_estimators_['rf']
lr_multi_model = joblib.load(os.path.join(TRAINAI_DIR, 'models', 'logistic_multivariate_rev.joblib'))
lr_qnet_model = joblib.load(os.path.join(TRAINAI_DIR, 'models', 'logistic_qnet_rev.joblib'))

models_dict = {
    'Stacking': stack_model,
    'RF': rf_model,
    'LR': lr_multi_model,
    'LR-qNet': lr_qnet_model
}

platforms = ['chantest', 'nanion', 'manual']

# ----------------------------------------------------
# 2. ROW-LEVEL PROBABILITIES PREDICTION & SAVE
# ----------------------------------------------------
print("\n=== PREDICTING & SAVING ROW-LEVEL PROBABILITIES ===")
rowlevel_probs = {}
dataset_cache = {}

for p_name in platforms:
    rowlevel_probs[p_name] = {}
    X, y, groups, report = load_rev(p_name, 'testing')
    dataset_cache[p_name] = (X, y, groups, report)
    
    for m_name, model in models_dict.items():
        if m_name == 'LR-qNet':
            probs = model.predict_proba(X[['qNet']])
        else:
            probs = model.predict_proba(X)
            
        rowlevel_probs[p_name][m_name] = probs
        
        npz_fname = f"rowlevel_probs_{m_name}_{p_name}_rev.npz"
        npz_path = os.path.join(RESULTS_DIR, npz_fname)
        np.savez_compressed(npz_path, probs=probs, y=y.values, groups=groups.values)
        size_kb = os.path.getsize(npz_path) / 1024.0
        print(f"  Saved {npz_fname:<35} | Shape: {probs.shape} | Size: {size_kb:.1f} KB")

# ----------------------------------------------------
# 3. POINT ESTIMATE GUARD CHECK (DRUG-LEVEL MEAN)
# ----------------------------------------------------
print("\n=== POINT ESTIMATE GUARD CHECK (DRUG-LEVEL MEAN) ===")
expected_f1 = {
    'Stacking': {'chantest': 0.694444, 'nanion': 0.621032, 'manual': 0.754762},
    'RF':       {'chantest': 0.877778, 'nanion': 0.819373, 'manual': 0.755556},
    'LR':       {'chantest': 0.819373, 'nanion': 0.754762, 'manual': 0.819373},
    'LR-qNet':  {'chantest': 0.811966, 'nanion': 0.756614, 'manual': 0.751515}
}

guard_passed = True
for m_name in models_dict.keys():
    for p_name in platforms:
        X, y, groups, _ = dataset_cache[p_name]
        probs = rowlevel_probs[p_name][m_name]
        
        df_p = pd.DataFrame(probs, columns=['prob_high', 'prob_int', 'prob_low'])
        df_p['drug_name'] = groups.values
        df_p['true_label'] = y.values
        
        drug_df = df_p.groupby('drug_name').mean()
        y_true_drug = drug_df['true_label'].astype(int).values
        probs_drug = drug_df[['prob_high', 'prob_int', 'prob_low']].values
        y_pred_drug = np.argmax(probs_drug, axis=1)
        
        calc_f1 = f1_score(y_true_drug, y_pred_drug, average='macro')
        exp_f1 = expected_f1[m_name][p_name]
        diff = abs(calc_f1 - exp_f1)
        pass_f1 = diff < 1e-4
        
        status_str = "PASS" if pass_f1 else "FAIL"
        print(f"  {m_name:<10} {p_name:<8} | Calc: {calc_f1:.6f} | Exp: {exp_f1:.6f} | Diff: {diff:.2e} -> {status_str}")
        if not pass_f1:
            guard_passed = False

if not guard_passed:
    print("\nERROR: Drug-level point estimate guard check failed! STOPPING.")
    sys.exit(1)
print("Guard Final Status: ALL PASS")

# ----------------------------------------------------
# 4. CiPA BOOTSTRAP PROTOCOL (10,000 ITERATIONS)
# ----------------------------------------------------
print("\n=== RUNNING CiPA BOOTSTRAP PROTOCOL (10,000 ITERATIONS) ===")

N_BOOTSTRAP = 10000
SEED = 42

def calc_auc_fast(pos_mat, score_mat):
    is_pos = pos_mat[:, :, None]
    is_neg = (~pos_mat)[:, None, :]
    s_pos = score_mat[:, :, None]
    s_neg = score_mat[:, None, :]
    
    pair_mask = is_pos & is_neg
    win = (s_pos > s_neg).astype(float) + 0.5 * (s_pos == s_neg).astype(float)
    
    n_pos = np.sum(is_pos, axis=(1, 2))
    n_neg = np.sum(is_neg, axis=(1, 2))
    n_pairs = np.sum(pair_mask, axis=(1, 2))
    
    valid = (n_pos > 0) & (n_neg > 0) & (n_pairs > 0)
    res = np.full(pos_mat.shape[0], np.nan)
    res[valid] = np.sum(win * pair_mask, axis=(1, 2))[valid] / n_pairs[valid]
    return res

def format_cell(arr):
    # arr: length 10000 array of floats or infs/nans
    finite_mask = np.isfinite(arr)
    def_count = np.sum(finite_mask)
    def_pct = (def_count / len(arr)) * 100.0
    
    if def_count == 0:
        return "N/A", def_pct, np.nan, np.nan, np.nan
        
    vals = arr[finite_mask]
    med = np.median(vals)
    lo = np.percentile(vals, 2.5)
    hi = np.percentile(vals, 97.5)
    
    def fmt_num(v):
        if np.isposinf(v):
            return "inf"
        elif np.isneginf(v):
            return "-inf"
        else:
            return f"{v:.3f}"
            
    res_str = f"{fmt_num(med)} ({fmt_num(lo)}-{fmt_num(hi)})"
    if def_pct < 100.0:
        res_str += f" [terdefinisi di {def_pct:.1f}% iterasi]"
        
    return res_str, def_pct, med, lo, hi

metric_rows = [
    'Macro-F1',
    'AUC-High', 'AUC-Int', 'AUC-Low',
    'Accuracy-High', 'Accuracy-Int', 'Accuracy-Low',
    'LR+-High', 'LR+-Int', 'LR+-Low',
    'LR--High', 'LR--Int', 'LR--Low'
]

results_all = []
sanity_flags = []

for p_name in platforms:
    X, y, groups, _ = dataset_cache[p_name]
    unique_drugs = groups.unique()
    drug_row_indices = {d: np.where(groups.values == d)[0] for d in unique_drugs}
    
    rng = np.random.RandomState(SEED)
    boot_indices = np.zeros((N_BOOTSTRAP, 16), dtype=int)
    for b in range(N_BOOTSTRAP):
        for i, d in enumerate(unique_drugs):
            boot_indices[b, i] = rng.choice(drug_row_indices[d])
            
    y_16 = np.array([y.values[drug_row_indices[d][0]] for d in unique_drugs])
    ytrue_10k = np.tile(y_16, (N_BOOTSTRAP, 1))
    
    platform_res = {m: {} for m in models_dict.keys()}
    
    for m_name in models_dict.keys():
        probs = rowlevel_probs[p_name][m_name]
        probs_10k = probs[boot_indices]           # (10000, 16, 3)
        ypred_10k = np.argmax(probs_10k, axis=2) # (10000, 16)
        
        # Macro F1
        counts_mat = np.zeros((N_BOOTSTRAP, 9), dtype=int)
        for c_t in range(3):
            for c_p in range(3):
                counts_mat[:, c_t * 3 + c_p] = np.sum((ytrue_10k == c_t) & (ypred_10k == c_p), axis=1)
        cm_mat = counts_mat.reshape(N_BOOTSTRAP, 3, 3)
        tp_mat = np.diagonal(cm_mat, axis1=1, axis2=2)
        fp_mat = np.sum(cm_mat, axis=1) - tp_mat
        fn_mat = np.sum(cm_mat, axis=2) - tp_mat
        
        with np.errstate(divide='ignore', invalid='ignore'):
            prec_m = np.where(tp_mat + fp_mat > 0, tp_mat / (tp_mat + fp_mat), 0.0)
            rec_m  = np.where(tp_mat + fn_mat > 0, tp_mat / (tp_mat + fn_mat), 0.0)
            f1_m   = np.where(prec_m + rec_m > 0, 2 * prec_m * rec_m / (prec_m + rec_m), 0.0)
            macro_f1_10k = np.mean(f1_m, axis=1)
            
        platform_res[m_name]['Macro-F1'] = macro_f1_10k
        
        class_names_map = {0: 'High', 1: 'Int', 2: 'Low'}
        for c in range(3):
            c_name = class_names_map[c]
            
            # AUC One-vs-Rest
            pos_c = (ytrue_10k == c)
            score_c = probs_10k[:, :, c]
            auc_c_10k = calc_auc_fast(pos_c, score_c)
            platform_res[m_name][f'AUC-{c_name}'] = auc_c_10k
            
            # Confusion matrix metrics for class c
            tp_c = tp_mat[:, c]
            fp_c = fp_mat[:, c]
            fn_c = fn_mat[:, c]
            tn_c = 16 - (tp_c + fp_c + fn_c)
            
            acc_c_10k = (tp_c + tn_c) / 16.0
            platform_res[m_name][f'Accuracy-{c_name}'] = acc_c_10k
            
            sens_c = tp_c / (tp_c + fn_c) # (10000,)
            spec_c = tn_c / (tn_c + fp_c) # (10000,)
            
            denom_pos = 1.0 - spec_c
            lr_pos_10k = np.where(denom_pos > 0, sens_c / denom_pos, np.inf)
            platform_res[m_name][f'LR+-{c_name}'] = lr_pos_10k
            
            denom_neg = spec_c
            lr_neg_10k = np.where(denom_neg > 0, (1.0 - sens_c) / denom_neg, np.inf)
            platform_res[m_name][f'LR--{c_name}'] = lr_neg_10k

    for mtr in metric_rows:
        rec_row = {'platform': p_name, 'metric': mtr}
        for m_name in models_dict.keys():
            arr = platform_res[m_name][mtr]
            cell_str, def_pct, med, lo, hi = format_cell(arr)
            rec_row[m_name] = cell_str
            
            # Sanity flag checks
            reasons = []
            if def_pct < 50.0:
                reasons.append(f"terdefinisi < 50% ({def_pct:.1f}%)")
            if np.isfinite(lo) and np.isfinite(hi):
                ci_width = hi - lo
                if ci_width > 0.40:
                    reasons.append(f"lebar CI > 0.40 ({ci_width:.3f})")
            elif np.isposinf(hi) or np.isposinf(med):
                reasons.append("CI tak terhingga (inf)")
                
            if reasons:
                sanity_flags.append({
                    'platform': p_name,
                    'model': m_name,
                    'metric': mtr,
                    'cell_value': cell_str,
                    'reason': "; ".join(reasons)
                })
                
        results_all.append(rec_row)

df_results = pd.DataFrame(results_all)
csv_out_path = os.path.join(RESULTS_DIR, 'cipa_bootstrap_rev.csv')
df_results.to_csv(csv_out_path, index=False)
print(f"\nSaved CSV results to: {csv_out_path}")

# ----------------------------------------------------
# 5. WRITE MARKDOWN REPORT (STEP22_CIPA_BOOTSTRAP.MD)
# ----------------------------------------------------
md_path = os.path.join(DOCS_DIR, 'STEP22_CIPA_BOOTSTRAP.md')

md_lines = []
md_lines.append("# STEP 22 — CIPA BOOTSTRAP PROTOCOL (LI ET AL. & FUADAH ET AL.)\n")
md_lines.append(f"**Execution Date**: 2026-07-31  ")
md_lines.append(f"**Protocol**: 10,000 paired iterations resampling 1 row per drug across 16 test drugs.  ")
md_lines.append(f"**Stacking Model SHA-256 Hash**: `{actual_hash}` (Assertion PASS)\n")

md_lines.append("## Executive Summary\n")
md_lines.append("Sesuai protokol CiPA (Li et al. & Fuadah et al.), bootstrap ini mengevaluasi variabilitas simulasi dengan me-resample 1 sampel acak dari 2.000 baris simulasi untuk masing-masing dari 16 obat uji di setiap iterasi. Seluruh 16 obat selalu terwakili di setiap iterasi (paired across models), sehingga variabilitas berasal murni dari simulasi ion-channel, bukan dari perubahan komposisi panel obat.\n")

md_lines.append("## Point Estimate Guard Assertions\n")
md_lines.append("Pengujian titik estimasi level obat (mean probabilitas 2.000 baris per obat) mengonfirmasi nilai persis:\n")
md_lines.append("| Model | Chantest | Nanion | Manual | Guard Status |")
md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
for m_name in models_dict.keys():
    c_f1 = expected_f1[m_name]['chantest']
    n_f1 = expected_f1[m_name]['nanion']
    m_f1 = expected_f1[m_name]['manual']
    md_lines.append(f"| {m_name} | {c_f1:.6f} | {n_f1:.6f} | {m_f1:.6f} | PASS |")

md_lines.append("\n---\n")

for p_name in platforms:
    md_lines.append(f"## Tabel Results — Platform: `{p_name.upper()}`\n")
    md_lines.append("| Metric | Stacking | RF | LR | LR-qNet |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    sub_df = df_results[df_results['platform'] == p_name]
    for _, r in sub_df.iterrows():
        md_lines.append(f"| {r['metric']} | {r['Stacking']} | {r['RF']} | {r['LR']} | {r['LR-qNet']} |")
        
    md_lines.append("\n")

md_lines.append("## SANITY FLAGS\n")
md_lines.append("Daftar sel metrik yang terdefinisi di bawah 50% iterasi atau memiliki lebar Confidence Interval (CI) > 0.40:\n")

if sanity_flags:
    md_lines.append("| Platform | Model | Metric | Cell Output | Flag Reason |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for sf in sanity_flags:
        md_lines.append(f"| {sf['platform']} | {sf['model']} | {sf['metric']} | `{sf['cell_value']}` | {sf['reason']} |")
else:
    md_lines.append("Tidak ada sel yang memicu sanity flag.\n")

md_content = "\n".join(md_lines)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"Saved Markdown report to: {md_path}")
print("\nCiPA Bootstrap Protocol script completed cleanly!")
