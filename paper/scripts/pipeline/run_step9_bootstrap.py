import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import warnings

# Suppress UndefinedMetricWarning if any
warnings.filterwarnings('ignore', category=UserWarning)

trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
output_lines = []
def log(msg=""):
    output_lines.append(str(msg))
    print(msg)

# -----------------
# INITIAL SETUP
# -----------------
log("==========================================")
log("=== STEP 9 — DRUG-LEVEL BOOTSTRAP ===")
log("==========================================")
log("Resampling unit: DRUG (16 held-out test drugs)")
log("Iterations: 10000")
log("Seed: 42")
log("==========================================\n")

SEED = 42
np.random.seed(SEED)
N_ITER = 10000

SEVERITY = {0: 2, 1: 1, 2: 0} # 0=High (Sev 2), 1=Int (Sev 1), 2=Low (Sev 0)
Q_HAT_11 = 0.762830
THRESH_11 = 1 - Q_HAT_11
Q_HAT_12 = 0.986695
THRESH_12 = 1 - Q_HAT_12

platforms = ['chantest', 'nanion', 'manual']
data = {}
for p in platforms:
    fpath = os.path.join(trainai_dir, 'results', f'cross_platform_probs_{p}_rev.csv')
    df = pd.read_csv(fpath)
    df = df[df['split'] == 'testing'].reset_index(drop=True)
    if len(df) != 16:
        log(f"ERROR: Expected 16 test drugs for {p}, found {len(df)}")
        sys.exit(1)
    data[p] = df

# Rule for degenerate resamples
log("DEGENERATE-RESAMPLE HANDLING:")
log("Rule applied: We compute macro-average F1 solely over the classes PRESENT in the true labels of each resample, adjusting to avoid undefined division by zero.")

# -----------------
# BOOTSTRAP LOOP
# -----------------
results = {p: {'f1': [], 'r0': [], 'r1': [], 'r2': [], 
               'c11_cov': [], 'c11_cor': [], 'c11_ovr': [], 'c11_und': [], 
               'c12_cov': [], 'missing_classes': 0} for p in platforms}

diff_c_n = []
diff_c_m = []
diff_n_m = []

for i in range(N_ITER):
    indices = np.random.choice(16, size=16, replace=True)
    
    iter_f1s = {}
    
    for p in platforms:
        df_resample = data[p].iloc[indices].copy()
        y_t = df_resample['true_label'].values
        y_p = df_resample['pred_label'].values
        
        # Missing classes
        present_classes = np.unique(y_t)
        if len(present_classes) < 3:
            results[p]['missing_classes'] += 1
            
        # F1
        f1 = f1_score(y_t, y_p, labels=present_classes, average='macro')
        results[p]['f1'].append(f1)
        iter_f1s[p] = f1
        
        # Recalls
        for c, r_list in [(0, 'r0'), (1, 'r1'), (2, 'r2')]:
            mask = (y_t == c)
            if mask.sum() > 0:
                results[p][r_list].append(np.sum((y_t == c) & (y_p == c)) / np.sum(mask))
            else:
                results[p][r_list].append(np.nan)
                
        # Conformal
        probs = df_resample[['prob_high', 'prob_int', 'prob_low']].values
        
        # Rank 11
        cov_11 = 0
        cor_11 = 0
        ovr_11 = 0
        und_11 = 0
        
        # Rank 12
        cov_12 = 0
        
        for k in range(16):
            p_vec = probs[k]
            t_lbl = y_t[k]
            t_sev = SEVERITY[t_lbl]
            
            # Rank 11 sets
            cset_11 = np.where(p_vec >= THRESH_11)[0].tolist()
            if not cset_11:
                cset_11 = [np.argmax(p_vec)]
            
            if t_lbl in cset_11:
                cov_11 += 1
            m_sev_11 = max([SEVERITY[c] for c in cset_11])
            if m_sev_11 == t_sev:
                cor_11 += 1
            elif m_sev_11 > t_sev:
                ovr_11 += 1
            else:
                und_11 += 1
                
            # Rank 12 sets
            cset_12 = np.where(p_vec >= THRESH_12)[0].tolist()
            if not cset_12:
                cset_12 = [np.argmax(p_vec)]
            
            if t_lbl in cset_12:
                cov_12 += 1
                
        results[p]['c11_cov'].append(cov_11 / 16.0)
        results[p]['c11_cor'].append(cor_11 / 16.0)
        results[p]['c11_ovr'].append(ovr_11 / 16.0)
        results[p]['c11_und'].append(und_11 / 16.0)
        results[p]['c12_cov'].append(cov_12 / 16.0)
        
    diff_c_n.append(iter_f1s['chantest'] - iter_f1s['nanion'])
    diff_c_m.append(iter_f1s['chantest'] - iter_f1s['manual'])
    diff_n_m.append(iter_f1s['nanion'] - iter_f1s['manual'])


def format_ci(arr, name, is_nan=False):
    if is_nan:
        med = np.nanmedian(arr)
        lo = np.nanpercentile(arr, 2.5)
        hi = np.nanpercentile(arr, 97.5)
    else:
        med = np.median(arr)
        lo = np.percentile(arr, 2.5)
        hi = np.percentile(arr, 97.5)
    return med, lo, hi

for p in platforms:
    log(f"Affected resamples missing a class for {p}: {results[p]['missing_classes']} out of 10000 adjusted.")
log("")

# Convert dict to dataframes for saving
csv_records = []
sanity_flags = []

for p in platforms:
    log(f"=== PLATFORM: {p.upper()} ===")
    
    # Point estimates for reference
    df_full = data[p]
    pt_y_t = df_full['true_label'].values
    pt_y_p = df_full['pred_label'].values
    pt_f1 = f1_score(pt_y_t, pt_y_p, average='macro')
    
    # 1. Macro-F1
    m, l, h = format_ci(results[p]['f1'], 'f1')
    log(f"Drug-level macro-F1 : {m:.6f} [95% CI: {l:.6f}, {h:.6f}] (Point Est: {pt_f1:.6f})")
    csv_records.append((p, 'macro-F1', m, l, h))
    if pt_f1 < l or pt_f1 > h:
        sanity_flags.append(f"FLAG: {p} macro-F1 point est ({pt_f1:.6f}) outside CI [{l:.6f}, {h:.6f}]")
    if (h - l) > 0.30:
        sanity_flags.append(f"FLAG: {p} macro-F1 CI width ({(h-l):.6f}) exceeds 0.30")
        
    # 2. Per-class recalls
    for c_idx, c_name, m_key in [(0, 'High', 'r0'), (1, 'Int', 'r1'), (2, 'Low', 'r2')]:
        m, l, h = format_ci(results[p][m_key], m_key, is_nan=True)
        log(f"Recall ({c_name:<4})       : {m:.6f} [95% CI: {l:.6f}, {h:.6f}]")
        csv_records.append((p, f'recall_{c_name.lower()}', m, l, h))
        if (h - l) > 0.30:
            sanity_flags.append(f"FLAG: {p} Recall ({c_name}) CI width ({(h-l):.6f}) exceeds 0.30")
            
    # 3. Conformal Safety @ Rank 11
    m_cor, l_cor, h_cor = format_ci(results[p]['c11_cor'], 'cor')
    m_ovr, l_ovr, h_ovr = format_ci(results[p]['c11_ovr'], 'ovr')
    m_und, l_und, h_und = format_ci(results[p]['c11_und'], 'und')
    
    log(f"Rank-11 Correct     : {m_cor:.6f} [95% CI: {l_cor:.6f}, {h_cor:.6f}]")
    log(f"Rank-11 Over-called : {m_ovr:.6f} [95% CI: {l_ovr:.6f}, {h_ovr:.6f}]")
    log(f"Rank-11 Under-triage: {m_und:.6f} [95% CI: {l_und:.6f}, {h_und:.6f}]")
    csv_records.append((p, 'conformal_rank11_correct', m_cor, l_cor, h_cor))
    csv_records.append((p, 'conformal_rank11_over', m_ovr, l_ovr, h_ovr))
    csv_records.append((p, 'conformal_rank11_under', m_und, l_und, h_und))
    
    if h_und > 0:
         sanity_flags.append(f"FLAG: {p} Under-triage upper bound ({h_und:.6f}) exceeds 0")
         
    # 4. Conformal Coverage
    m_c11, l_c11, h_c11 = format_ci(results[p]['c11_cov'], 'cov11')
    m_c12, l_c12, h_c12 = format_ci(results[p]['c12_cov'], 'cov12')
    log(f"Rank-11 Coverage    : {m_c11:.6f} [95% CI: {l_c11:.6f}, {h_c11:.6f}]")
    log(f"Rank-12 Coverage    : {m_c12:.6f} [95% CI: {l_c12:.6f}, {h_c12:.6f}]")
    csv_records.append((p, 'conformal_rank11_coverage', m_c11, l_c11, h_c11))
    csv_records.append((p, 'conformal_rank12_coverage', m_c12, l_c12, h_c12))
    
    log("")

log("=== PAIRWISE PLATFORM COMPARISON (Macro-F1 Differences) ===")
def log_diff(arr, name):
    m = np.median(arr)
    l = np.percentile(arr, 2.5)
    h = np.percentile(arr, 97.5)
    contains_zero = (l <= 0 <= h)
    cz_str = "contains zero" if contains_zero else "does NOT contain zero"
    log(f"{name:<20}: {m:>9.6f} [95% CI: {l:>9.6f}, {h:>9.6f}] -> {cz_str}")
    return m, l, h

m_cn, l_cn, h_cn = log_diff(diff_c_n, 'Chantest - Nanion')
m_cm, l_cm, h_cm = log_diff(diff_c_m, 'Chantest - Manual')
m_nm, l_nm, h_nm = log_diff(diff_n_m, 'Nanion - Manual')

csv_records.extend([
    ('pairwise', 'chantest_minus_nanion', m_cn, l_cn, h_cn),
    ('pairwise', 'chantest_minus_manual', m_cm, l_cm, h_cm),
    ('pairwise', 'nanion_minus_manual', m_nm, l_nm, h_nm),
])

log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
if sanity_flags:
    for flag in sanity_flags:
        log(flag)
else:
    log("All sanity checks passed. No CI widths exceed 0.30, point estimates are bounded, and under-triage upper bounds are 0.")

# Save CSV
df_out = pd.DataFrame(csv_records, columns=['platform', 'statistic', 'median', 'ci_lower', 'ci_upper'])
# Generate per-iteration CSV
iter_records = []
for p in platforms:
    for stat_name, key in [
        ('macro_f1', 'f1'), ('recall_high', 'r0'), ('recall_int', 'r1'), ('recall_low', 'r2'),
        ('conformal_rank11_correct', 'c11_cor'), ('conformal_rank11_over', 'c11_ovr'), ('conformal_rank11_under', 'c11_und'),
        ('conformal_rank11_coverage', 'c11_cov'), ('conformal_rank12_coverage', 'c12_cov')
    ]:
        # Using a list comprehension to build the huge dataframe quickly
        arr = results[p][key]
        for it, val in enumerate(arr):
            iter_records.append((it, p, stat_name, val))
            
# Also add pairwise differences
for it, val in enumerate(diff_c_n):
    iter_records.append((it, 'pairwise', 'chantest_minus_nanion', val))
for it, val in enumerate(diff_c_m):
    iter_records.append((it, 'pairwise', 'chantest_minus_manual', val))
for it, val in enumerate(diff_n_m):
    iter_records.append((it, 'pairwise', 'nanion_minus_manual', val))

df_iters = pd.DataFrame(iter_records, columns=['iteration', 'platform', 'statistic', 'value'])
df_iters.to_csv(os.path.join(trainai_dir, 'results', 'bootstrap_distributions_rev.csv'), index=False)

with open(os.path.join(trainai_dir, '..', 'docs', 'STEP9_BOOTSTRAP.md'), 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

log("\nSaved docs/STEP9_BOOTSTRAP.md and results/bootstrap_distributions_rev.csv")
