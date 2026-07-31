import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Environment setup
trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

log_messages = []
def log(msg=""):
    log_messages.append(str(msg))
    print(msg)

SEED = 42
N_BOOTSTRAP = 10000

SEVERITY = {0: 2, 1: 1, 2: 0} # high=2, int=1, low=0
inv_label_map = {0: 'high', 1: 'intermediate', 2: 'low'}
class_names = ['high', 'intermediate', 'low']
platforms = ['chantest', 'nanion', 'manual']
models = ['Stacking', 'Random Forest', 'LR Multivariate', 'LR qNet-only']

qhat_rank11 = {
    'Stacking': 0.762830,
    'Random Forest': 0.533987,
    'LR Multivariate': 0.645363,
    'LR qNet-only': 0.663626
}

prob_file_map = {
    'Stacking': {p: os.path.join(trainai_dir, 'results', f'cross_platform_probs_{p}_rev.csv') for p in platforms},
    'Random Forest': {p: os.path.join(trainai_dir, 'results', f'rf_probs_{p}_rev.csv') for p in platforms},
    'LR Multivariate': {p: os.path.join(trainai_dir, 'results', f'logistic_multivariate_probs_{p}_rev.csv') for p in platforms},
    'LR qNet-only': {p: os.path.join(trainai_dir, 'results', f'logistic_qnet_probs_{p}_rev.csv') for p in platforms}
}

log("==========================================")
log("=== STEP 21 — CI PER-CLASS & METRICS ===")
log("==========================================")

# ----------------------------------------------------
# SUMBER DATA & CONFIRMATION
# ----------------------------------------------------
log("\n=== CONFIRMATION ROW COUNTS (SPLIT='TESTING') ===")

test_datasets = {}

for m_name in models:
    test_datasets[m_name] = {}
    for p_name in platforms:
        fpath = prob_file_map[m_name][p_name]
        if not os.path.exists(fpath):
            log(f"ERROR: File not found {fpath}")
            sys.exit(1)
        df_full = pd.read_csv(fpath)
        df_test = df_full[df_full['split'] == 'testing'].reset_index(drop=True)
        r_cnt = len(df_test)
        log(f"Model: {m_name:<15} | Platform: {p_name:<8} | Rows testing: {r_cnt} -> {'PASS' if r_cnt == 16 else 'FAIL'}")
        if r_cnt != 16:
            log(f"ERROR: File {fpath} does not have exactly 16 testing rows!")
            sys.exit(1)
        test_datasets[m_name][p_name] = df_test

# ----------------------------------------------------
# BAGIAN 6 — GUARD ASSERTIONS (POINT ESTIMATES)
# ----------------------------------------------------
log("\n==========================================")
log("=== BAGIAN 6 — GUARD ASSERTIONS ===")
log("==========================================")

expected_point_f1 = {
    'Stacking':        {'chantest': 0.6944444444444443, 'nanion': 0.6210317460317459, 'manual': 0.7547619047619047},
    'Random Forest':   {'chantest': 0.8777777777777778, 'nanion': 0.8193732193732194, 'manual': 0.7555555555555555},
    'LR Multivariate': {'chantest': 0.8193732193732194, 'nanion': 0.7547619047619047, 'manual': 0.8193732193732194},
    'LR qNet-only':    {'chantest': 0.8119658119658121, 'nanion': 0.7566137566137566, 'manual': 0.7515151515151516}
}

guard_failed = False

for m_name in models:
    for p_name in platforms:
        df_p = test_datasets[m_name][p_name]
        y_true = df_p['true_label'].values
        y_pred = df_p['pred_label'].values
        f1_val = f1_score(y_true, y_pred, average='macro')
        exp_val = expected_point_f1[m_name][p_name]
        diff = abs(f1_val - exp_val)
        pass_f1 = diff < 1e-6
        log(f"Point F1 Assert: {m_name:<15} {p_name:<8} | Val={f1_val:.6f} Exp={exp_val:.6f} Diff={diff:.2e} -> {'PASS' if pass_f1 else 'FAIL'}")
        if not pass_f1:
            guard_failed = True

# Guard for Stacking + conformal rank 11
stack_ref_q = qhat_rank11['Stacking'] # 0.762830
thresh_ref = 1.0 - stack_ref_q

c_correct_cnt = 0
n_correct_cnt = 0
m_correct_cnt = 0

c_under_cnt = 0
n_under_cnt = 0
m_under_cnt = 0

c_sz_total = 0
n_sz_total = 0
m_sz_total = 0

for p_name in platforms:
    df_p = test_datasets['Stacking'][p_name]
    corr = 0
    under = 0
    sz_sum = 0
    for _, row in df_p.iterrows():
        y_t = int(row['true_label'])
        p_vec = [row['prob_high'], row['prob_int'], row['prob_low']]
        p_set = [c for c in range(3) if p_vec[c] >= thresh_ref]
        sz_sum += len(p_set)
        
        if len(p_set) > 0:
            act_sev = max(SEVERITY[c] for c in p_set)
            t_sev = SEVERITY[y_t]
            if act_sev == t_sev:
                corr += 1
            elif act_sev < t_sev:
                under += 1
                
    if p_name == 'chantest':
        c_correct_cnt = corr
        c_under_cnt = under
        c_sz_total = sz_sum / 16.0
    elif p_name == 'nanion':
        n_correct_cnt = corr
        n_under_cnt = under
        n_sz_total = sz_sum / 16.0
    elif p_name == 'manual':
        m_correct_cnt = corr
        m_under_cnt = under
        m_sz_total = sz_sum / 16.0

pass_c = (c_correct_cnt == 11 and n_correct_cnt == 11 and m_correct_cnt == 11)
pass_u = (c_under_cnt == 0 and n_under_cnt == 0 and m_under_cnt == 0)
pass_sz = (abs(c_sz_total - 1.1250) < 1e-4 and abs(n_sz_total - 1.1250) < 1e-4 and abs(m_sz_total - 1.3750) < 1e-4)

log(f"Stacking Conf Rank 11 Correctness (11/11/11 expected): Chantest={c_correct_cnt}, Nanion={n_correct_cnt}, Manual={m_correct_cnt} -> {'PASS' if pass_c else 'FAIL'}")
log(f"Stacking Conf Rank 11 Under-triage (0/0/0 expected):   Chantest={c_under_cnt}, Nanion={n_under_cnt}, Manual={m_under_cnt} -> {'PASS' if pass_u else 'FAIL'}")
log(f"Stacking Conf Rank 11 Set Size (1.125/1.125/1.375 exp): Chantest={c_sz_total:.4f}, Nanion={n_sz_total:.4f}, Manual={m_sz_total:.4f} -> {'PASS' if pass_sz else 'FAIL'}")

if guard_failed or not (pass_c and pass_u and pass_sz):
    log("ERROR: Point estimate guard check failed! Run cannot be trusted.")
    sys.exit(1)
else:
    log("Guard Assertions Final Check: ALL PASS")

# ----------------------------------------------------
# BOOTSTRAP RESAMPLING SETUP (PAIRED ACROSS MODELS)
# ----------------------------------------------------
log("\n==========================================")
log("=== BOOTSTRAP RESAMPLING SETUP (10,000 ITERATIONS) ===")
log("==========================================")

log("Aturan Resample Degenerate:")
log("1. Jika suatu kelas (high/int/low) tidak muncul pada resample (0 sampel), macro-average F1 dihitung HANYA atas kelas yang ADA.")
log("2. Jika metrik tidak terdefinisi (mis. AUC tanpa positif/negatif, precision/recall denominator 0), resample tersebut DIKELUARKAN dari agregasi metrik tersebut.")
log("3. Paired resampling: Indeks 16 obat latih/uji yang di-resample SAMA EXACT untuk keempat model di setiap iterasi b.")

# Pre-generate 10,000 bootstrap index arrays of size 16
boot_indices_all = []
for b in range(N_BOOTSTRAP):
    rng = np.random.RandomState(SEED + b)
    idx_b = rng.choice(16, size=16, replace=True)
    boot_indices_all.append(idx_b)

csv_records = []

# ----------------------------------------------------
# HELPER FUNCTIONS FOR CALCULATIONS
# ----------------------------------------------------
def format_ci(pe, med, lo, hi):
    return f"{pe:.3f} ({lo:.3f}-{hi:.3f})"

def calc_ovr_auc(y_true, scores, c):
    y_bin = (y_true == c).astype(int)
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return roc_auc_score(y_bin, scores)

def calc_ord_auc(y_true, p_high, p_int, ambang):
    if ambang == 1:
        s = p_high + p_int
        y_bin = ((y_true == 0) | (y_true == 1)).astype(int)
    else: # ambang 2
        s = p_high
        y_bin = (y_true == 0).astype(int)
        
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return roc_auc_score(y_bin, s)

def calc_class_prec_rec_f1(y_true, y_pred, c):
    tp = np.sum((y_true == c) & (y_pred == c))
    fp = np.sum((y_true != c) & (y_pred == c))
    fn = np.sum((y_true == c) & (y_pred != c))
    tn = np.sum((y_true != c) & (y_pred != c))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = 0.0 if not (np.isnan(prec) or np.isnan(rec)) else np.nan
    else:
        f1 = 2 * prec * rec / (prec + rec)
        
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    # LR+ and LR-
    if not np.isnan(rec) and not np.isnan(spec):
        fpr = 1.0 - spec
        lr_pos = rec / fpr if fpr > 0 else np.inf
        lr_neg = (1.0 - rec) / spec if spec > 0 else np.inf
    else:
        lr_pos = np.nan
        lr_neg = np.nan
        
    return prec, rec, f1, spec, lr_pos, lr_neg

def calc_macro_f1_degenerate(y_true, y_pred):
    present_classes = np.unique(y_true)
    f1s = []
    for c in present_classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return np.mean(f1s) if len(f1s) > 0 else 0.0

# Store all results for report generation
results_store = {}

# ----------------------------------------------------
# COMPUTATIONS FOR BAGIAN 1 - BAGIAN 5
# ----------------------------------------------------
for m_name in models:
    results_store[m_name] = {}
    for p_name in platforms:
        results_store[m_name][p_name] = {}
        df_p = test_datasets[m_name][p_name]
        
        y_true_orig = df_p['true_label'].values
        y_pred_orig = df_p['pred_label'].values
        p_high_orig = df_p['prob_high'].values
        p_int_orig  = df_p['prob_int'].values
        p_low_orig  = df_p['prob_low'].values
        
        q_hat_val = qhat_rank11[m_name]
        thresh_val = 1.0 - q_hat_val
        
        # Point estimates
        pe_metrics = {}
        
        # Per-class metrics
        for c in range(3):
            c_name = class_names[c]
            prob_c_orig = df_p[f'prob_{c_name[:4] if c_name!="intermediate" else "int"}'].values
            prec, rec, f1_c, spec, lr_p, lr_n = calc_class_prec_rec_f1(y_true_orig, y_pred_orig, c)
            auc_ovr = calc_ovr_auc(y_true_orig, prob_c_orig, c)
            
            pe_metrics[f'prec_{c_name}'] = prec
            pe_metrics[f'rec_{c_name}'] = rec
            pe_metrics[f'f1_{c_name}'] = f1_c
            pe_metrics[f'auc_ovr_{c_name}'] = auc_ovr
            pe_metrics[f'lr_pos_{c_name}'] = lr_p
            pe_metrics[f'lr_neg_{c_name}'] = lr_n
            
        # Macro F1
        pe_metrics['macro_f1_drug'] = f1_score(y_true_orig, y_pred_orig, average='macro')
        
        # Ordinal AUCs
        pe_metrics['auc_ord_1'] = calc_ord_auc(y_true_orig, p_high_orig, p_int_orig, 1)
        pe_metrics['auc_ord_2'] = calc_ord_auc(y_true_orig, p_high_orig, p_int_orig, 2)
        
        # Conformal Outcomes on Rank 11
        conf_corr = 0
        conf_over = 0
        conf_under = 0
        conf_abst = 0
        conf_szs = []
        conf_cov = 0
        
        for idx_d in range(16):
            y_t = y_true_orig[idx_d]
            p_vec = [p_high_orig[idx_d], p_int_orig[idx_d], p_low_orig[idx_d]]
            p_set = [c for c in range(3) if p_vec[c] >= thresh_val]
            conf_szs.append(len(p_set))
            
            if len(p_set) == 0:
                conf_abst += 1
            else:
                act_sev = max(SEVERITY[c] for c in p_set)
                t_sev = SEVERITY[y_t]
                if act_sev == t_sev:
                    conf_corr += 1
                elif act_sev > t_sev:
                    conf_over += 1
                else:
                    conf_under += 1
            if y_t in p_set:
                conf_cov += 1
                
        pe_metrics['conf_corr'] = conf_corr / 16.0
        pe_metrics['conf_over'] = conf_over / 16.0
        pe_metrics['conf_under'] = conf_under / 16.0
        pe_metrics['conf_abst'] = conf_abst / 16.0
        pe_metrics['conf_avg_sz'] = np.mean(conf_szs)
        pe_metrics['conf_emp_cov'] = conf_cov / 16.0
        
        # ----------------------------------------------------
        # BOOTSTRAP ITERATIONS FOR THIS MODEL & PLATFORM
        # ----------------------------------------------------
        boot_dist = {k: [] for k in pe_metrics.keys()}
        
        degenerate_count = 0
        inf_lr_pos_count = {c_name: 0 for c_name in class_names}
        inf_lr_neg_count = {c_name: 0 for c_name in class_names}
        
        for b in range(N_BOOTSTRAP):
            boot_idx = boot_indices_all[b]
            
            y_b = y_true_orig[boot_idx]
            yp_b = y_pred_orig[boot_idx]
            ph_b = p_high_orig[boot_idx]
            pi_b = p_int_orig[boot_idx]
            pl_b = p_low_orig[boot_idx]
            
            # Check degenerate class count
            if len(np.unique(y_b)) < 3:
                degenerate_count += 1
                
            # Macro F1
            mf1_b = calc_macro_f1_degenerate(y_b, yp_b)
            boot_dist['macro_f1_drug'].append(mf1_b)
            
            # Ordinal AUCs
            a1_b = calc_ord_auc(y_b, ph_b, pi_b, 1)
            a2_b = calc_ord_auc(y_b, ph_b, pi_b, 2)
            boot_dist['auc_ord_1'].append(a1_b)
            boot_dist['auc_ord_2'].append(a2_b)
            
            # Per-class metrics
            for c in range(3):
                c_name = class_names[c]
                prob_c_b = ph_b if c==0 else (pi_b if c==1 else pl_b)
                prec_b, rec_b, f1_cb, spec_b, lrp_b, lrn_b = calc_class_prec_rec_f1(y_b, yp_b, c)
                auc_ovr_b = calc_ovr_auc(y_b, prob_c_b, c)
                
                boot_dist[f'prec_{c_name}'].append(prec_b)
                boot_dist[f'rec_{c_name}'].append(rec_b)
                boot_dist[f'f1_{c_name}'].append(f1_cb)
                boot_dist[f'auc_ovr_{c_name}'].append(auc_ovr_b)
                
                if np.isinf(lrp_b):
                    inf_lr_pos_count[c_name] += 1
                    boot_dist[f'lr_pos_{c_name}'].append(np.nan)
                else:
                    boot_dist[f'lr_pos_{c_name}'].append(lrp_b)
                    
                if np.isinf(lrn_b):
                    inf_lr_neg_count[c_name] += 1
                    boot_dist[f'lr_neg_{c_name}'].append(np.nan)
                else:
                    boot_dist[f'lr_neg_{c_name}'].append(lrn_b)
                    
            # Conformal outcomes
            c_corr_b = 0
            c_over_b = 0
            c_under_b = 0
            c_abst_b = 0
            c_szs_b = []
            c_cov_b = 0
            
            for idx_d in range(16):
                y_tb = y_b[idx_d]
                p_vec_b = [ph_b[idx_d], pi_b[idx_d], pl_b[idx_d]]
                p_set_b = [c for c in range(3) if p_vec_b[c] >= thresh_val]
                c_szs_b.append(len(p_set_b))
                
                if len(p_set_b) == 0:
                    c_abst_b += 1
                else:
                    act_sev = max(SEVERITY[c] for c in p_set_b)
                    t_sev = SEVERITY[y_tb]
                    if act_sev == t_sev:
                        c_corr_b += 1
                    elif act_sev > t_sev:
                        c_over_b += 1
                    else:
                        c_under_b += 1
                if y_tb in p_set_b:
                    c_cov_b += 1
                    
            boot_dist['conf_corr'].append(c_corr_b / 16.0)
            boot_dist['conf_over'].append(c_over_b / 16.0)
            boot_dist['conf_under'].append(c_under_b / 16.0)
            boot_dist['conf_abst'].append(c_abst_b / 16.0)
            boot_dist['conf_avg_sz'].append(np.mean(c_szs_b))
            boot_dist['conf_emp_cov'].append(c_cov_b / 16.0)

        # ----------------------------------------------------
        # AGGREGATE BOOTSTRAP DISTRIBUTIONS & RECORD STATS
        # ----------------------------------------------------
        summary_stats = {}
        for m_key, pe_val in pe_metrics.items():
            vals = np.array(boot_dist[m_key])
            valid_vals = vals[~np.isnan(vals)]
            n_used = len(valid_vals)
            
            if n_used > 0:
                med = float(np.median(valid_vals))
                lo = float(np.percentile(valid_vals, 2.5))
                hi = float(np.percentile(valid_vals, 97.5))
            else:
                med, lo, hi = np.nan, np.nan, np.nan
                
            summary_stats[m_key] = {
                'point_estimate': pe_val,
                'median': med,
                'ci_lower': lo,
                'ci_upper': hi,
                'n_used': n_used
            }
            
            # Extract class name if per-class metric
            cls_field = 'all'
            for cn in class_names:
                if f'_{cn}' in m_key:
                    cls_field = cn
                    break
                    
            csv_records.append({
                'model': m_name,
                'platform': p_name,
                'kelas': cls_field,
                'metrik': m_key,
                'titik_estimasi': pe_val,
                'median_bootstrap': med,
                'ci_bawah': lo,
                'ci_atas': hi,
                'n_resample_terpakai': n_used
            })
            
        summary_stats['degenerate_count'] = degenerate_count
        summary_stats['inf_lr_pos_count'] = inf_lr_pos_count
        summary_stats['inf_lr_neg_count'] = inf_lr_neg_count
        
        results_store[m_name][p_name] = summary_stats

log("Calculations & Bootstrap Resampling Completed Successfully!")

# ----------------------------------------------------
# SAVE CSV OUTPUT
# ----------------------------------------------------
csv_path = os.path.join(trainai_dir, 'results', 'ci_perclass_rev.csv')
df_csv = pd.DataFrame(csv_records)
df_csv.to_csv(csv_path, index=False)
log(f"\nSaved CSV results to {csv_path}")

# ----------------------------------------------------
# SANITY FLAGS AUDIT
# ----------------------------------------------------
sanity_flags = []

for r in csv_records:
    lo = r['ci_bawah']
    hi = r['ci_atas']
    n_u = r['n_resample_terpakai']
    m_k = r['metrik']
    mod = r['model']
    plat = r['platform']
    cls_f = r['kelas']
    pe = r['titik_estimasi']
    
    # 1. CI width > 0.40
    if not np.isnan(lo) and not np.isnan(hi):
        width = hi - lo
        if width > 0.40:
            sanity_flags.append(f"CI Width > 0.40: Model '{mod}' Platform '{plat}' Metric '{m_k}' (Width = {width:.3f}: {lo:.3f} - {hi:.3f})")
            
    # 2. Undefined in > 10% of resamples (< 9000 used)
    if n_u < 9000:
        undef_pct = (10000 - n_u) / 100.0
        sanity_flags.append(f"Undefined > 10%: Model '{mod}' Platform '{plat}' Metric '{m_k}' (Undefined in {undef_pct:.1f}% resamples)")
        
    # 3. Class with zero recall
    if 'rec_' in m_k and pe == 0.0:
        sanity_flags.append(f"Zero Recall: Model '{mod}' Platform '{plat}' Class '{cls_f}' has point recall = 0.0")

# ----------------------------------------------------
# WRITE MARKDOWN REPORT (DOCS/STEP21_CI_PERCLASS.MD)
# ----------------------------------------------------
log("\n==========================================")
log("=== WRITING DOCS/STEP21_CI_PERCLASS.MD ===")
log("==========================================")

doc_path = os.path.join(trainai_dir, 'docs', 'STEP21_CI_PERCLASS.md')

md_lines = []
md_lines.append("# Step 21: Confidence Intervals & Per-Class Metrics (CiPA Format)\n")
md_lines.append("This document reports point estimates, bootstrap medians, and 95% percentile confidence intervals (10,000 paired resamples over drugs, seed 42) for four models across three platforms (`chantest`, `nanion`, `manual`).\n")
md_lines.append("---\n")

# BAGIAN 1 — METRIK PER KELAS, HARD LABEL
md_lines.append("## Bagian 1 — Metrik Per Kelas (Hard Label)\n")

for metric_type, title in [
    ('prec', 'Precision Per Kelas'),
    ('rec', 'Recall (Sensitivitas) Per Kelas'),
    ('f1', 'F1-Score Per Kelas'),
    ('auc_ovr', 'AUC One-vs-Rest Per Kelas')
]:
    md_lines.append(f"### {title}\n")
    md_lines.append("| Model | Kelas | `chantest` | `nanion` | `manual` |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    
    for m_name in models:
        for c_name in class_names:
            row_str = f"| {m_name} | `{c_name}` | "
            for p_name in platforms:
                st = results_store[m_name][p_name][f'{metric_type}_{c_name}']
                pe = st['point_estimate']
                med = st['median']
                lo = st['ci_lower']
                hi = st['ci_upper']
                if np.isnan(pe) or np.isnan(med):
                    cell = "N/A"
                else:
                    cell = format_ci(pe, med, lo, hi)
                row_str += f"{cell} | "
            md_lines.append(row_str)
    md_lines.append("\n")

md_lines.append("---\n")

# BAGIAN 2 — MACRO-F1 DENGAN CI
md_lines.append("## Bagian 2 — Macro-F1 Level Obat dan Level Baris\n")
md_lines.append("| Model | Platform | Drug-Level Macro-F1 (Point Estimate) | Drug-Level Macro-F1 (Bootstrap Median & 95% CI) | Row-Level Macro-F1 (Point Estimate) |")
md_lines.append("| :--- | :--- | :---: | :---: | :---: |")

row_f1_ref = {
    'Stacking':        {'chantest': 0.688289, 'nanion': 0.632400, 'manual': 0.692524},
    'Random Forest':   {'chantest': 0.821534, 'nanion': 0.773452, 'manual': 0.737576},
    'LR Multivariate': {'chantest': 0.816781, 'nanion': 0.770725, 'manual': 0.748463},
    'LR qNet-only':    {'chantest': 0.752276, 'nanion': 0.718603, 'manual': 0.715266}
}

for m_name in models:
    for p_name in platforms:
        st = results_store[m_name][p_name]['macro_f1_drug']
        pe = st['point_estimate']
        med = st['median']
        lo = st['ci_lower']
        hi = st['ci_upper']
        row_pe = row_f1_ref[m_name][p_name]
        md_lines.append(f"| {m_name} | `{p_name}` | {pe:.4f} | {format_ci(pe, med, lo, hi)} | {row_pe:.4f} |")

md_lines.append("\n---\n")

# BAGIAN 3 — AUC ORDINAL DENGAN CI
md_lines.append("## Bagian 3 — AUC Ordinal Dua Ambang Dengan CI\n")
md_lines.append("| Model | Platform | Ambang 1: $P(\\text{risk} > \\text{low})$ | Ambang 2: $P(\\text{risk} > \\text{int})$ | Resample Undefined Ambang 2 |")
md_lines.append("| :--- | :--- | :---: | :---: | :---: |")

for m_name in models:
    for p_name in platforms:
        st1 = results_store[m_name][p_name]['auc_ord_1']
        st2 = results_store[m_name][p_name]['auc_ord_2']
        
        cell1 = format_ci(st1['point_estimate'], st1['median'], st1['ci_lower'], st1['ci_upper'])
        cell2 = format_ci(st2['point_estimate'], st2['median'], st2['ci_lower'], st2['ci_upper']) if not np.isnan(st2['median']) else "N/A"
        undef2 = N_BOOTSTRAP - st2['n_used']
        
        md_lines.append(f"| {m_name} | `{p_name}` | {cell1} | {cell2} | {undef2} / 10,000 |")

md_lines.append("\n---\n")

# BAGIAN 4 — RASIO LIKELIHOOD (LR+ DAN LR-)
md_lines.append("## Bagian 4 — Rasio Likelihood (LR+ dan LR-)\n")

for metric_key, title in [('lr_pos', 'Positive Likelihood Ratio (LR+)'), ('lr_neg', 'Negative Likelihood Ratio (LR-)')]:
    md_lines.append(f"### {title}\n")
    md_lines.append("| Model | Kelas | Platform | Point Estimate | Bootstrap Median & 95% CI | Resample Terpakai (Defined) | Infinite Resamples |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for m_name in models:
        for c_name in class_names:
            for p_name in platforms:
                st = results_store[m_name][p_name][f'{metric_key}_{c_name}']
                pe = st['point_estimate']
                med = st['median']
                lo = st['ci_lower']
                hi = st['ci_upper']
                n_u = st['n_used']
                inf_cnt = results_store[m_name][p_name]['inf_lr_pos_count' if metric_key=='lr_pos' else 'inf_lr_neg_count'][c_name]
                
                pe_str = f"{pe:.3f}" if not np.isinf(pe) and not np.isnan(pe) else ("Inf" if np.isinf(pe) else "N/A")
                ci_str = format_ci(pe, med, lo, hi) if not np.isnan(med) else "Inf / N/A"
                
                md_lines.append(f"| {m_name} | `{c_name}` | `{p_name}` | {pe_str} | {ci_str} | {n_u} | {inf_cnt} |")
    md_lines.append("\n")

md_lines.append("---\n")

# BAGIAN 5 — OUTCOME CONFORMAL DENGAN CI (RANK 11)
md_lines.append("## Bagian 5 — Outcome Conformal Prediction Sets (Rank 11 Operating Point)\n")
md_lines.append("| Model | Platform | Proporsi Benar | Proporsi Over-called | Proporsi Under-triaged | Proporsi Abstain | Rata-rata Ukuran Set | Coverage Empiris |")
md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

for m_name in models:
    for p_name in platforms:
        st_corr = results_store[m_name][p_name]['conf_corr']
        st_over = results_store[m_name][p_name]['conf_over']
        st_under = results_store[m_name][p_name]['conf_under']
        st_abst = results_store[m_name][p_name]['conf_abst']
        st_sz   = results_store[m_name][p_name]['conf_avg_sz']
        st_cov  = results_store[m_name][p_name]['conf_emp_cov']
        
        cell_corr = format_ci(st_corr['point_estimate'], st_corr['median'], st_corr['ci_lower'], st_corr['ci_upper'])
        cell_over = format_ci(st_over['point_estimate'], st_over['median'], st_over['ci_lower'], st_over['ci_upper'])
        
        # Special Rule of Three for Zero Under-triage
        if st_under['point_estimate'] == 0.0 and st_under['median'] == 0.0 and st_under['ci_upper'] == 0.0:
            cell_under = "0.000 (Rule-of-Three Upper Bound: 0.1875)"
        else:
            cell_under = format_ci(st_under['point_estimate'], st_under['median'], st_under['ci_lower'], st_under['ci_upper'])
            
        cell_abst = format_ci(st_abst['point_estimate'], st_abst['median'], st_abst['ci_lower'], st_abst['ci_upper'])
        cell_sz   = format_ci(st_sz['point_estimate'], st_sz['median'], st_sz['ci_lower'], st_sz['ci_upper'])
        cell_cov  = format_ci(st_cov['point_estimate'], st_cov['median'], st_cov['ci_lower'], st_cov['ci_upper'])
        
        md_lines.append(f"| {m_name} | `{p_name}` | {cell_corr} | {cell_over} | {cell_under} | {cell_abst} | {cell_sz} | {cell_cov} |")

md_lines.append("\n*Catatan*: Untuk under-triage yang bernilai nol pada sampel pengujian (0 events dalam N=16), interval bootstrap menyusut menjadi titik (0.000). Sesuai aturan Rule of Three, batas atas interval kepercayaan 95% dilaporkan sebagai 3/N = 3/16 = 0.1875 (18.75%).\n")
md_lines.append("---\n")

# SANITY FLAGS SECTION
md_lines.append("## Sanity Flags\n")
if sanity_flags:
    for flag in sanity_flags:
        md_lines.append(f"- ⚠️ **WARNING**: {flag}\n")
else:
    md_lines.append("NONE (All CIs <= 0.40 width, clean defined metrics, non-zero recall across all classes).\n")

md_lines.append("\n---\n*Report generated automatically by `scripts/reporting/run_step21_ci_perclass.py`.*")

with open(doc_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(md_lines))

log(f"Successfully generated documentation at {doc_path}")

# ----------------------------------------------------
# PRINT SANITY FLAGS TO CONSOLE
# ----------------------------------------------------
log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
if sanity_flags:
    for flag in sanity_flags:
        log(f"- [WARNING] {flag}")
else:
    log("NONE")

log("\n==========================================")
log("=== STEP 21 COMPLETED SUCCESSFULLY ===")
log("==========================================")
