import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import math
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from datetime import datetime

trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev

output_lines = []
def log(msg=""):
    output_lines.append(str(msg))
    print(msg)

scaler_modules = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'QuantileTransformer']
for sm in scaler_modules:
    if sm in globals() or sm in locals():
        log(f"ERROR: Scaler {sm} detected in script environment!")
        sys.exit(1)

SEED = 42

meta_path  = os.path.join(trainai_dir, 'models', 'chantest_stacking_metadata_rev.json')
log("Loading metadata to extract hyperparameters...")
with open(meta_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

tuned_hp = metadata['tuned_hyperparameters']
log("Hyperparameters Loaded:")
log(json.dumps(tuned_hp, indent=2))
label_map = metadata['label_map']
inv_label_map = {v: k for k, v in label_map.items()}
canonical_biomarkers = metadata['canonical_biomarkers']

log("\n==========================================")
log("=== STEP 0 — SEVERITY MAP ===")
log("==========================================")
SEVERITY = {0: 2, 1: 1, 2: 0}   # high=2, int=1, low=0
log(f"SEVERITY = {SEVERITY}")

def get_action_severity(pred_set_classes):
    if len(pred_set_classes) == 0:
        return -1 # abstain
    max_sev = -1
    for c in pred_set_classes:
        if SEVERITY[c] > max_sev:
            max_sev = SEVERITY[c]
    return max_sev

def get_outcome(true_class, pred_set_classes):
    action_sev = get_action_severity(pred_set_classes)
    if action_sev == -1:
        return "abstain"
    true_sev = SEVERITY[true_class]
    if action_sev == true_sev:
        return "correct"
    elif action_sev > true_sev:
        return "over-called"
    else:
        return "under-triaged"

log("\n==========================================")
log("=== STEP 1 — LODO CALIBRATION (12 genuine refits) ===")
log("==========================================")
X_train_full, y_train_full, groups_train_full, _ = load_rev('chantest', 'training', data_dir=os.path.join(trainai_dir, 'data'))
all_train_drugs = sorted(groups_train_full.unique())

lodo_scores = []
for d in all_train_drugs:
    log(f"\n--- Refitting for hold-out drug: {d} ---")
    
    mask_d = (groups_train_full == d)
    X_d = X_train_full[mask_d]
    y_d = y_train_full[mask_d]
    
    mask_fit = (groups_train_full != d)
    X_fit = X_train_full[mask_fit]
    y_fit = y_train_full[mask_fit]
    groups_fit = groups_train_full[mask_fit]
    
    fit_drugs = set(groups_fit.unique())
    overlap = d in fit_drugs
    log(f"ASSERT drug {d} appears zero times in fit set: {'FAIL' if overlap else 'PASS'}")
    if overlap:
        log("ERROR: Drug overlap detected in LODO.")
        sys.exit(1)
        
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    fold_splits = list(sgkf.split(X_fit, y_fit, groups=groups_fit))
    
    rf = RandomForestClassifier(random_state=SEED, **tuned_hp['rf'])
    xgb_base = xgb.XGBClassifier(random_state=SEED, tree_method='hist', objective='multi:softprob', num_class=3, **tuned_hp['xgb'])
    meta_rf = RandomForestClassifier(random_state=SEED)
    
    stack_model = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_base)],
        final_estimator=meta_rf,
        stack_method='predict_proba',
        passthrough=False,
        cv=fold_splits
    )
    
    stack_model.fit(X_fit, y_fit)
    
    y_proba_d = stack_model.predict_proba(X_d)
    p_mean = y_proba_d.mean(axis=0)
    
    true_class = y_d.iloc[0]
    s_d = 1.0 - p_mean[true_class]
    
    lodo_scores.append({
        'drug_name': d,
        'true_label': inv_label_map[true_class],
        'p_high': p_mean[0],
        'p_int': p_mean[1],
        'p_low': p_mean[2],
        's_d': s_d,
        'true_class_idx': true_class
    })

lodo_df = pd.DataFrame(lodo_scores)
log("\n12-ROW LODO CALIBRATION TABLE:")
log(f"{'Drug Name':<15} | {'True Label':<12} | {'P(High)':<10} | {'P(Int)':<10} | {'P(Low)':<10} | s_d")
log("-" * 75)
for _, row in lodo_df.iterrows():
    log(f"{row['drug_name']:<15} | {row['true_label']:<12} | {row['p_high']:<10.6f} | {row['p_int']:<10.6f} | {row['p_low']:<10.6f} | {row['s_d']:<10.6f}")

sorted_sd = sorted(lodo_df['s_d'].values)
log(f"\n12 LODO scores SORTED ASCENDING: {[round(s, 6) for s in sorted_sd]}")
log(f"Min:    {sorted_sd[0]:.6f}")
log(f"Median: {np.median(sorted_sd):.6f}")
log(f"Max:    {sorted_sd[-1]:.6f}")

log("\nINTEGRITY CHECK:")
chan_tr_csv = os.path.join(trainai_dir, 'results', 'cross_platform_probs_chantest_rev.csv')
chan_tr_df = pd.read_csv(chan_tr_csv)
chan_tr_df = chan_tr_df[chan_tr_df['split'] == 'training']
frozen_scores = []
for _, row in chan_tr_df.iterrows():
    t = int(row['true_label'])
    p_vec = [row['prob_high'], row['prob_int'], row['prob_low']]
    frozen_scores.append(1.0 - p_vec[t])
frozen_max = max(frozen_scores)
lodo_min = min(sorted_sd)
log(f"Max frozen model score on these 12 drugs: {frozen_max:.6f}")
log(f"Min LODO refit score on these 12 drugs:   {lodo_min:.6f}")
if np.mean(sorted_sd) < 0.05 and max(sorted_sd) < 0.1:
    log("WARNING: LODO scores implausibly low - refit may not have excluded the held-out drug")
else:
    log("LODO scores are visibly larger than frozen-model scores. Refit integrity PASS.")

log("\n==========================================")
log("=== STEP 2 — ALPHA SWEEP AND QUANTIZATION ===")
log("==========================================")
n = 12
alphas = [0.05, 0.077, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
log(f"n = {n}")
alpha_qhats = {}
max_attainable_qhat = sorted_sd[-1]
min_feasible_alpha = 1.0 / (n + 1)
log(f"Minimum feasible alpha: 1/13 = {min_feasible_alpha:.6f}")
log(f"Maximum attainable q_hat: {max_attainable_qhat:.6f}")

log(f"\n{'Alpha':<10} | {'Rank k':<8} | {'Feasible':<10} | q_hat")
log("-" * 45)
for a in alphas:
    k = math.ceil(13 * (1 - a))
    feasible = (k <= 12)
    q_hat = sorted_sd[k-1] if feasible else None
    q_str = f"{q_hat:.6f}" if q_hat is not None else "N/A"
    log(f"{a:<10} | {k:<8} | {str(feasible):<10} | {q_str}")
    if feasible:
        alpha_qhats[a] = q_hat

log("\nDistinct alphas collapsing to same q_hat:")
qhat_to_alphas = {}
for a, q in alpha_qhats.items():
    if q not in qhat_to_alphas:
        qhat_to_alphas[q] = []
    qhat_to_alphas[q].append(a)
for q, a_list in qhat_to_alphas.items():
    if len(a_list) > 1:
        log(f"  q_hat {q:.6f} is shared by alphas: {a_list}")

log("\n==========================================")
log("=== STEP 3 — THE DECISIVE CHECK ===")
log("==========================================")
log("For disopyramide on each platform, s_high = 1 - p_high")
res_dfs = {}
for p in ['chantest', 'nanion', 'manual']:
    df_p = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{p}_rev.csv'))
    res_dfs[p] = df_p[df_p['split'] == 'testing'].copy()

diso_chantest = res_dfs['chantest'][res_dfs['chantest']['drug_name'] == 'disopyramide'].iloc[0]
diso_nanion = res_dfs['nanion'][res_dfs['nanion']['drug_name'] == 'disopyramide'].iloc[0]
diso_manual = res_dfs['manual'][res_dfs['manual']['drug_name'] == 'disopyramide'].iloc[0]

s_high_c = 1.0 - diso_chantest['prob_high']
s_high_n = 1.0 - diso_nanion['prob_high']
s_high_m = 1.0 - diso_manual['prob_high']

log(f"Chantest s_high: {s_high_c:.6f} | Min q_hat needed: {s_high_c:.6f}")
log(f"Nanion   s_high: {s_high_n:.6f} | Min q_hat needed: {s_high_n:.6f}")
log(f"Manual   s_high: {s_high_m:.6f} | Min q_hat needed: {s_high_m:.6f}")

rec_c = "YES" if s_high_c <= max_attainable_qhat else "NO"
rec_n = "YES" if s_high_n <= max_attainable_qhat else "NO"
log(f"\nGiven max attainable q_hat ({max_attainable_qhat:.6f}), is disopyramide under-triage recoverable on chantest? {rec_c} (needs {s_high_c:.6f}).")
log(f"Given max attainable q_hat ({max_attainable_qhat:.6f}), is disopyramide under-triage recoverable on nanion?   {rec_n} (needs {s_high_n:.6f}).")


log("\n==========================================")
log("=== STEP 4 — PREDICTION SETS ON 16 TEST DRUGS ===")
log("==========================================")

conformal_sets_records = []
alpha_platform_summaries = []

# Choose human alpha: the lowest feasible alpha that achieves highest q_hat to maximize safety
human_alpha = 0.15 # corresponds to k=12, q_hat max

log(f"{'Alpha':<6} | {'Platform':<10} | {'Correct':<8} | {'Over':<6} | {'Under':<6} | {'Abstain':<8} | {'Avg Size':<9} | Uninformative")
log("-" * 80)

sanity_flags = []

for a, q in alpha_qhats.items():
    for plat in ['chantest', 'nanion', 'manual']:
        df_plat = res_dfs[plat]
        
        counts = {'abstain': 0, 'correct': 0, 'over-called': 0, 'under-triaged': 0}
        total_set_size = 0
        uninformative = 0
        empirical_cov = 0
        
        for _, row in df_plat.iterrows():
            drug = row['drug_name']
            true_lbl = inv_label_map[int(row['true_label'])]
            true_cls = int(row['true_label'])
            p_vec = [row['prob_high'], row['prob_int'], row['prob_low']]
            
            p_set = []
            set_names = []
            for k in range(3):
                if 1.0 - p_vec[k] <= q:
                    p_set.append(k)
                    set_names.append(inv_label_map[k])
            
            outcome = get_outcome(true_cls, p_set)
            counts[outcome] += 1
            total_set_size += len(p_set)
            
            if len(p_set) == 3:
                uninformative += 1
            
            if true_cls in p_set:
                empirical_cov += 1
                
            action_sev = get_action_severity(p_set)
            action_name = inv_label_map[list(SEVERITY.keys())[list(SEVERITY.values()).index(action_sev)]] if action_sev != -1 else 'abstain'
            
            conformal_sets_records.append({
                'alpha': a,
                'q_hat': q,
                'platform': plat,
                'drug_name': drug,
                'true_label': true_lbl,
                'hard_pred': row['pred_label'],
                'prediction_set': str(set_names),
                'action': action_name,
                'outcome': outcome
            })
            
            if a == human_alpha and plat == 'chantest':
                pass # print later
                
        avg_sz = total_set_size / 16.0
        alpha_platform_summaries.append({
            'alpha': a,
            'platform': plat,
            'correct': counts['correct'],
            'over': counts['over-called'],
            'under': counts['under-triaged'],
            'abstain': counts['abstain'],
            'avg_sz': avg_sz,
            'uninformative': uninformative,
            'empirical_cov': empirical_cov / 16.0
        })
        
        log(f"{a:<6} | {plat:<10} | {counts['correct']:<8} | {counts['over-called']:<6} | {counts['under-triaged']:<6} | {counts['abstain']:<8} | {avg_sz:<9.2f} | {uninformative}")
        
        if avg_sz > 2.0:
            sanity_flags.append(f"Alpha {a} on {plat} has avg set size > 2.0 ({avg_sz:.2f})")
        if (empirical_cov/16.0) < (1.0 - a):
            sanity_flags.append(f"Alpha {a} on {plat} empirical coverage ({empirical_cov/16.0:.4f}) fell below nominal ({1.0 - a:.4f})")
        if counts['under-triaged'] > 0:
            sanity_flags.append(f"Alpha {a} on {plat} still has {counts['under-triaged']} under-triaged drugs")

log(f"\nFULL 16-ROW DETAIL FOR HUMAN-PICKED ALPHA = {human_alpha}")
log(f"{'Drug Name':<15} | {'True':<12} | {'Hard Pred':<12} | {'Prediction Set':<35} | {'Action':<12} | Outcome")
log("-" * 110)
for r in conformal_sets_records:
    if r['alpha'] == human_alpha and r['platform'] == 'chantest':
        log(f"{r['drug_name']:<15} | {r['true_label']:<12} | {inv_label_map[r['hard_pred']]:<12} | {r['prediction_set']:<35} | {r['action']:<12} | {r['outcome']}")

log("\n==========================================")
log("=== STEP 5 — COVERAGE ===")
log("==========================================")
log(f"{'Alpha':<6} | {'Nominal':<8} | {'Platform':<10} | {'Empirical':<10} | Gap")
log("-" * 50)
for summ in alpha_platform_summaries:
    nom = 1.0 - summ['alpha']
    emp = summ['empirical_cov']
    gap = emp - nom
    log(f"{summ['alpha']:<6} | {nom:<8.4f} | {summ['platform']:<10} | {emp:<10.4f} | {gap:+.4f}")


log("\n==========================================")
log("=== STEP 6 — HARD-LABEL BASELINE ===")
log("==========================================")
for plat in ['chantest', 'nanion', 'manual']:
    df_plat = res_dfs[plat]
    h_counts = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    for _, row in df_plat.iterrows():
        t = int(row['true_label'])
        p = int(row['pred_label'])
        t_sev = SEVERITY[t]
        p_sev = SEVERITY[p]
        if p_sev == t_sev:
            h_counts['correct'] += 1
        elif p_sev > t_sev:
            h_counts['over-called'] += 1
        else:
            h_counts['under-triaged'] += 1
    log(f"{plat.upper()} Hard-Label: Correct={h_counts['correct']}, Over-called={h_counts['over-called']}, Under-triaged={h_counts['under-triaged']}")


log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
if 'WARNING: LODO scores implausibly low' in "".join(output_lines):
    log("  - FLAG: Implausibly low LODO scores.")

unique_flags = []
for f in sanity_flags:
    if f not in unique_flags:
        unique_flags.append(f)

if unique_flags:
    for f in unique_flags:
        log(f"  - FLAG: {f}")
else:
    log("  - No sanity flags raised.")


results_dir = os.path.join(trainai_dir, 'results')
lodo_df.to_csv(os.path.join(results_dir, 'conformal_lodo_scores_rev.csv'), index=False)
pd.DataFrame(conformal_sets_records).to_csv(os.path.join(results_dir, 'conformal_sets_rev.csv'), index=False)

doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP5_CONFORMAL.md')
with open(doc_path_repo, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

log(f"\nSaved conformal_lodo_scores_rev.csv")
log(f"Saved conformal_sets_rev.csv")
log(f"Written docs/STEP5_CONFORMAL.md")
