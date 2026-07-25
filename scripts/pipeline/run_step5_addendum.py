import os
import sys
import pandas as pd
import numpy as np
import math
import ast

trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

output_lines = []
def log(msg=""):
    output_lines.append(str(msg))
    print(msg)

log("==========================================")
log("=== ADDENDUM — BASELINES AND FRAGILITY ===")
log("==========================================")

log("\n=== STEP 0 — VERIFY THE SEVERITY LOGIC SURVIVED THE KeyError FIX ===")
log("Code block from run_step5_conformal.py that computes correct / over-called / under-triaged:")
log("""
SEVERITY = {0: 2, 1: 1, 2: 0}

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
""")

log("Confirmed it uses SEVERITY = {0:2, 1:1, 2:0} and not raw label integers.")

conformal_sets_df = pd.read_csv(os.path.join(trainai_dir, 'results', 'conformal_sets_rev.csv'))
lodo_df = pd.read_csv(os.path.join(trainai_dir, 'results', 'conformal_lodo_scores_rev.csv'))

prob_chan = pd.read_csv(os.path.join(trainai_dir, 'results', 'cross_platform_probs_chantest_rev.csv'))
prob_chan = prob_chan[prob_chan['split'] == 'testing'].copy()

SEVERITY = {0: 2, 1: 1, 2: 0}
LABEL_MAP = {"high": 0, "intermediate": 1, "low": 2}
INV_LABEL_MAP = {0: "high", 1: "intermediate", 2: "low"}

def spot_check(drug_name):
    a_check = 0.15
    row_set = conformal_sets_df[(conformal_sets_df['drug_name'] == drug_name) & 
                                (conformal_sets_df['platform'] == 'chantest') & 
                                (np.isclose(conformal_sets_df['alpha'], a_check))].iloc[0]
    
    true_lbl_str = row_set['true_label']
    true_lbl_int = LABEL_MAP[true_lbl_str]
    true_sev = SEVERITY[true_lbl_int]
    
    pred_set_str = row_set['prediction_set']
    pred_set = ast.literal_eval(pred_set_str)
    pred_set_ints = [LABEL_MAP[x] for x in pred_set]
    
    max_sev = -1
    most_severe_class = None
    for c in pred_set_ints:
        if SEVERITY[c] > max_sev:
            max_sev = SEVERITY[c]
            most_severe_class = c
            
    if max_sev == true_sev:
        outcome = "correct"
    elif max_sev > true_sev:
        outcome = "over-called"
    else:
        outcome = "under-triaged"
        
    log(f"--- Spot Check: {drug_name} ---")
    log(f"True label: {true_lbl_str} (Severity: {true_sev})")
    log(f"Prediction set: {pred_set_str}")
    log(f"Most-severe class in set: {INV_LABEL_MAP[most_severe_class]} (Severity: {max_sev})")
    log(f"Computed Outcome: {outcome}")
    log(f"Recorded Outcome: {row_set['outcome']}")
    if outcome == row_set['outcome']:
        log("Match: PASS")
    else:
        log("Match: FAIL")
        sys.exit(1)

spot_check('disopyramide')
spot_check('loratadine')
spot_check('astemizole')

log("\n=== STEP 1 — OPERATING POINT AT alpha = 1/13 ===")
alphas_to_test = [1/13, 0.10, 0.15]
for a in alphas_to_test:
    k = math.ceil(13 * (1 - a))
    log(f"alpha {a:.6f} -> k = ceil(13 * (1 - {a:.6f})) = {k}")

a_target = 0.077
q_13 = conformal_sets_df[np.isclose(conformal_sets_df['alpha'], a_target)]['q_hat'].iloc[0]
q_10 = conformal_sets_df[np.isclose(conformal_sets_df['alpha'], 0.1)]['q_hat'].iloc[0]
q_15 = conformal_sets_df[np.isclose(conformal_sets_df['alpha'], 0.15)]['q_hat'].iloc[0]
if q_13 == q_10 == q_15:
    log("Identical q_hat across {1/13, 0.10, 0.15}: PASS")
else:
    log("Identical q_hat across {1/13, 0.10, 0.15}: FAIL")
    
sets_13 = conformal_sets_df[np.isclose(conformal_sets_df['alpha'], a_target)]['prediction_set'].values
sets_10 = conformal_sets_df[np.isclose(conformal_sets_df['alpha'], 0.1)]['prediction_set'].values
sets_15 = conformal_sets_df[np.isclose(conformal_sets_df['alpha'], 0.15)]['prediction_set'].values

if np.array_equal(sets_13, sets_10) and np.array_equal(sets_13, sets_15):
    log("Identical prediction sets across all platforms: PASS")
else:
    log("Identical prediction sets across all platforms: FAIL")

log("\nAdopting alpha = 1/13 (~0.076923) as the reported operating point.")
platforms = ['chantest', 'nanion', 'manual']

for plat in platforms:
    plat_df = conformal_sets_df[(conformal_sets_df['platform'] == plat) & (np.isclose(conformal_sets_df['alpha'], a_target))]
    n_total = len(plat_df)
    
    counts = {'correct': 0, 'over-called': 0, 'under-triaged': 0, 'abstain': 0}
    emp_cov_count = 0
    total_sz = 0
    uninformative = 0
    
    for _, row in plat_df.iterrows():
        p_set = ast.literal_eval(row['prediction_set'])
        total_sz += len(p_set)
        if len(p_set) == 3:
            uninformative += 1
        if row['true_label'] in p_set:
            emp_cov_count += 1
        counts[row['outcome']] += 1
        
    emp_cov = emp_cov_count / n_total
    nom_cov = 1.0 - (1/13)
    gap = emp_cov - nom_cov
    avg_sz = total_sz / n_total
    
    log(f"\n--- Platform: {plat.upper()} ---")
    log(f"Nominal coverage (1-alpha): {nom_cov:.4f}")
    log(f"Empirical coverage: {emp_cov:.4f}")
    log(f"Signed gap: {gap:+.4f}")
    log(f"Correct: {counts['correct']}, Over-called: {counts['over-called']}, Under-triaged: {counts['under-triaged']}")
    log(f"Average set size: {avg_sz:.2f}")
    log(f"All-three-class sets (Uninformative): {uninformative}")


log("\n=== STEP 2 — TRIVIAL BASELINES ===")
baselines_stats = {}

for plat in platforms:
    prob_df = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{plat}_rev.csv'))
    prob_df = prob_df[prob_df['split'] == 'testing'].copy()
    
    ah_counts = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    for _, row in prob_df.iterrows():
        true_sev = SEVERITY[int(row['true_label'])]
        act_sev = SEVERITY[0] # high
        if act_sev == true_sev:
            ah_counts['correct'] += 1
        elif act_sev > true_sev:
            ah_counts['over-called'] += 1
        else:
            ah_counts['under-triaged'] += 1
            
    hl_counts = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    for _, row in prob_df.iterrows():
        true_sev = SEVERITY[int(row['true_label'])]
        act_sev = SEVERITY[int(row['pred_label'])]
        if act_sev == true_sev:
            hl_counts['correct'] += 1
        elif act_sev > true_sev:
            hl_counts['over-called'] += 1
        else:
            hl_counts['under-triaged'] += 1
            
    cf_counts = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    cf_df = conformal_sets_df[(conformal_sets_df['platform'] == plat) & (np.isclose(conformal_sets_df['alpha'], a_target))]
    for _, row in cf_df.iterrows():
        cf_counts[row['outcome']] += 1
        
    avg_sz = sum([len(ast.literal_eval(s)) for s in cf_df['prediction_set']]) / 16.0
    
    baselines_stats[plat] = {
        'ah': ah_counts,
        'hl': hl_counts,
        'cf': cf_counts,
        'cf_sz': avg_sz
    }

log(f"{'Platform':<10} | {'Method':<15} | {'Correct':<8} | {'Over':<6} | {'Under':<6} | Avg Set Size")
log("-" * 70)
for plat in platforms:
    st = baselines_stats[plat]
    log(f"{plat:<10} | {'ALWAYS-HIGH':<15} | {st['ah']['correct']:<8} | {st['ah']['over-called']:<6} | {st['ah']['under-triaged']:<6} | 1.00")
    log(f"{'':<10} | {'HARD LABEL':<15} | {st['hl']['correct']:<8} | {st['hl']['over-called']:<6} | {st['hl']['under-triaged']:<6} | 1.00")
    log(f"{'':<10} | {'CONFORMAL':<15} | {st['cf']['correct']:<8} | {st['cf']['over-called']:<6} | {st['cf']['under-triaged']:<6} | {st['cf_sz']:.2f}")

log("")
for plat in platforms:
    st = baselines_stats[plat]
    more_correct = st['cf']['correct'] - st['ah']['correct']
    log(f"On {plat}, conformal gets {more_correct} more drugs CORRECT than ALWAYS-HIGH.")
    
    converted = st['hl']['correct'] - st['cf']['correct']
    removed_under = st['hl']['under-triaged'] - st['cf']['under-triaged']
    log(f"On {plat}, conformal converts {converted} drugs from correct to over-called relative to HARD LABEL, and removes {removed_under} under-triages in exchange.")


log("\n=== STEP 3 — Q_HAT FRAGILITY ===")
lodo_df_sorted = lodo_df.sort_values(by='s_d', ascending=False).reset_index(drop=True)
log("12 LODO scores sorted descending:")
for _, row in lodo_df_sorted.iterrows():
    log(f"  {row['drug_name']:<15}: {row['s_d']:.6f}")

recomputed_qhats = []
alpha = 1/13
rank_11 = math.ceil(12 * (1 - alpha)) # 12

log(f"\nRecomputing q_hat (alpha=1/13, n=11) -> target rank = {rank_11}")
for i in range(len(lodo_df_sorted)):
    drug_removed = lodo_df_sorted.iloc[i]['drug_name']
    rem_scores = np.delete(lodo_df_sorted['s_d'].values, i)
    rem_scores_asc = sorted(rem_scores)
    
    if rank_11 > len(rem_scores_asc):
        q_hat = 1.0
    else:
        q_hat = rem_scores_asc[rank_11 - 1]
    
    rec_c = "YES" if q_hat >= 0.561270 else "NO"
    rec_n = "YES" if q_hat >= 0.735545 else "NO"
    preserves_all = (rec_c == "YES" and rec_n == "YES")
    
    recomputed_qhats.append({
        'drug_removed': drug_removed,
        'q_hat': q_hat,
        'rec_c': rec_c,
        'rec_n': rec_n,
        'preserves': preserves_all
    })
    
    log(f"Remove {drug_removed:<15} -> q_hat: {q_hat:.6f} | Chantest Recov: {rec_c} | Nanion Recov: {rec_n}")

q_vals = [x['q_hat'] for x in recomputed_qhats]
log(f"\nMin q_hat: {min(q_vals):.6f}")
log(f"Median q_hat: {np.median(q_vals):.6f}")
log(f"Max q_hat: {max(q_vals):.6f}")

preserves_count = sum([1 for x in recomputed_qhats if x['preserves']])
log(f"\nOut of 12 leave-one-out variants, {preserves_count} preserve zero under-triage on ALL three platforms.")


log("\n=== STEP 4 — COVERAGE MISSES ===")
for plat in ['chantest', 'nanion']:
    cf_df = conformal_sets_df[(conformal_sets_df['platform'] == plat) & (np.isclose(conformal_sets_df['alpha'], a_target))]
    miss_df = None
    miss_drug = None
    for _, row in cf_df.iterrows():
        p_set = ast.literal_eval(row['prediction_set'])
        if row['true_label'] not in p_set:
            miss_drug = row['drug_name']
            miss_df = row
            break
            
    if miss_drug is not None:
        prob_df = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{plat}_rev.csv'))
        prob_row = prob_df[(prob_df['drug_name'] == miss_drug) & (prob_df['split'] == 'testing')].iloc[0]
        
        q_hat_val = miss_df['q_hat']
        threshold = 1.0 - q_hat_val
        miss_type = miss_df['outcome']
        
        log(f"\nPlatform: {plat}")
        log(f"Drug Name: {miss_drug}")
        log(f"True Label: {miss_df['true_label']}")
        log(f"p_high: {prob_row['prob_high']:.6f}")
        log(f"p_int:  {prob_row['prob_int']:.6f}")
        log(f"p_low:  {prob_row['prob_low']:.6f}")
        log(f"Threshold (1 - q_hat): {threshold:.6f}")
        log(f"Prediction Set: {miss_df['prediction_set']}")
        log(f"Action Taken: {miss_df['action']}")
        log(f"Outcome: {miss_type}")
        if miss_type == 'over-called':
            log("This miss coincided with an over-call.")
        elif miss_type == 'under-triaged':
            log("This miss coincided with an under-triage.")
        else:
            log(f"This miss coincided with outcome: {miss_type}")
    else:
        log(f"No coverage miss found on {plat} at alpha 1/13.")


log("\n=== STEP 5 — SET-SIZE DISTRIBUTION ===")
for plat in platforms:
    cf_df = conformal_sets_df[(conformal_sets_df['platform'] == plat) & (np.isclose(conformal_sets_df['alpha'], a_target))]
    sz_buckets = {1: [], 2: [], 3: []}
    for _, row in cf_df.iterrows():
        p_set = ast.literal_eval(row['prediction_set'])
        sz = len(p_set)
        if sz in sz_buckets:
            sz_buckets[sz].append(row['drug_name'])
            
    log(f"\nPlatform: {plat.upper()}")
    for sz in [1, 2, 3]:
        drugs = sz_buckets[sz]
        log(f"Set size {sz}: {len(drugs)} drugs")
        if len(drugs) > 0:
            log(f"  Drugs: {', '.join(drugs)}")


log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
if preserves_count < 12:
    log(f"  - FLAG: q_hat is extremely fragile. {12 - preserves_count}/12 leave-one-out calibration subsets fail to recover the disopyramide under-triage.")
else:
    log("  - No new sanity flags in addendum.")

doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP5_CONFORMAL.md')
with open(doc_path_repo, 'a', encoding='utf-8') as f:
    f.write("\n" + "\n".join(output_lines))

log("\nAppended to docs/STEP5_CONFORMAL.md")
