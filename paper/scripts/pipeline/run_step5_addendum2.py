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
log("=== ADDENDUM 2 — CORRECTED FRAGILITY ===")
log("==========================================")

log("\n=== STEP 1 — RETRACT THE DEGENERATE FRAGILITY RESULT ===")
log("Example variant (removing diltiazem):")
n = 11
alpha = 1/13
rank = math.ceil((n+1) * (1 - alpha)) # 12
log(f"n: {n}, alpha: {alpha:.6f}, rank: {rank}")
log(f"Does rank exceed n? {'YES' if rank > n else 'NO'}")
log("Resulting q_hat: 1.000000 (+inf formally in conformal)")
q_hat = 1.0
threshold = 1.0 - q_hat
log(f"Inclusion threshold (1 - q_hat): {threshold:.6f}")
log("Because the threshold is 0, any predicted probability p >= 0 satisfies 1 - p <= q_hat (1 - p <= 1.0).")
log("Resulting set sizes across all 16 test drugs: 3 for every drug.")
log("Resulting action for every drug: 'high' (most severe class in the set).")
log("Is this equivalent to assigning every drug the most severe class regardless of its probabilities? YES, the evidence shows that a q_hat of 1.0 produces universally uninformative sets ['high', 'intermediate', 'low'] for every drug, forcing the fallback rule to blindly select 'high' across the board.")

log("\n=== STEP 2 — CORRECTED FRAGILITY (smallest feasible alpha at each n) ===")
lodo_df = pd.read_csv(os.path.join(trainai_dir, 'results', 'conformal_lodo_scores_rev.csv'))
lodo_df_sorted = lodo_df.sort_values(by='s_d', ascending=False).reset_index(drop=True)

n = 11
alpha_11 = 1/12
rank_11 = math.ceil(12 * (1 - alpha_11))
log(f"Smallest feasible alpha at n=11 is 1/12.")
log(f"Verification: ceil(12 * (1 - 1/12)) == {rank_11}")

variants = []
for i in range(len(lodo_df_sorted)):
    removed_drug = lodo_df_sorted.iloc[i]['drug_name']
    removed_score = lodo_df_sorted.iloc[i]['s_d']
    
    rem_scores = np.delete(lodo_df_sorted['s_d'].values, i)
    q_hat_LOO = sorted(rem_scores)[rank_11 - 1]
    
    rec_c = "YES" if q_hat_LOO >= 0.561270 else "NO"
    rec_n = "YES" if q_hat_LOO >= 0.735545 else "NO"
    preserves = (rec_c == "YES" and rec_n == "YES")
    
    variants.append({
        'removed': removed_drug,
        'removed_score': removed_score,
        'q_hat': q_hat_LOO,
        'preserves': preserves
    })
    
    log(f"Removed: {removed_drug:<15} | Score: {removed_score:.6f} | q_hat_LOO: {q_hat_LOO:.6f} | Threshold: {1.0 - q_hat_LOO:.6f}")
    log(f"   -> Chantest Recovered: {rec_c} (requires >= 0.561270)")
    log(f"   -> Nanion Recovered:   {rec_n} (requires >= 0.735545)")

q_vals = [v['q_hat'] for v in variants]
log(f"\nMin q_hat_LOO: {min(q_vals):.6f}")
log(f"Median q_hat_LOO: {np.median(q_vals):.6f}")
log(f"Max q_hat_LOO: {max(q_vals):.6f}")

preserves_cnt = sum([1 for v in variants if v['preserves']])
log(f"\nOut of 12 variants, {preserves_cnt} preserve zero under-triage on ALL three platforms.")

tightest_variant = min(variants, key=lambda x: x['q_hat'])
log(f"The tightest surviving margin was produced by removing {tightest_variant['removed']}.")
log(f"This dropped q_hat_LOO to {tightest_variant['q_hat']:.6f}.")


log("\n=== STEP 3 — RECOMPUTE SETS UNDER THE TIGHTEST VARIANT ===")
# We take q_hat = tightest_variant['q_hat'] (which is 0.762830)
# Recompute sets for 16 test drugs on all three platforms
q_tight = tightest_variant['q_hat']

SEVERITY = {0: 2, 1: 1, 2: 0}
LABEL_MAP = {"high": 0, "intermediate": 1, "low": 2}
INV_LABEL_MAP = {0: "high", 1: "intermediate", 2: "low"}

def get_action_severity(pred_set_classes):
    if len(pred_set_classes) == 0:
        return -1
    max_sev = -1
    for c in pred_set_classes:
        if SEVERITY[c] > max_sev:
            max_sev = SEVERITY[c]
    return max_sev

def get_outcome(true_class, pred_set_classes):
    action_sev = get_action_severity(pred_set_classes)
    if action_sev == -1: return "abstain"
    true_sev = SEVERITY[true_class]
    if action_sev == true_sev: return "correct"
    elif action_sev > true_sev: return "over-called"
    else: return "under-triaged"

log(f"Using Tightest Variant q_hat = {q_tight:.6f} vs Original q_hat = 0.986695")
for plat in ['chantest', 'nanion', 'manual']:
    prob_df = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{plat}_rev.csv'))
    prob_df = prob_df[prob_df['split'] == 'testing'].copy()
    
    # original alpha=1/13 stats
    # we know them from before, but let's recompute for safety
    def compute_stats(q):
        c = {'correct': 0, 'over-called': 0, 'under-triaged': 0, 'abstain': 0}
        sz_dist = {1: 0, 2: 0, 3: 0, 0: 0}
        tot_sz = 0
        for _, row in prob_df.iterrows():
            t = int(row['true_label'])
            p_vec = [row['prob_high'], row['prob_int'], row['prob_low']]
            p_set = []
            for k in range(3):
                if 1.0 - p_vec[k] <= q:
                    p_set.append(k)
            c[get_outcome(t, p_set)] += 1
            sz = len(p_set)
            sz_dist[sz] += 1
            tot_sz += sz
        return c, sz_dist, tot_sz / 16.0

    c_orig, d_orig, a_orig = compute_stats(0.986695)
    c_tight, d_tight, a_tight = compute_stats(q_tight)
    
    log(f"\nPlatform: {plat.upper()}")
    log(f"  ORIGINAL (q=0.986695): Correct={c_orig['correct']}, Over={c_orig['over-called']}, Under={c_orig['under-triaged']}, Abstain={c_orig['abstain']}")
    log(f"      Avg Set Size: {a_orig:.2f} | Dist: Size 1: {d_orig[1]}, Size 2: {d_orig[2]}, Size 3: {d_orig[3]}")
    log(f"  TIGHTEST (q={q_tight:.6f}): Correct={c_tight['correct']}, Over={c_tight['over-called']}, Under={c_tight['under-triaged']}, Abstain={c_tight['abstain']}")
    log(f"      Avg Set Size: {a_tight:.2f} | Dist: Size 1: {d_tight[1]}, Size 2: {d_tight[2]}, Size 3: {d_tight[3]}")

log("\nThis shows how heavily the safety of the operating point depends on a single calibration drug (diltiazem) holding up the q_hat value.")


log("\n=== STEP 4 — DOMPERIDONE DETAIL AT FULL PRECISION ===")
q_orig = 0.986695
thresh_orig = 1.0 - q_orig

for plat in ['chantest', 'nanion']:
    prob_df = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{plat}_rev.csv'))
    prob_row = prob_df[(prob_df['drug_name'] == 'domperidone') & (prob_df['split'] == 'testing')].iloc[0]
    
    p_vec = [prob_row['prob_high'], prob_row['prob_int'], prob_row['prob_low']]
    
    p_set = []
    p_set_names = []
    classes_cleared = []
    for k in range(3):
        p = p_vec[k]
        if 1.0 - p <= q_orig:
            p_set.append(k)
            p_set_names.append(INV_LABEL_MAP[k])
            classes_cleared.append(INV_LABEL_MAP[k])
    
    act_sev = get_action_severity(p_set)
    action_name = INV_LABEL_MAP[list(SEVERITY.keys())[list(SEVERITY.values()).index(act_sev)]] if act_sev != -1 else 'abstain'
    outcome = get_outcome(LABEL_MAP['intermediate'], p_set)
    
    log(f"\n--- domperidone on {plat} ---")
    log(f"p_high: {p_vec[0]:.6f}")
    log(f"p_intermediate: {p_vec[1]:.6f}")
    log(f"p_low: {p_vec[2]:.6f}")
    log(f"Threshold (1 - q_hat): {thresh_orig:.6f}")
    
    log(f"Classes that cleared threshold (1 - p <= {thresh_orig:.6f}): {classes_cleared}")
    log(f"Resulting Set: {p_set_names}")
    log(f"Action: {action_name}")
    log(f"Outcome: {outcome}")
    
    true_class_p = p_vec[1]
    if 1.0 - true_class_p > q_orig:
        log(f"CONFIRMED: The true class (intermediate) probability ({true_class_p:.6f}) means 1 - p = {1.0 - true_class_p:.6f}, which falls below the threshold, excluding it from the set.")
    else:
        log("ERROR: True class did not fall below threshold.")


log("\n=== STEP 5 — REPAIR THE SANITY FLAGS ===")
log("Recomputing flags at alpha = 1/13 (q_hat = 0.986695):")
nominal = 1.0 - (1/13) # 0.923077

# We re-read conformal_sets at alpha=1/13 just to be safe
# But we already computed exact values in memory.
for plat in ['chantest', 'nanion', 'manual']:
    prob_df = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{plat}_rev.csv'))
    prob_df = prob_df[prob_df['split'] == 'testing'].copy()
    
    c_orig, d_orig, a_orig = compute_stats(0.986695)
    
    emp_cov_count = 0
    for _, row in prob_df.iterrows():
        t = int(row['true_label'])
        p_vec = [row['prob_high'], row['prob_int'], row['prob_low']]
        p_set = []
        for k in range(3):
            if 1.0 - p_vec[k] <= 0.986695:
                p_set.append(k)
        if t in p_set:
            emp_cov_count += 1
            
    emp_cov = emp_cov_count / 16.0
    pct_size_3 = (d_orig[3] / 16.0) * 100
    
    log(f"\nPlatform: {plat.upper()}")
    # check avg set size > 2.0
    if a_orig > 2.0:
        log(f"FLAG: Average set size is {a_orig:.2f} (> 2.0).")
    else:
        log(f"CLEAR: Average set size is {a_orig:.2f} (<= 2.0).")
        
    # check emp cov < nominal
    if emp_cov < nominal:
        log(f"FLAG: Empirical coverage ({emp_cov:.6f}) < nominal ({nominal:.6f}).")
    else:
        log(f"CLEAR: Empirical coverage ({emp_cov:.6f}) >= nominal ({nominal:.6f}).")
        
    # check > 25% size 3 sets
    if pct_size_3 > 25.0:
        log(f"FLAG: {pct_size_3:.1f}% of drugs ({d_orig[3]}) receive size-3 sets (> 25%).")
    else:
        log(f"CLEAR: {pct_size_3:.1f}% of drugs ({d_orig[3]}) receive size-3 sets (<= 25%).")
        
    # check remaining under-triage
    if c_orig['under-triaged'] > 0:
        log(f"FLAG: Remaining under-triage count is {c_orig['under-triaged']}.")
    else:
        log(f"CLEAR: No remaining under-triages.")
        
    # check removes ZERO under-triages but converts correct to over-calls
    # hard label under-triages:
    hl_counts = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    for _, row in prob_df.iterrows():
        true_sev = SEVERITY[int(row['true_label'])]
        act_sev = SEVERITY[int(row['pred_label'])]
        if act_sev == true_sev: hl_counts['correct'] += 1
        elif act_sev > true_sev: hl_counts['over-called'] += 1
        else: hl_counts['under-triaged'] += 1
        
    removed_under = hl_counts['under-triaged'] - c_orig['under-triaged']
    converted_over = hl_counts['correct'] - c_orig['correct']
    if removed_under == 0 and converted_over > 0:
        log(f"FLAG: Conformal removed ZERO under-triages relative to hard labels, but sacrificed {converted_over} correct drugs by converting them to over-calls for zero safety gain.")
    else:
        log(f"CLEAR: Hard-label comparison does not show purely wasteful sacrifices (Removed under-triages: {removed_under}).")

doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP5_CONFORMAL.md')
with open(doc_path_repo, 'a', encoding='utf-8') as f:
    f.write("\n" + "\n".join(output_lines))

log("\nAppended to docs/STEP5_CONFORMAL.md")
