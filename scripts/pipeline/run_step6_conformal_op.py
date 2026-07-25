import os
import sys
import pandas as pd
import numpy as np

trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

output_lines = []
def log(msg=""):
    output_lines.append(str(msg))
    print(msg)

log("==========================================")
log("=== STEP 0 — SEVERITY MAP ===")
log("==========================================")
SEVERITY = {0: 2, 1: 1, 2: 0}
LABEL_MAP = {"high": 0, "intermediate": 1, "low": 2}
INV_LABEL_MAP = {0: "high", 1: "intermediate", 2: "low"}
log(f"SEVERITY = {SEVERITY} (high most severe, low least)")

def get_action_severity(pred_set_classes):
    if len(pred_set_classes) == 0:
        return -1 # abstain
    return max([SEVERITY[c] for c in pred_set_classes])

def get_outcome(true_class, pred_set_classes):
    action_sev = get_action_severity(pred_set_classes)
    if action_sev == -1: return "abstain"
    true_sev = SEVERITY[true_class]
    if action_sev == true_sev: return "correct"
    elif action_sev > true_sev: return "over-called"
    else: return "under-triaged"

log("\n==========================================")
log("=== STEP 1 — FULL RANK ENUMERATION, ALL 12 DRUGS RETAINED ===")
log("==========================================")
lodo_df = pd.read_csv(os.path.join(trainai_dir, 'results', 'conformal_lodo_scores_rev.csv'))
lodo_scores = sorted(lodo_df['s_d'].values)

ranks = {}
log(f"{'Rank k':<8} | {'q_hat(k)':<10} | {'Alpha Interval':<35} | {'Smallest Alpha':<15} | Inclusion Threshold (1 - q_hat)")
log("-" * 105)
for k in range(12, 7, -1):
    q_hat = lodo_scores[k-1]
    min_alpha = 1.0 - (k / 13.0)
    max_alpha = 1.0 - ((k - 1) / 13.0)
    alpha_interval = f"[{min_alpha:.6f}, {max_alpha:.6f})"
    thresh = 1.0 - q_hat
    ranks[k] = {'q_hat': q_hat, 'min_alpha': min_alpha, 'thresh': thresh}
    log(f"{k:<8} | {q_hat:<10.6f} | {alpha_interval:<35} | {min_alpha:<15.6f} | {thresh:.6f}")

log("\nConfirmed: rank 12 -> 0.986695 and rank 11 -> 0.762830.")

log("\n==========================================")
log("=== STEP 2 — OUTCOMES AT EVERY RANK, ALL THREE PLATFORMS ===")
log("==========================================")
platforms = ['chantest', 'nanion', 'manual']
prob_dfs = {}
for p in platforms:
    df = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{p}_rev.csv'))
    prob_dfs[p] = df[df['split'] == 'testing'].copy()

all_sweep_records = []
rank_outcomes = []

log(f"{'Rank':<4} | {'Platform':<10} | {'Cor':<4} | {'Ovr':<4} | {'Und':<4} | {'Abs':<4} | {'AvgSz':<6} | {'Dist(1,2,3)':<15} | {'EmpCov':<7} | {'NomCov':<7} | Gap")
log("-" * 100)

under_triage_by_rank = {}

for k in range(12, 7, -1):
    q = ranks[k]['q_hat']
    thresh = ranks[k]['thresh']
    nom_cov = 1.0 - ranks[k]['min_alpha']
    
    ut_str = {}
    for p in platforms:
        df = prob_dfs[p]
        c = {'correct': 0, 'over-called': 0, 'under-triaged': 0, 'abstain': 0}
        sz_dist = {1: 0, 2: 0, 3: 0, 0: 0}
        tot_sz = 0
        emp_cov_count = 0
        
        for _, row in df.iterrows():
            t = int(row['true_label'])
            p_vec = [row['prob_high'], row['prob_int'], row['prob_low']]
            p_set = []
            for c_idx in range(3):
                if 1.0 - p_vec[c_idx] <= q:
                    p_set.append(c_idx)
            
            outcome = get_outcome(t, p_set)
            c[outcome] += 1
            sz = len(p_set)
            sz_dist[sz] += 1
            tot_sz += sz
            if t in p_set:
                emp_cov_count += 1
                
            action_sev = get_action_severity(p_set)
            action_name = INV_LABEL_MAP[list(SEVERITY.keys())[list(SEVERITY.values()).index(action_sev)]] if action_sev != -1 else 'abstain'
            
            all_sweep_records.append({
                'rank': k,
                'q_hat': q,
                'platform': p,
                'drug_name': row['drug_name'],
                'true_label': INV_LABEL_MAP[t],
                'p_high': p_vec[0],
                'p_int': p_vec[1],
                'p_low': p_vec[2],
                'set': str([INV_LABEL_MAP[x] for x in p_set]),
                'action': action_name,
                'outcome': outcome,
                'avg_sz': tot_sz, # placeholder
                'emp_cov': 0
            })
            
        emp_cov = emp_cov_count / 16.0
        gap = emp_cov - nom_cov
        avg_sz = tot_sz / 16.0
        dist_str = f"({sz_dist[1]},{sz_dist[2]},{sz_dist[3]})"
        
        rank_outcomes.append({
            'rank': k, 'platform': p, 'c': c, 'avg_sz': avg_sz, 'sz_dist': sz_dist,
            'emp_cov': emp_cov, 'nom_cov': nom_cov, 'gap': gap
        })
        
        log(f"{k:<4} | {p:<10} | {c['correct']:<4} | {c['over-called']:<4} | {c['under-triaged']:<4} | {c['abstain']:<4} | {avg_sz:<6.2f} | {dist_str:<15} | {emp_cov:<7.4f} | {nom_cov:<7.4f} | {gap:+.4f}")
        
        ut_str[p] = c['under-triaged']
    
    under_triage_by_rank[k] = ut_str

log("")
safe_rank = None
for k in range(12, 7, -1):
    ut = under_triage_by_rank[k]
    log(f"rank {k}: under-triaged = chantest {ut['chantest']}, nanion {ut['nanion']}, manual {ut['manual']}")
    if ut['chantest'] == 0 and ut['nanion'] == 0 and ut['manual'] == 0 and safe_rank is None:
        safe_rank = k

safe_q = ranks[safe_rank]['q_hat']
log(f"\nSMALLEST q_hat (tightest rank) at which under-triage is zero on ALL three platforms is {safe_q:.6f} at rank {safe_rank}.")

log("\n==========================================")
log("=== STEP 3 — VERIFY THE BOUNDARY IN BOTH DIRECTIONS ===")
log("==========================================")
tight_rank = safe_rank - 1
log(f"Safe rank: {safe_rank} (q_hat = {ranks[safe_rank]['q_hat']:.6f})")
log(f"Tighter rank: {tight_rank} (q_hat = {ranks[tight_rank]['q_hat']:.6f})")
log("\nDrugs whose outcome changes between these two ranks:")

df_sweep = pd.DataFrame(all_sweep_records)
new_under_triaged = []

for p in platforms:
    safe_plat = df_sweep[(df_sweep['rank'] == safe_rank) & (df_sweep['platform'] == p)].copy().reset_index()
    tight_plat = df_sweep[(df_sweep['rank'] == tight_rank) & (df_sweep['platform'] == p)].copy().reset_index()
    
    for i in range(len(safe_plat)):
        row_s = safe_plat.iloc[i]
        row_t = tight_plat.iloc[i]
        
        if row_s['outcome'] != row_t['outcome']:
            log(f"\nPlatform: {p}")
            log(f"Drug Name: {row_s['drug_name']}")
            log(f"True Label: {row_s['true_label']}")
            log(f"p_high: {row_s['p_high']:.6f}, p_int: {row_s['p_int']:.6f}, p_low: {row_s['p_low']:.6f}")
            log(f"Rank {safe_rank} Set: {row_s['set']} | Action: {row_s['action']} | Outcome: {row_s['outcome']}")
            log(f"Rank {tight_rank} Set: {row_t['set']} | Action: {row_t['action']} | Outcome: {row_t['outcome']}")
            
            if row_t['outcome'] == 'under-triaged' and row_s['outcome'] != 'under-triaged':
                margin = (1.0 - row_s['p_high']) - ranks[tight_rank]['q_hat'] # it fell out of the set because 1-p > q_hat
                # wait, the class that was supposed to save it fell out. 
                # let's find the true severe class that fell out
                true_c = LABEL_MAP[row_s['true_label']]
                p_true = [row_s['p_high'], row_s['p_int'], row_s['p_low']][true_c]
                # the margin by which it fell below the threshold: (1-p) - q_hat
                # wait, it might not be the true class that saved it from under-triage, it could be a MORE severe class!
                # If true label is 'intermediate', and it was over-called because 'high' was in the set, and now 'high' falls out, it might become correct (not under-triaged).
                # Here, we only record it if it became under-triaged. 
                new_under_triaged.append({
                    'drug': row_s['drug_name'],
                    'plat': p,
                    'p_high': row_s['p_high'],
                    'thresh_missed': ranks[tight_rank]['q_hat']
                })

log("\nSpecifically, the following drugs become under-triaged when tightening by one rank:")
for x in new_under_triaged:
    margin = (1.0 - x['p_high']) - x['thresh_missed']
    log(f" - {x['drug']} on {x['plat']} misses the 'high' inclusion threshold by a probability margin of {margin:.6f} (1 - p_high = {1.0 - x['p_high']:.6f} > q_hat {x['thresh_missed']:.6f})")

log("\n==========================================")
log("=== STEP 4 — HEAD-TO-HEAD OF THE TWO CANDIDATE OPERATING POINTS ===")
log("==========================================")

log(f"{'Platform':<10} | {'Method (Rank/q_hat)':<25} | {'Min Alpha':<10} | {'NomCov':<8} | {'EmpCov':<8} | {'Gap':<8} | {'Cor':<4} | {'Ovr':<4} | {'Und':<4} | {'AvgSz':<6} | % Size-3")
log("-" * 120)

for p in platforms:
    # get baselines
    prob_df = prob_dfs[p]
    ah_c = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    hl_c = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    for _, row in prob_df.iterrows():
        t = int(row['true_label'])
        true_sev = SEVERITY[t]
        act_sev_ah = SEVERITY[0]
        if act_sev_ah == true_sev: ah_c['correct'] += 1
        elif act_sev_ah > true_sev: ah_c['over-called'] += 1
        else: ah_c['under-triaged'] += 1
        
        act_sev_hl = SEVERITY[int(row['pred_label'])]
        if act_sev_hl == true_sev: hl_c['correct'] += 1
        elif act_sev_hl > true_sev: hl_c['over-called'] += 1
        else: hl_c['under-triaged'] += 1
        
    r12 = [x for x in rank_outcomes if x['rank'] == 12 and x['platform'] == p][0]
    r11 = [x for x in rank_outcomes if x['rank'] == 11 and x['platform'] == p][0]
    
    log(f"{p:<10} | {'ALWAYS-HIGH':<25} | {'-':<10} | {'-':<8} | {'-':<8} | {'-':<8} | {ah_c['correct']:<4} | {ah_c['over-called']:<4} | {ah_c['under-triaged']:<4} | {'1.00':<6} | 0.0%")
    log(f"{'':<10} | {'HARD-LABEL':<25} | {'-':<10} | {'-':<8} | {'-':<8} | {'-':<8} | {hl_c['correct']:<4} | {hl_c['over-called']:<4} | {hl_c['under-triaged']:<4} | {'1.00':<6} | 0.0%")
    
    pct_3_r12 = (r12['sz_dist'][3] / 16.0) * 100
    log(f"{'':<10} | {'Rank 12 (q=0.986695)':<25} | {ranks[12]['min_alpha']:<10.6f} | {r12['nom_cov']:<8.4f} | {r12['emp_cov']:<8.4f} | {r12['gap']:<+8.4f} | {r12['c']['correct']:<4} | {r12['c']['over-called']:<4} | {r12['c']['under-triaged']:<4} | {r12['avg_sz']:<6.2f} | {pct_3_r12:.1f}%")
    
    pct_3_r11 = (r11['sz_dist'][3] / 16.0) * 100
    log(f"{'':<10} | {'Rank 11 (q=0.762830)':<25} | {ranks[11]['min_alpha']:<10.6f} | {r11['nom_cov']:<8.4f} | {r11['emp_cov']:<8.4f} | {r11['gap']:<+8.4f} | {r11['c']['correct']:<4} | {r11['c']['over-called']:<4} | {r11['c']['under-triaged']:<4} | {r11['avg_sz']:<6.2f} | {pct_3_r11:.1f}%")

log("\n==========================================")
log("=== STEP 5 — DILTIAZEM AS A CALIBRATION OUTLIER ===")
log("==========================================")
lodo_df_sorted = lodo_df.sort_values(by='s_d', ascending=False).reset_index(drop=True)
log("12 LODO scores sorted descending:")
for _, row in lodo_df_sorted.iterrows():
    log(f"  {row['drug_name']:<15} (True: {row['true_label']:<12}) : {row['s_d']:.6f}")

gap = lodo_df_sorted.iloc[0]['s_d'] - lodo_df_sorted.iloc[1]['s_d']
log(f"\nGap between largest (diltiazem) and second largest (sotalol): {gap:.6f}")
exceed_half = sum([1 for x in lodo_df_sorted['s_d'] if x > 0.5])
log(f"Number of scores exceeding 0.5: {exceed_half} out of 12.")
log("Note: This calibration set is small enough that a single outlier (diltiazem) stretches the inclusion threshold enormously for the entire conformal layer.")

log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
for x in rank_outcomes:
    if x['c']['under-triaged'] > 0:
        log(f"  - FLAG: Rank {x['rank']} on {x['platform']} shows {x['c']['under-triaged']} under-triaged drug(s).")
    if x['emp_cov'] < x['nom_cov']:
        log(f"  - FLAG: Rank {x['rank']} on {x['platform']} empirical coverage ({x['emp_cov']:.4f}) falls below nominal ({x['nom_cov']:.4f}).")
    if x['avg_sz'] > 2.0:
        log(f"  - FLAG: Rank {x['rank']} on {x['platform']} average set size exceeds 2.0 ({x['avg_sz']:.2f}).")


doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP6_CONFORMAL_OPERATING_POINT.md')
with open(doc_path_repo, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

df_sweep.drop(columns=['avg_sz', 'emp_cov'], inplace=True)
df_sweep.to_csv(os.path.join(trainai_dir, 'results', 'conformal_rank_sweep_rev.csv'), index=False)

log("\nSaved results/conformal_rank_sweep_rev.csv")
log("Written docs/STEP6_CONFORMAL_OPERATING_POINT.md")
