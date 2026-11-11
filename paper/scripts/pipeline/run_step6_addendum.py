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
log("=== ADDENDUM — CORRECTED BOUNDARY AND FINAL TABLE ===")
log("==========================================")

log("\n=== STEP 1 — FIX THE BOUNDARY LINE ===")
sweep_df = pd.read_csv(os.path.join(trainai_dir, 'results', 'conformal_rank_sweep_rev.csv'))

# Evaluate under-triages per rank
ranks_present = sorted(sweep_df['rank'].unique(), reverse=True)

rank_ut = {}
for r in ranks_present:
    r_df = sweep_df[sweep_df['rank'] == r]
    q_hat = r_df['q_hat'].iloc[0]
    
    ut_c = len(r_df[(r_df['platform'] == 'chantest') & (r_df['outcome'] == 'under-triaged')])
    ut_n = len(r_df[(r_df['platform'] == 'nanion') & (r_df['outcome'] == 'under-triaged')])
    ut_m = len(r_df[(r_df['platform'] == 'manual') & (r_df['outcome'] == 'under-triaged')])
    
    rank_ut[r] = {'q': q_hat, 'c': ut_c, 'n': ut_n, 'm': ut_m}
    log(f"rank {r} (q_hat={q_hat:.6f}): under-triage count for chantest={ut_c} / nanion={ut_n} / manual={ut_m}")

tightest_safe_rank = None
for r in ranks_present:
    if rank_ut[r]['c'] == 0 and rank_ut[r]['n'] == 0 and rank_ut[r]['m'] == 0:
        tightest_safe_rank = r
        
tighter_rank = tightest_safe_rank - 1

log(f"\nTIGHTEST (smallest q_hat) rank at which under-triage is zero on ALL three platforms: Rank {tightest_safe_rank} (q_hat = {rank_ut[tightest_safe_rank]['q']:.6f})")
log(f"The next tighter rank is Rank {tighter_rank} (q_hat = {rank_ut[tighter_rank]['q']:.6f}) with under-triages: chantest={rank_ut[tighter_rank]['c']}, nanion={rank_ut[tighter_rank]['n']}, manual={rank_ut[tighter_rank]['m']}.")

err_before = "SMALLEST q_hat (tightest rank) at which under-triage is zero on ALL three platforms is 0.986695 at rank 12."
err_after = f"SMALLEST q_hat (tightest rank) at which under-triage is zero on ALL three platforms is {rank_ut[tightest_safe_rank]['q']:.6f} at rank {tightest_safe_rank}."

log("\nERRONEOUS SENTENCE (BEFORE):")
log(f'"{err_before}"')
log("CORRECTED REPLACEMENT (AFTER):")
log(f'"{err_after}"')


log("\n=== STEP 2 — PER-PLATFORM DETAIL AT RANK 12 AND RANK 11 ===")
platforms = ['chantest', 'nanion', 'manual']

SEVERITY = {0: 2, 1: 1, 2: 0}
LABEL_MAP = {"high": 0, "intermediate": 1, "low": 2}
INV_LABEL_MAP = {0: "high", 1: "intermediate", 2: "low"}

def compute_rank_stats(plat, r):
    df_p = sweep_df[(sweep_df['platform'] == plat) & (sweep_df['rank'] == r)]
    c = {'correct': 0, 'over-called': 0, 'under-triaged': 0, 'abstain': 0}
    sz_dist = {1: 0, 2: 0, 3: 0, 0: 0}
    tot_sz = 0
    emp_cov = 0
    
    for _, row in df_p.iterrows():
        p_set = ast.literal_eval(row['set'])
        c[row['outcome']] += 1
        sz = len(p_set)
        sz_dist[sz] += 1
        tot_sz += sz
        if row['true_label'] in p_set:
            emp_cov += 1
            
    q = df_p['q_hat'].iloc[0]
    min_alpha = 1.0 - (r / 13.0)
    nom_cov = 1.0 - min_alpha
    emp_frac = emp_cov / 16.0
    gap = emp_frac - nom_cov
    avg_sz = tot_sz / 16.0
    pct_3 = (sz_dist[3] / 16.0) * 100
    
    return {
        'cor': c['correct'], 'ovr': c['over-called'], 'und': c['under-triaged'], 'abs': c['abstain'],
        'avg_sz': avg_sz, 'sz1': sz_dist[1], 'sz2': sz_dist[2], 'sz3': sz_dist[3], 'pct3': pct_3,
        'emp_c': emp_cov, 'emp_f': emp_frac, 'nom': nom_cov, 'gap': gap, 'min_alpha': min_alpha, 'q': q
    }

log(f"{'Platform':<10} | {'Rank':<7} | {'Cor':<3} | {'Ovr':<3} | {'Und':<3} | {'Abs':<3} | {'AvgSz':<7} | {'Sz1':<3} | {'Sz2':<3} | {'Sz3':<3} | {'%Sz3':<6} | {'EmpCov':<11} | {'NomCov':<7} | {'Gap':<7}")
log("-" * 125)

rank_stats = {}
for p in platforms:
    rank_stats[p] = {}
    for r in [12, 11, 10]:
        if r in ranks_present:
            st = compute_rank_stats(p, r)
            rank_stats[p][r] = st
            if r in [12, 11]:
                log(f"{p:<10} | {r:<7} | {st['cor']:<3} | {st['ovr']:<3} | {st['und']:<3} | {st['abs']:<3} | {st['avg_sz']:<7.4f} | {st['sz1']:<3} | {st['sz2']:<3} | {st['sz3']:<3} | {st['pct3']:<5.1f}% | {st['emp_c']}/16 ({st['emp_f']*100:.1f}%) | {st['nom']:<7.4f} | {st['gap']:+.4f}")


log("\n=== STEP 3 — COVERAGE MISSES AT RANK 11 ===")
miss_counts = {}
for p in platforms:
    log(f"\n--- Misses on {p.upper()} at Rank 11 ---")
    df_p = sweep_df[(sweep_df['platform'] == p) & (sweep_df['rank'] == 11)]
    q_hat = df_p['q_hat'].iloc[0]
    thresh = 1.0 - q_hat
    
    misses = []
    for _, row in df_p.iterrows():
        p_set = ast.literal_eval(row['set'])
        if row['true_label'] not in p_set:
            misses.append(row)
            
    m_types = {'over-called': 0, 'under-triaged': 0, 'abstain': 0}
    for m in misses:
        t_sev = SEVERITY[LABEL_MAP[m['true_label']]]
        if m['action'] == 'abstain':
            a_sev = -1
        else:
            a_sev = SEVERITY[LABEL_MAP[m['action']]]
        
        m_types[m['outcome']] += 1
        
        log(f"{m['drug_name']}")
        log(f"  True Label: {m['true_label']}")
        log(f"  Probs: p_high={m['p_high']:.6f} / p_int={m['p_int']:.6f} / p_low={m['p_low']:.6f}")
        log(f"  Threshold (1-q_hat): {thresh:.6f}")
        log(f"  Set: {m['set']}")
        log(f"  Action: {m['action']}")
        log(f"  Severity: Action({a_sev}) vs True({t_sev})")
        log(f"  Outcome: {m['outcome']}")
        
    log(f"Summary for {p}: ALL coverage misses are over-calls? {'YES' if m_types['under-triaged'] == 0 else 'NO'}. Over-calls: {m_types['over-called']}, Under-triages: {m_types['under-triaged']}")
    if m_types['under-triaged'] > 0:
        log("ERROR: under-triage count inconsistent with Step 2")
        sys.exit(1)


log("\n=== STEP 4 — COVERAGE UNCERTAINTY ===")
z = 1.95996 # 95% CI approx 1.96

for r in [12, 11]:
    log(f"\n--- Rank {r} ---")
    for p in platforms:
        st = rank_stats[p][r]
        p_hat = st['emp_f']
        n = 16.0
        
        # Wilson score interval formula
        denominator = 1 + z**2 / n
        center = p_hat + z**2 / (2 * n)
        spread = z * math.sqrt((p_hat * (1 - p_hat)) / n + z**2 / (4 * n**2))
        
        lower = (center - spread) / denominator
        upper = (center + spread) / denominator
        
        nom_cov = st['nom']
        inside = (nom_cov >= lower) and (nom_cov <= upper)
        log(f"{p:<10} | Emp: {p_hat:.4f} | 95% Wilson CI: [{lower:.4f}, {upper:.4f}] | Nominal {nom_cov:.4f} inside? {'YES' if inside else 'NO'}")


log("\n=== STEP 5 — THE RANK-10 BOUNDARY EVIDENCE ===")
log("At rank 10 (q_hat = 0.703345), identifying drugs that become under-triaged:")
r10_q = rank_stats['chantest'][10]['q']
thresh10 = 1.0 - r10_q

for p in platforms:
    df_p10 = sweep_df[(sweep_df['platform'] == p) & (sweep_df['rank'] == 10)]
    for _, row in df_p10.iterrows():
        if row['outcome'] == 'under-triaged':
            t_sev = SEVERITY[LABEL_MAP[row['true_label']]]
            
            p_vec = [row['p_high'], row['p_int'], row['p_low']]
            true_p = p_vec[LABEL_MAP[row['true_label']]]
            
            # We want to find the margin by which the severe class fell below threshold.
            # Wait, "the severe class" means the true class? Or the class that would have saved it?
            # E.g., true is high (sev 2), action is int (sev 1). High fell below threshold.
            # Margin = (1 - p_high) - q_hat
            
            severe_class_p = p_vec[LABEL_MAP[row['true_label']]] # assuming true is the most severe class it missed
            # But what if true is int, and action is low? Then severe class is int.
            # So margin is (1 - true_p) - q_hat
            margin = (1.0 - severe_class_p) - r10_q
            
            log(f"Drug: {row['drug_name']} on {p}")
            log(f"  True label: {row['true_label']}")
            log(f"  p_mean: p_high={row['p_high']:.6f}, p_int={row['p_int']:.6f}, p_low={row['p_low']:.6f}")
            log(f"  Threshold: {thresh10:.6f}")
            log(f"  Set: {row['set']}")
            log(f"  Action: {row['action']}")
            log(f"  Margin by which severe class ({row['true_label']}) fell below threshold: {margin:.6f}")


log("\n=== STEP 6 — FINAL SUMMARY TABLE FOR THE PAPER ===")
log(f"{'Platform':<10} | {'Method (Rank/Base)':<20} | {'q_hat':<10} | {'Min Alpha':<10} | {'NomCov':<8} | {'EmpCov':<8} | {'Cor':<4} | {'Ovr':<4} | {'Und':<4} | {'AvgSz':<6} | % Sz-3")
log("-" * 125)

for p in platforms:
    # Recompute baselines
    prob_df = pd.read_csv(os.path.join(trainai_dir, 'results', f'cross_platform_probs_{p}_rev.csv'))
    prob_df = prob_df[prob_df['split'] == 'testing'].copy()
    
    ah_c = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    hl_c = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    for _, row in prob_df.iterrows():
        t_sev = SEVERITY[int(row['true_label'])]
        if SEVERITY[0] == t_sev: ah_c['correct'] += 1
        elif SEVERITY[0] > t_sev: ah_c['over-called'] += 1
        else: ah_c['under-triaged'] += 1
        
        act_sev_hl = SEVERITY[int(row['pred_label'])]
        if act_sev_hl == t_sev: hl_c['correct'] += 1
        elif act_sev_hl > t_sev: hl_c['over-called'] += 1
        else: hl_c['under-triaged'] += 1
        
    log(f"{p:<10} | {'ALWAYS-HIGH':<20} | {'-':<10} | {'-':<10} | {'-':<8} | {'-':<8} | {ah_c['correct']:<4} | {ah_c['over-called']:<4} | {ah_c['under-triaged']:<4} | {'1.0000':<6} | 0.0%")
    log(f"{'':<10} | {'HARD-LABEL':<20} | {'-':<10} | {'-':<10} | {'-':<8} | {'-':<8} | {hl_c['correct']:<4} | {hl_c['over-called']:<4} | {hl_c['under-triaged']:<4} | {'1.0000':<6} | 0.0%")
    
    for r in [12, 11, 10]:
        st = rank_stats[p][r]
        log(f"{'':<10} | {f'Rank {r}':<20} | {st['q']:<10.6f} | {st['min_alpha']:<10.6f} | {st['nom']:<8.4f} | {st['emp_f']:<8.4f} | {st['cor']:<4} | {st['ovr']:<4} | {st['und']:<4} | {st['avg_sz']:<6.4f} | {st['pct3']:.1f}%")

log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
flags_raised = False
for p in platforms:
    st11 = rank_stats[p][11]
    p_hat = st11['emp_f']
    n = 16.0
    denominator = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat)) / n + z**2 / (4 * n**2))
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    
    nom_cov11 = st11['nom']
    if not (lower <= nom_cov11 <= upper):
        log(f"  - FLAG: On {p} at Rank 11, nominal coverage ({nom_cov11:.4f}) falls OUTSIDE the Wilson interval [{lower:.4f}, {upper:.4f}].")
        flags_raised = True
        
    if st11['pct3'] > 25.0:
        log(f"  - FLAG: On {p} at Rank 11, percent size-3 sets ({st11['pct3']:.1f}%) exceeds 25%.")
        flags_raised = True

if not flags_raised:
    log("  - CLEAR: No sanity flags raised for this section.")

doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP6_CONFORMAL_OPERATING_POINT.md')
with open(doc_path_repo, 'a', encoding='utf-8') as f:
    f.write("\n" + "\n".join(output_lines))

log("\nAppended to docs/STEP6_CONFORMAL_OPERATING_POINT.md")
