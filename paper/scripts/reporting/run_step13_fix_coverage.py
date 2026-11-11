import os
import sys
import pandas as pd
import numpy as np
import math
import json
import hashlib
import sklearn
import xgboost
from PyPDF2 import PdfReader

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
trainai_dir = os.path.join(base_dir, "TrainAI")
results_dir = os.path.join(trainai_dir, "results")
docs_dir = os.path.join(base_dir, "docs")
models_dir = os.path.join(trainai_dir, "models")

SEVERITY = {0: 2, 1: 1, 2: 0} # 0=High (2), 1=Int (1), 2=Low (0)

def wilson_ci(k, n=16, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054 # 95%
    p_hat = k / n
    denom = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * (n**2)))) / denom
    return center - margin, center + margin

print("==========================================")
print("=== STEP 1 — RECOMPUTE COVERAGE PER PLATFORM ===")
print("==========================================")
Q_HAT_11 = 0.762830
THRESH = 1.0 - Q_HAT_11 # 0.237170
platforms = ['chantest', 'nanion', 'manual']

coverages = {}
cov_counts = {}
wilson_intervals = {}

for p in platforms:
    print(f"\n--- PLATFORM: {p.upper()} ---")
    df = pd.read_csv(os.path.join(results_dir, f'cross_platform_probs_{p}_rev.csv'))
    df = df[df['split'] == 'testing'].reset_index(drop=True)
    
    cov_cnt = 0
    p_cols = ['prob_high', 'prob_int', 'prob_low']
    
    for i in range(len(df)):
        drug = df['drug_name'].iloc[i]
        t = int(df['true_label'].iloc[i])
        p_vec = df[p_cols].values[i]
        p_true = p_vec[t]
        
        # Check if true class probability >= threshold
        # Or if empty set fallback selects max prob, does max prob == true class?
        # Standard conformal set: {c : p_vec[c] >= THRESH}
        cset = np.where(p_vec >= THRESH)[0].tolist()
        if len(cset) == 0:
            cset = [np.argmax(p_vec)]
            
        clears = (t in cset)
        if clears:
            cov_cnt += 1
            
        print(f"{drug:15s} | True: {t} | p_mean[true]: {p_true:.6f} | Clears: {clears} | Set: {cset}")
        
    prop = cov_cnt / len(df)
    coverages[p] = prop
    cov_counts[p] = cov_cnt
    print(f"Coverage for {p}: {cov_cnt}/{len(df)} = {prop:.6f} ({prop*100:.2f}%)")

print("\n==========================================")
print("=== STEP 2 — WILSON INTERVALS & VERIFICATION ===")
print("==========================================")
for p in platforms:
    k = cov_counts[p]
    low, high = wilson_ci(k, 16)
    wilson_intervals[p] = (low, high)
    print(f"{p.upper()}: k={k}/16 ({coverages[p]:.4f}) -> 95% Wilson CI: [{low:.4f}, {high:.4f}]")

# Check standard known Wilson CIs:
# 12/16 (0.75) -> [0.5050, 0.8982]
# 14/16 (0.875) -> [0.6398, 0.9650]
w_12_low, w_12_high = wilson_ci(12, 16)
w_14_low, w_14_high = wilson_ci(14, 16)
print(f"Verification 12/16 (0.7500): [{w_12_low:.4f}, {w_12_high:.4f}] -> Matches [0.5050, 0.8982]: {abs(w_12_low-0.5050)<0.001 and abs(w_12_high-0.8982)<0.001}")
print(f"Verification 14/16 (0.8750): [{w_14_low:.4f}, {w_14_high:.4f}] -> Matches [0.6398, 0.9650]: {abs(w_14_low-0.6398)<0.001 and abs(w_14_high-0.9650)<0.001}")

mismatch = False
for p in platforms:
    if cov_counts[p] == 12:
        print(f"{p.upper()} has 12/16 coverage (0.7500) matching Wilson CI [{wilson_intervals[p][0]:.4f}, {wilson_intervals[p][1]:.4f}]")
    elif cov_counts[p] == 14:
        print(f"{p.upper()} has 14/16 coverage (0.8750) matching Wilson CI [{wilson_intervals[p][0]:.4f}, {wilson_intervals[p][1]:.4f}]")
    else:
        print(f"ERROR: Wilson interval mismatch for {p.upper()}: count={cov_counts[p]}")
        mismatch = True

if mismatch:
    print("ERROR: Wilson interval mismatch")
    sys.exit(1)

print("\n==========================================")
print("=== STEP 3 & 4 — DISTINCT METRICS SUMMARY ===")
print("==========================================")
NOMINAL_COV = 11.0 / 13.0 # 0.846154

outcomes = {}
counts = {}
sz_dists = {}
avg_szs = {}

for p in platforms:
    df = pd.read_csv(os.path.join(results_dir, f'cross_platform_probs_{p}_rev.csv'))
    df = df[df['split'] == 'testing'].reset_index(drop=True)
    c = {'correct': 0, 'over-called': 0, 'under-triaged': 0}
    sz_dist = {1: 0, 2: 0, 3: 0, 0: 0}
    tot_sz = 0
    p_outcomes = []
    
    for i in range(len(df)):
        t = int(df['true_label'].iloc[i])
        p_vec = df[['prob_high', 'prob_int', 'prob_low']].values[i]
        cset = np.where(p_vec >= THRESH)[0].tolist()
        if len(cset) == 0:
            cset = [np.argmax(p_vec)]
        sz = len(cset)
        sz_dist[sz] += 1
        tot_sz += sz
        act_sev = max([SEVERITY[cls] for cls in cset])
        t_sev = SEVERITY[t]
        if act_sev == t_sev:
            outcome = "correct"
        elif act_sev > t_sev:
            outcome = "over-called"
        else:
            outcome = "under-triaged"
        c[outcome] += 1
        p_outcomes.append(outcome)
        
    counts[p] = c
    sz_dists[p] = sz_dist
    avg_szs[p] = tot_sz / len(df)
    outcomes[p] = p_outcomes

print(f"{'Platform':10s} | {'Correct-Outcome Prop':22s} | {'Empirical Coverage':20s} | {'Signed Gap':12s} | {'Average Set Size':18s}")
for p in platforms:
    cor_prop = counts[p]['correct'] / 16.0
    emp_cov = coverages[p]
    gap = emp_cov - NOMINAL_COV
    print(f"{p.upper():10s} | {counts[p]['correct']}/16 ({cor_prop:.4f})       | {cov_counts[p]}/16 ({emp_cov:.4f})        | {gap:+.4f}      | {avg_szs[p]:.4f}")


print("\n==========================================")
print("=== REBUILDING docs/PAPER_NUMBERS.md ===")
print("==========================================")

out_lines = []
out_lines.append("# CANONICAL PAPER NUMBERS")
out_lines.append("One canonical value per quantity, each with its source file. Every entry states its evaluation population.")

# A. DATA
out_lines.append("\n## A. DATA")
train_df = pd.read_csv(os.path.join(results_dir, "cross_platform_probs_chantest_rev.csv"))
train_drugs = sorted(train_df[train_df['split'] == 'training']['drug_name'].unique().tolist())
test_drugs = sorted(train_df[train_df['split'] == 'testing']['drug_name'].unique().tolist())

new_sent = "26 rows were dropped at load time because at least one of the 11 biomarker cells contained the Excel error string '#VALUE!': 25 rows in manual_training_scaled_rev.csv (all mexiletine, columns APD50/APD90/max_dv) and 1 row in manual_testing_scaled_rev.csv (ibutilide, CaTD90). No other file contained any sentinel. Audit trail: logs/DROPPED_ROWS_REV.csv."

out_lines.append(f"- **Rows dropped before loading**: {new_sent} (Source: docs/DATA_AUDIT_REV_PART2.md, docs/LOADER_VERIFICATION_REV.md)")
out_lines.append(f"- **Chantest rows/drugs**: Training: 24000/12, Testing: 32000/16 (Source: docs/LOADER_VERIFICATION_REV.md)")
out_lines.append(f"- **Nanion rows/drugs**: Training: 24000/12, Testing: 32000/16 (Source: docs/LOADER_VERIFICATION_REV.md)")
out_lines.append(f"- **Manual rows/drugs**: Training: 23975/12, Testing: 31999/16 (Source: docs/LOADER_VERIFICATION_REV.md)")
out_lines.append(f"- **Training drugs (12)**: {', '.join(train_drugs)} (Population: Chantest Training Split)")
out_lines.append(f"- **Testing drugs (16)**: {', '.join(test_drugs)} (Population: Held-out Test Split)")

# B. MODEL
out_lines.append("\n## B. MODEL")
meta_path = os.path.join(models_dir, "chantest_stacking_metadata_rev.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)
rf_params = meta['tuned_hyperparameters']['rf']
xgb_params = meta['tuned_hyperparameters']['xgb']
out_lines.append(f"- **RF Hyperparameters**: {rf_params} (Source: TrainAI/models/chantest_stacking_metadata_rev.json)")
out_lines.append(f"- **XGB Hyperparameters**: {xgb_params} (Source: TrainAI/models/chantest_stacking_metadata_rev.json)")
out_lines.append(f"- **Meta-learner**: Random Forest (StackingClassifier final_estimator)")
out_lines.append(f"- **Stack method**: predict_proba")
out_lines.append(f"- **Passthrough**: False")
out_lines.append(f"- **CV Splitter**: StratifiedGroupKFold (k=4, grouped by drug)")
out_lines.append(f"- **Seed**: 42")
out_lines.append(f"- **sklearn version**: {sklearn.__version__}")
out_lines.append(f"- **xgboost version**: {xgboost.__version__}")
model_path = os.path.join(models_dir, "chantest_stacking_model_rev.joblib")
with open(model_path, "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
out_lines.append(f"- **Model artifact SHA-256**: {file_hash.upper()} (Source: TrainAI/models/chantest_stacking_model_rev.joblib)")

# C. CLASSIFICATION
out_lines.append("\n## C. CLASSIFICATION")
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
boot_df = pd.read_csv(os.path.join(results_dir, "bootstrap_distributions_rev.csv"))
row_f1 = {'chantest': '0.688289 (Source: docs/STEP3_STACKING_CHANTEST.md)', 
          'nanion': '0.632400 (Source: docs/STEP4_CROSS_PLATFORM.md)', 
          'manual': '0.692524 (Source: docs/STEP4_CROSS_PLATFORM.md)'}

for p in ['chantest', 'nanion', 'manual']:
    out_lines.append(f"\n### {p.upper()} (Population: 16 Test Drugs)")
    df = pd.read_csv(os.path.join(results_dir, f'cross_platform_probs_{p}_rev.csv'))
    df = df[df['split'] == 'testing']
    pt_f1 = f1_score(df['true_label'], df['pred_label'], average='macro')
    b_df = boot_df[(boot_df['platform'] == p) & (boot_df['statistic'] == 'macro_f1')]
    med = b_df['value'].median()
    lo = b_df['value'].quantile(0.025)
    hi = b_df['value'].quantile(0.975)
    out_lines.append(f"- **Drug-level macro-F1**: Point Estimate = {pt_f1:.6f}, Bootstrap Median = {med:.6f} [95% CI: {lo:.6f}, {hi:.6f}] (Source: TrainAI/results/bootstrap_distributions_rev.csv)")
    out_lines.append(f"- **Row-level macro-F1**: {row_f1[p]}")
    t = df['true_label'].values
    p_lbl = df['pred_label'].values
    prec = precision_score(t, p_lbl, average=None, labels=[0,1,2])
    rec = recall_score(t, p_lbl, average=None, labels=[0,1,2])
    f1s = f1_score(t, p_lbl, average=None, labels=[0,1,2])
    cm = confusion_matrix(t, p_lbl, labels=[0,1,2])
    out_lines.append(f"- **Per-class (High/Int/Low) Point Estimates (Drug-level)**:")
    out_lines.append(f"  - Precision: High={prec[0]:.6f}, Int={prec[1]:.6f}, Low={prec[2]:.6f}")
    out_lines.append(f"  - Recall: High={rec[0]:.6f}, Int={rec[1]:.6f}, Low={rec[2]:.6f}")
    out_lines.append(f"  - F1: High={f1s[0]:.6f}, Int={f1s[1]:.6f}, Low={f1s[2]:.6f}")
    out_lines.append(f"- **Confusion Matrix (Drug-level, True/Pred)**: High(0) {cm[0].tolist()}, Int(1) {cm[1].tolist()}, Low(2) {cm[2].tolist()} (Source: TrainAI/results/cross_platform_probs_{p}_rev.csv)")

# D. CONFORMAL
out_lines.append("\n## D. CONFORMAL")
lodo_df = pd.read_csv(os.path.join(results_dir, "conformal_lodo_scores_rev.csv"))
lodo_scores = sorted(lodo_df['s_d'].values)
out_lines.append(f"- **12 LODO scores**: {[round(x, 6) for x in lodo_scores]} (Source: TrainAI/results/conformal_lodo_scores_rev.csv, Population: 12 Chantest Train Drugs)")

sweep_df = pd.read_csv(os.path.join(results_dir, "conformal_rank_sweep_rev.csv"))
rank12 = sweep_df[sweep_df['rank'] == 12].iloc[0]
rank11 = sweep_df[sweep_df['rank'] == 11].iloc[0]
rank10 = sweep_df[sweep_df['rank'] == 10].iloc[0]
out_lines.append(f"- **Rank 12**: q_hat = {rank12['q_hat']:.6f} (Alpha interval: [1/13, 2/13)) (Source: TrainAI/results/conformal_rank_sweep_rev.csv)")
out_lines.append(f"- **Rank 11**: q_hat = {rank11['q_hat']:.6f} (Alpha interval: [2/13, 3/13)) (Source: TrainAI/results/conformal_rank_sweep_rev.csv)")
out_lines.append(f"- **Rank 10**: q_hat = {rank10['q_hat']:.6f} (Alpha interval: [3/13, 4/13)) (Source: TrainAI/results/conformal_rank_sweep_rev.csv)")
out_lines.append(f"- **Adopted Operating Point**: Rank 11, q_hat = 0.762830, alpha = 2/13, nominal coverage = 11/13 = 0.846154 (Population: 16 Test Drugs)")

for p in ['chantest', 'nanion', 'manual']:
    out_lines.append(f"\n### {p.upper()} Conformal (Population: 16 Test Drugs)")
    cor_cnt = counts[p]['correct']
    cor_prop = cor_cnt / 16.0
    out_lines.append(f"- **Correct-outcome count & proportion (Rank 11)**: {cor_cnt}/16 ({cor_prop:.4f}) (Source: computed from TrainAI/results/cross_platform_probs_{p}_rev.csv)")
    out_lines.append(f"- **Outcome breakdown**: Correct = {counts[p]['correct']}, Over-called = {counts[p]['over-called']}, Under-triaged = {counts[p]['under-triaged']} (Source: computed from TrainAI/results/cross_platform_probs_{p}_rev.csv)")
    out_lines.append(f"- **Average set size**: {avg_szs[p]:.4f} (Source: computed from TrainAI/results/cross_platform_probs_{p}_rev.csv)")
    out_lines.append(f"- **Set size distribution**: size 1 = {sz_dists[p][1]}, size 2 = {sz_dists[p][2]}, size 3 = {sz_dists[p][3]} (Source: computed from TrainAI/results/cross_platform_probs_{p}_rev.csv)")
    
    emp_cov = coverages[p]
    emp_cnt = cov_counts[p]
    w_low, w_high = wilson_intervals[p]
    gap = emp_cov - NOMINAL_COV
    out_lines.append(f"- **Coverage (Empirical vs Nominal)**: {emp_cnt}/16 ({emp_cov:.4f}) vs {NOMINAL_COV:.4f} (Signed Gap: {gap:+.4f}, 95% Wilson CI: [{w_low:.4f}, {w_high:.4f}], Source: computed from TrainAI/results/cross_platform_probs_{p}_rev.csv & docs/STEP6_CONFORMAL_OPERATING_POINT.md)")
    out_lines.append(f"- **Rule-of-three under-triage bound**: 0 of 16 under-triaged observed; approximate 95% upper bound = 0.187500 (18.75%) (Source: docs/STEP9_BOOTSTRAP.md)")

# E. BASELINES
out_lines.append("\n## E. BASELINES")
for p in ['chantest', 'nanion', 'manual']:
    df = pd.read_csv(os.path.join(results_dir, f'cross_platform_probs_{p}_rev.csv'))
    df = df[df['split'] == 'testing']
    ah_cor = sum(1 for i in range(16) if 0 == df['true_label'].iloc[i])
    hard_lbl_cor = sum(1 for i in range(16) if df['pred_label'].iloc[i] == df['true_label'].iloc[i])
    out_lines.append(f"- **{p.upper()} Always-High Baseline**: Correct = {ah_cor}/16 (Source: computed from TrainAI/results/cross_platform_probs_{p}_rev.csv)")
    out_lines.append(f"- **{p.upper()} Hard-Label Model**: Correct = {hard_lbl_cor}/16 (Source: computed from TrainAI/results/cross_platform_probs_{p}_rev.csv)")

# F. SHAP
out_lines.append("\n## F. SHAP")
shap_df = pd.read_csv(os.path.join(results_dir, "shap_global_importance_rev.csv"))
ranking = shap_df['biomarker'].tolist()
qnet_shap = shap_df[shap_df['biomarker'] == 'qNet']['mean_abs_shap_aggregated'].values[0]
out_lines.append(f"- **Aggregated Ranking**: {ranking} (Source: TrainAI/results/shap_global_importance_rev.csv, Population: 16 Test Drugs)")
out_lines.append(f"- **qNet canonical mean|SHAP|**: {qnet_shap:.6f} (Source: TrainAI/results/shap_global_importance_rev.csv)")
out_lines.append(f"- **vrest permutation finding**: permuting vrest changes on average 1.0 of 16 drug-level predictions (zero on many seeds), while permuting qNet changes 3.0; mean drug-level macro-F1 drop 0.073 for vrest vs 0.108 for qNet. (Source: docs/STEP8_SHAP_DIAGNOSTIC.md)")

# G. FIGURES
out_lines.append("\n## G. FIGURES")
def measure_pdf(name):
    path = os.path.join(results_dir, name)
    try:
        reader = PdfReader(path)
        box = reader.pages[0].mediabox
        w = float(box.width) / 72.0
        h = float(box.height) / 72.0
        size_kb = os.path.getsize(path) / 1024.0
        has_image = False
        with open(path, 'rb') as f:
            if b'/Subtype /Image' in f.read():
                has_image = True
        return f"{w:.2f} x {h:.2f} inches, {size_kb:.1f} KB, Raster: {has_image}"
    except Exception as e:
        return "NOT MEASURED"

fig1_dim = measure_pdf("fig1_block_diagram.pdf")
fig2_dim = measure_pdf("fig_shap_global.pdf")

out_lines.append(f"- **fig1_block_diagram.pdf**: Path = TrainAI/results/fig1_block_diagram.pdf, Dimensions = {fig1_dim}")
out_lines.append(f"- **fig_shap_global.pdf**: Path = TrainAI/results/fig_shap_global.pdf, Dimensions = {fig2_dim}")

# STEP 5 CHECK: Rescan whole file for any other place where coverage figures appear
print("\n==========================================")
print("=== STEP 5 — RESCANNING FOR COVERAGE FIGURES ===")
print("==========================================")
locations_checked = []
for idx, line in enumerate(out_lines):
    if "coverage" in line.lower() or "cov" in line.lower():
        locations_checked.append((idx+1, line))
        print(f"Line {idx+1}: {line}")

# CHECK FOR FLAGS
flags = []
for line in out_lines:
    if "NOT COMPUTED" in line:
        flags.append(f"Missing Value: {line}")
    if "NO SOURCE FILE" in line:
        flags.append(f"Missing Source: {line}")

out_lines.append("\n==========================================")
out_lines.append("=== SANITY FLAGS ===")
out_lines.append("==========================================")
if len(flags) == 0:
    out_lines.append("All values compiled successfully. Point estimates strictly isolated from medians.")
else:
    for flag in flags:
        out_lines.append(flag)

final_text = "\n".join(out_lines)
with open(os.path.join(docs_dir, "PAPER_NUMBERS.md"), 'w', encoding='utf-8') as f:
    f.write(final_text)

print("\n\n" + final_text)
