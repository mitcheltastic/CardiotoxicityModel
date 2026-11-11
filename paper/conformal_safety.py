import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

print("===== CONFORMAL SAFETY RE-EVALUATION =====")

q_hat_154 = 0.6275
biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
label_map = {'high': 0, 'intermediate': 1, 'low': 2}
inv_label_map = {0: 'High', 1: 'Int', 2: 'Low'}

bundle = joblib.load("./artifacts/deployable_chantest_3class.joblib")
model = bundle['model']
scaler = bundle['scaler']

datasets = ['chantest', 'nanion', 'manual']

def severity(c):
    # High (0) is most severe (3), Int (1) is 2, Low (2) is 1
    return {0: 3, 1: 2, 2: 1}[c]

def get_outcome(act, true_c):
    if act is None:
        return "abstain"
    if act == true_c:
        return "correct"
    if severity(act) > severity(true_c):
        return "over-called"
    return "under-triaged"

results_agg = []
audit_rows = []
coverage_rows = []

import matplotlib
matplotlib.use('Agg')

for ds in datasets:
    test_file = f"data/{ds}_testing.csv" if ds != "manual" else f"data/{ds}_testing(in).csv"
    df = pd.read_csv(test_file).dropna()
    X = df[biomarkers].values
    y_str = df['label'].values
    y = np.array([label_map[y_s.lower()] if isinstance(y_s, str) else y_s for y_s in y_str])
    drugs = df['drug_name'].values
    
    unique_drugs = sorted(list(set(drugs)))
    X_sc = scaler.transform(X)
    y_proba = model.predict_proba(X_sc)
    
    hard_correct = 0
    hard_over = 0
    hard_under = 0
    
    conf_correct = 0
    conf_over = 0
    conf_under = 0
    conf_abstain = 0
    total_conf_size = 0
    covered = 0
    
    for dr in unique_drugs:
        idx = (drugs == dr)
        true_c = y[idx][0]
        
        p_mean = y_proba[idx].mean(axis=0)
        hard_pred = p_mean.argmax()
        
        ho = get_outcome(hard_pred, true_c)
        if ho == 'correct': hard_correct += 1
        elif ho == 'over-called': hard_over += 1
        elif ho == 'under-triaged': hard_under += 1
        
        pred_set = [c for c in [0, 1, 2] if 1.0 - p_mean[c] <= q_hat_154]
        if len(pred_set) == 0:
            conf_act = None
        else:
            conf_act = sorted(pred_set, key=severity, reverse=True)[0]
            
        co = get_outcome(conf_act, true_c)
        if co == 'correct': conf_correct += 1
        elif co == 'over-called': conf_over += 1
        elif co == 'under-triaged': conf_under += 1
        elif co == 'abstain': conf_abstain += 1
        
        if true_c in pred_set:
            covered += 1
            
        total_conf_size += len(pred_set)
        
        set_str = "{" + ",".join([inv_label_map[c] for c in pred_set]) + "}"
        audit_rows.append({
            'Dataset': ds.upper(),
            'Drug': dr,
            'True': inv_label_map[true_c],
            'Hard Label': inv_label_map[hard_pred],
            'Conformal Set': set_str,
            'Conformal Action': inv_label_map[conf_act] if conf_act is not None else "ABSTAIN",
            'Outcome': co
        })
        
    avg_conf_size = total_conf_size / 16.0
    
    results_agg.append({
        'Dataset': ds.upper(),
        'System': 'hard-label (argmax)',
        'Correct': hard_correct,
        'Over-called': hard_over,
        'Under-triaged': hard_under,
        'Avg set size': 1.00
    })
    
    results_agg.append({
        'Dataset': ds.upper(),
        'System': 'conformal @ alpha=0.154',
        'Correct': conf_correct,
        'Over-called': conf_over,
        'Under-triaged': conf_under,
        'Avg set size': avg_conf_size
    })
    
    status = "in-distribution (calibration domain)" if ds == 'chantest' else "domain-shifted"
    coverage_rows.append({
        'Dataset': ds.upper(),
        'Status': status,
        'Empirical Coverage': f"{covered/16.0:.4f}",
        'Target 1-alpha': "0.8460",
        'Diff': f"{(covered/16.0) - 0.8460:+.4f}"
    })

print("\n=== OUTPUT 1: Headline Table ===")
df_agg = pd.DataFrame(results_agg)
print(df_agg.to_string(index=False))

print("\n=== OUTPUT 2: Per-Drug Audit (alpha=0.154) ===")
df_audit = pd.DataFrame(audit_rows)
for ds in ['CHANTEST', 'NANION', 'MANUAL']:
    print(f"\n[{ds}]")
    print(df_audit[df_audit['Dataset'] == ds].drop('Dataset', axis=1).to_string(index=False))

print("\n=== OUTPUT 3: Coverage Table ===")
df_cov = pd.DataFrame(coverage_rows)
print(df_cov.to_string(index=False))

print("\n=== OUTPUT 4: Degenerate-alpha Note ===")
print("Sorted calibration scores: 0.0181, 0.0878, 0.1084, 0.1634, 0.1714, 0.2967, 0.3939, 0.4584, 0.5599, 0.6149, 0.6275, 0.9898")
print("At alpha=1/13 (approx 0.077), the required quantile falls outside the empirical distribution of 12 scores (since ceil(13 x 0.923) = 12, it selects the maximum score of 0.9898 from diltiazem). Consequently, this threshold is completely determined by a single LODO outlier. Setting alpha=0.077 with only 12 calibration samples is effectively degenerate, producing loose, uninformative full-support prediction sets. This highlights a fundamental limitation: valid conformal prediction with small calibration sets cannot reliably bound error rates below 1/(n+1).")

print("\n=== OUTPUT 5: Figure Generation ===")

fig, ax = plt.subplots(figsize=(12, 7))

datasets_labels = ['CHANTEST', 'NANION', 'MANUAL']
x = np.arange(len(datasets_labels))
width = 0.35

hard_correct = df_agg[df_agg['System'].str.contains('hard')]['Correct'].values
hard_over = df_agg[df_agg['System'].str.contains('hard')]['Over-called'].values
hard_under = df_agg[df_agg['System'].str.contains('hard')]['Under-triaged'].values

conf_correct = df_agg[df_agg['System'].str.contains('conformal')]['Correct'].values
conf_over = df_agg[df_agg['System'].str.contains('conformal')]['Over-called'].values
conf_under = df_agg[df_agg['System'].str.contains('conformal')]['Under-triaged'].values

ax.bar(x - width/2, hard_correct, width, label='Correct (Hard)', color='#4CAF50')
ax.bar(x - width/2, hard_over, width, bottom=hard_correct, label='Over-called (Hard)', color='#FFC107')
ax.bar(x - width/2, hard_under, width, bottom=hard_correct+hard_over, label='Under-triaged (Hard)', color='#F44336', hatch='//')

ax.bar(x + width/2, conf_correct, width, label='Correct (Conf)', color='#81C784')
ax.bar(x + width/2, conf_over, width, bottom=conf_correct, label='Over-called (Conf)', color='#FFE082')
ax.bar(x + width/2, conf_under, width, bottom=conf_correct+conf_over, label='Under-triaged (Conf)', color='#EF9A9A', hatch='\\\\')

ax.set_ylabel('Number of Drugs (out of 16)', fontsize=12)
ax.set_title('Clinical Safety Outcomes: Hard-Label vs Split-Conformal Prediction (α=0.154)', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(datasets_labels, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Add value labels
for i, v in enumerate(hard_under):
    if v > 0:
        ax.text(x[i] - width/2, hard_correct[i] + hard_over[i] + v/2, str(v), color='white', fontweight='bold', ha='center', va='center')

for i, v in enumerate(conf_under):
    if v > 0:
        ax.text(x[i] + width/2, conf_correct[i] + conf_over[i] + v/2, str(v), color='white', fontweight='bold', ha='center', va='center')

plt.tight_layout()
fig_path = os.path.abspath('./figures/conformal_safety_comparison.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure successfully saved to {fig_path}")
