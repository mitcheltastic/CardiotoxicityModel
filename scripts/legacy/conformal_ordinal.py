import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import math
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
import sys

print("===== ORDINAL CONFORMAL SAFETY EVALUATION =====")

def severity(c_orig):
    return {0: 2, 1: 1, 2: 0}[c_orig]

def calc_score(p_mean, cand_c_orig):
    cand_sev = severity(cand_c_orig)
    score = 0
    for c in [0, 1, 2]:
        score += p_mean[c] * abs(severity(c) - cand_sev)
    return score

print("\n--- Step 1: LODO Calibration Scores ---")
biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
df_train = pd.read_csv("data/chantest_training.csv").dropna()
X_all = df_train[biomarkers].values
label_map = {'high': 0, 'intermediate': 1, 'low': 2}
inv_label_map = {0: 'High', 1: 'Int', 2: 'Low'}

y_str = df_train['label'].values
y_all = np.array([label_map[y_s.lower()] if isinstance(y_s, str) else y_s for y_s in y_str])
drugs_all = df_train['drug_name'].values

train_drugs = sorted(list(set(drugs_all)))

class FixedGroupsCV:
    def __init__(self, n_splits, groups):
        self.cv = StratifiedGroupKFold(n_splits=n_splits)
        self.groups = groups
    def split(self, X, y, groups=None):
        return self.cv.split(X, y, self.groups)
    def get_n_splits(self, X, y, groups=None):
        return self.cv.n_splits

scores = []
print(f"{'Drug Name':<16} | {'True':<5} | {'P(High)':<8} | {'P(Int)':<8} | {'P(Low)':<8} | {'Score s':<8}")
print("-" * 70)

for dr in train_drugs:
    test_idx = (drugs_all == dr)
    train_idx = ~test_idx
    
    X_tr, y_tr, drugs_tr = X_all[train_idx], y_all[train_idx], drugs_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]
    true_label = y_te[0]
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    meta = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    model = StackingClassifier(
        estimators=[('RF', rf), ('XGB', xgb)],
        final_estimator=meta,
        cv=FixedGroupsCV(4, drugs_tr)
    )
    model.fit(X_tr_sc, y_tr)
    
    y_proba = model.predict_proba(X_te_sc)
    p_mean = y_proba.mean(axis=0)
    score = calc_score(p_mean, true_label)
    scores.append(score)
    print(f"{dr:<16} | {inv_label_map[true_label]:<5} | {p_mean[0]:.4f}   | {p_mean[1]:.4f}   | {p_mean[2]:.4f}   | {score:.4f}")

print("\n--- Step 2: Conformal Quantiles ---")
scores_sorted = sorted(scores)
print("Sorted calibration scores:")
print(", ".join([f"{s:.4f}" for s in scores_sorted]))

alpha = 0.154
n = 12
idx = math.ceil((n + 1) * (1 - alpha))
if idx > n:
    q_hat = scores_sorted[-1]
else:
    q_hat = scores_sorted[idx - 1]

print(f"Alpha {alpha:.3f} -> q_hat = {q_hat:.4f} (from index {min(idx, n)})")


print("\n--- Step 3: Apply to Test Sets ---")
bundle_final = joblib.load("./artifacts/deployable_chantest_3class.joblib")
model_final = bundle_final['model']
scaler_final = bundle_final['scaler']

datasets = ['chantest', 'nanion', 'manual']

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

for ds in datasets:
    test_file = f"data/{ds}_testing.csv" if ds != "manual" else f"data/{ds}_testing(in).csv"
    df_test = pd.read_csv(test_file).dropna()
    
    X_test = df_test[biomarkers].values
    y_str_test = df_test['label'].values
    y_test = np.array([label_map[y_s.lower()] if isinstance(y_s, str) else y_s for y_s in y_str_test])
    drugs_test = df_test['drug_name'].values
    unique_drugs = sorted(list(set(drugs_test)))
    
    X_test_sc = scaler_final.transform(X_test)
    y_proba = model_final.predict_proba(X_test_sc)
    
    hard_correct = 0; hard_over = 0; hard_under = 0
    conf_correct = 0; conf_over = 0; conf_under = 0; conf_abstain = 0
    total_conf_size = 0
    covered = 0
    
    for dr in unique_drugs:
        idx = (drugs_test == dr)
        true_c = y_test[idx][0]
        
        p_mean = y_proba[idx].mean(axis=0)
        hard_pred = p_mean.argmax()
        
        ho = get_outcome(hard_pred, true_c)
        if ho == 'correct': hard_correct += 1
        elif ho == 'over-called': hard_over += 1
        elif ho == 'under-triaged': hard_under += 1
        
        pred_set_sev = []
        for cand_orig in [0, 1, 2]:
            s_cand = calc_score(p_mean, cand_orig)
            if s_cand <= q_hat:
                pred_set_sev.append(severity(cand_orig))
                
        if len(pred_set_sev) > 0:
            min_sev = min(pred_set_sev)
            max_sev = max(pred_set_sev)
            contiguous_set_sev = list(range(min_sev, max_sev + 1))
        else:
            contiguous_set_sev = []
            
        sev_to_orig = {2: 0, 1: 1, 0: 2}
        pred_set = [sev_to_orig[s] for s in contiguous_set_sev]
        
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
        
        pred_set_sorted = sorted(pred_set, key=severity, reverse=True)
        set_str = "{" + ",".join([inv_label_map[c] for c in pred_set_sorted]) + "}"
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
        'System': 'ordinal conformal @ alpha=0.154',
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
