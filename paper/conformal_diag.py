import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
import math
import sys

print("===== DRUG-LEVEL SPLIT-CONFORMAL PREDICTION =====")

print("\n--- Step 1: LODO Calibration Scores ---")
biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
df_train = pd.read_csv("data/chantest_training.csv").dropna()
X_all = df_train[biomarkers].values
label_map = {'high': 0, 'intermediate': 1, 'low': 2}
y_str = df_train['label'].values
y_all = np.array([label_map[y.lower()] if isinstance(y, str) else y for y in y_str])
drugs_all = df_train['drug_name'].values

train_drugs = sorted(list(set(drugs_all)))
assert len(train_drugs) == 12

scores = []
inv_label_map = {0: 'High', 1: 'Int', 2: 'Low'}

class FixedGroupsCV:
    def __init__(self, n_splits, groups):
        self.cv = StratifiedGroupKFold(n_splits=n_splits)
        self.groups = groups
    def split(self, X, y, groups=None):
        return self.cv.split(X, y, self.groups)
    def get_n_splits(self, X, y, groups=None):
        return self.cv.n_splits

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
    score = 1.0 - p_mean[true_label]
    scores.append(score)
    print(f"{dr:<16} | {inv_label_map[true_label]:<5} | {p_mean[0]:.4f}   | {p_mean[1]:.4f}   | {p_mean[2]:.4f}   | {score:.4f}")

print("\n--- Step 2: Conformal Quantiles ---")
scores_sorted = sorted(scores)
print("Sorted calibration scores:")
for i, s in enumerate(scores_sorted):
    print(f"  {i+1}: {s:.4f}")

alphas = [0.077, 0.154, 0.231]
n = 12
q_hat = {}
print(f"\nState plainly: An alpha < 1/13 (approx 0.0769) is not achievable with 12 calibration drugs, because we cannot set a threshold higher than the maximum observed nonconformity score in a distribution-free way.")
for a in alphas:
    idx = math.ceil((n + 1) * (1 - a))
    if idx > n:
        q_val = scores_sorted[-1]
    else:
        q_val = scores_sorted[idx - 1]
    q_hat[a] = q_val
    print(f"Alpha {a:.3f} -> q_hat = {q_val:.4f} (from index {min(idx, n)})")


print("\n--- Step 3: Apply to Test Sets ---")
bundle_final = joblib.load("./artifacts/deployable_chantest_3class.joblib")
model_final = bundle_final['model']
scaler_final = bundle_final['scaler']

datasets = ['chantest', 'nanion', 'manual']
for ds in datasets:
    print(f"\n==========================================")
    print(f"DATASET: {ds.upper()}")
    print(f"==========================================")
    
    test_file = f"data/{ds}_testing.csv" if ds != "manual" else f"data/{ds}_testing(in).csv"
    df_test = pd.read_csv(test_file).dropna()
    
    X_test = df_test[biomarkers].values
    y_test_str = df_test['label'].values
    y_test = np.array([label_map[y.lower()] if isinstance(y, str) else y for y in y_test_str])
    drugs_test = df_test['drug_name'].values
    unique_drugs = sorted(list(set(drugs_test)))
    
    X_test_sc = scaler_final.transform(X_test)
    y_proba = model_final.predict_proba(X_test_sc)
    
    for a in alphas:
        q = q_hat[a]
        print(f"\n--- Alpha {a:.3f} (q_hat = {q:.4f}) ---")
        
        covered_count = 0
        total_size = 0
        sizes = {0: 0, 1: 0, 2: 0, 3: 0}
        critical_misses = 0
        
        print(f"{'Drug Name':<16} | {'True':<5} | {'Prediction Set':<20} | {'Size':<4}")
        print("-" * 55)
        
        for dr in unique_drugs:
            idx = (drugs_test == dr)
            true_label = y_test[idx][0]
            
            p_mean = y_proba[idx].mean(axis=0)
            
            pred_set = []
            for c in [0, 1, 2]:
                if 1.0 - p_mean[c] <= q:
                    pred_set.append(c)
            
            if true_label in pred_set:
                covered_count += 1
            
            set_size = len(pred_set)
            total_size += set_size
            sizes[set_size] = sizes.get(set_size, 0) + 1
            
            if 2 in pred_set and true_label in [0, 1]:
                critical_misses += 1
                
            pred_set_str = "{" + ",".join([inv_label_map[c] for c in pred_set]) + "}"
            print(f"{dr:<16} | {inv_label_map[true_label]:<5} | {pred_set_str:<20} | {set_size:<4}")
            
        print(f"\nSummary for {ds.upper()} at alpha={a}:")
        print(f"Empirical Coverage: {covered_count/16:.4f} ({covered_count}/16 drugs)")
        print(f"Average Set Size: {total_size/16:.4f}")
        print(f"Set Sizes - Empty: {sizes.get(0,0)}, Singleton: {sizes.get(1,0)}, Doubleton: {sizes.get(2,0)}, Tripleton: {sizes.get(3,0)}")
        print(f"Residual Under-triage Risk (Low in set but true High/Int): {critical_misses} drugs")
