import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS, LABEL_MAP

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
SEVERITY = {0: 2, 1: 1, 2: 0} # 0=High (Sev 2), 1=Int (Sev 1), 2=Low (Sev 0)

def get_action_severity(pred_set):
    if len(pred_set) == 0:
        return -1
    return max([SEVERITY[c] for c in pred_set])

def evaluate_conformal_on_platform(drug_summary, q_hat):
    # Inclusion threshold: p >= 1 - q_hat
    thresh = 1.0 - q_hat
    
    records = []
    emp_covered = 0
    tot_size = 0
    n_ambiguous = 0
    n_empty = 0
    n_under_triage = 0
    
    for _, row in drug_summary.iterrows():
        t = int(row['true_label'])
        probs = [row['prob_high'], row['prob_int'], row['prob_low']]
        
        # Build prediction set
        pred_set = [c for c in range(3) if probs[c] >= thresh]
        # Sort by severity descending
        pred_set.sort(key=lambda c: SEVERITY[c], reverse=True)
        
        size = len(pred_set)
        tot_size += size
        
        is_cov = (t in pred_set)
        if is_cov:
            emp_covered += 1
            
        if size > 1:
            n_ambiguous += 1
        elif size == 0:
            n_empty += 1
            
        # Action rule matching app.py / STEP6:
        # If set not empty, action is most severe class in pred_set (first class)
        # If set empty, fallback action is argmax of probabilities
        if size > 0:
            action_class = pred_set[0]
            action_sev = SEVERITY[action_class]
        else:
            action_class = int(np.argmax(probs))
            action_sev = SEVERITY[action_class]
            
        true_sev = SEVERITY[t]
        
        # Outcome:
        if action_sev < true_sev:
            outcome = "under-triaged"
            n_under_triage += 1
        elif action_sev > true_sev:
            outcome = "over-called"
        else:
            outcome = "correct"
            
        records.append({
            'drug_name': row['drug_name'],
            'true_tier': INV_LABEL_MAP[t],
            'prediction_set': [INV_LABEL_MAP[c] for c in pred_set],
            'set_size': size,
            'action_tier': INV_LABEL_MAP[action_class],
            'outcome': outcome,
            'is_covered': is_cov,
            'is_empty': (size == 0),
            'is_ambiguous': (size > 1)
        })
        
    n_drugs = len(drug_summary)
    return {
        'empirical_coverage': emp_covered / n_drugs,
        'covered_count': f"{emp_covered}/{n_drugs}",
        'avg_set_size': tot_size / n_drugs,
        'num_ambiguous': n_ambiguous,
        'num_empty_ood': n_empty,
        'num_under_triage': n_under_triage,
        'drug_details': records
    }

def run_task5():
    print("==================================================================")
    print("=== TASK 5: CONFORMAL RECALIBRATION (DEPLOYED LR STACKING) ======")
    print("==================================================================")
    
    # 1. Load Chantest Training Dataset
    X_train_full, y_train_full, groups_train_full, _ = load_rev('chantest', 'training', data_dir=os.path.join(TRAINAI_DIR, 'data'))
    train_drugs = sorted(groups_train_full.unique())
    print(f"Loaded Chantest training: {len(X_train_full)} rows across {len(train_drugs)} drugs: {train_drugs}")
    
    # 2. LODO Loop
    # Setup matching build_hf_bundle_rev.py
    lodo_scores = []
    
    for d in train_drugs:
        mask_d = (groups_train_full == d)
        X_d = X_train_full[mask_d].values
        y_d = y_train_full[mask_d].values
        
        mask_fit = (groups_train_full != d)
        X_fit = X_train_full[mask_fit].values
        y_fit = y_train_full[mask_fit].values
        groups_fit = groups_train_full[mask_fit].values
        
        # 4-split StratifiedGroupKFold on the 11 fit drugs
        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=False)
        cv_splits = list(sgkf.split(X_fit, y_fit, groups=groups_fit))
        
        rf_base = RandomForestClassifier(
            n_estimators=400,
            criterion='entropy',
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        xgb_base = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=10,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        
        stack = StackingClassifier(
            estimators=[('RF', rf_base), ('XGB', xgb_base)],
            final_estimator=meta_learner,
            cv=cv_splits,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        stack.fit(X_fit, y_fit)
        
        # Predict on hold-out drug d
        y_proba_d = stack.predict_proba(X_d)
        p_mean = y_proba_d.mean(axis=0)
        
        true_class = y_d[0]
        s_d = 1.0 - p_mean[true_class]
        
        lodo_scores.append({
            'drug_name': d,
            'true_class': INV_LABEL_MAP[true_class],
            'true_class_idx': int(true_class),
            'prob_high': float(p_mean[0]),
            'prob_int': float(p_mean[1]),
            'prob_low': float(p_mean[2]),
            's_d': float(s_d)
        })
        print(f"  Hold-out drug: {d:<16} | True: {INV_LABEL_MAP[true_class]:<12} | P(true)={p_mean[true_class]:.6f} | s_d = {s_d:.6f}")

    df_lodo = pd.DataFrame(lodo_scores)
    df_lodo_sorted = df_lodo.sort_values(by='s_d').reset_index(drop=True)
    df_lodo_sorted['rank'] = df_lodo_sorted.index + 1
    
    print("\n--- 12 CALIBRATION SCORES (SORTED ASCENDING) ---")
    print(df_lodo_sorted[['rank', 'drug_name', 'true_class', 's_d', 'prob_high', 'prob_int', 'prob_low']].to_string(index=False))
    
    # Rank 11 selection:
    # n = 12 calibration drugs
    # k = 11
    # q_hat = s_{(11)}
    rank_11_row = df_lodo_sorted[df_lodo_sorted['rank'] == 11].iloc[0]
    q_hat_11 = rank_11_row['s_d']
    alpha_11 = 1.0 - (11.0 / 13.0) # 2/13 = 0.153846
    nom_cov_11 = 11.0 / 13.0       # 84.615%
    thresh_11 = 1.0 - q_hat_11
    
    print(f"\n--- CONFORMAL OPERATING POINT (RANK 11) ---")
    print(f"Rank 11 Drug:     {rank_11_row['drug_name']} ({rank_11_row['true_class']})")
    print(f"q_hat (Rank 11):  {q_hat_11:.6f}")
    print(f"Alpha:            {alpha_11:.6f} (smallest valid alpha for Rank 11)")
    print(f"Nominal Coverage: {nom_cov_11*100:.3f}% ({nom_cov_11:.6f})")
    print(f"Inclusion Thresh: {thresh_11:.6f} (1 - q_hat)")
    
    # 3. Apply to held-out test sets across all three platforms
    # Load deployed model bundle to get predictions on test sets
    bundle = joblib.load(os.path.join(TRAINAI_DIR, 'hf_space', 'deployable_chantest_3class.joblib'))
    deployed_model = bundle['model']
    
    platforms = ['chantest', 'nanion', 'manual']
    conformal_test_results = {}
    
    for platform in platforms:
        X_test, y_test, groups_test, _ = load_rev(platform, 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
        row_probs = deployed_model.predict_proba(X_test.values)
        
        df_row = pd.DataFrame(row_probs, columns=['prob_high', 'prob_int', 'prob_low'])
        df_row['drug_name'] = groups_test.values
        df_row['true_label'] = y_test.values
        
        drug_summary = df_row.groupby('drug_name').agg({
            'true_label': 'first',
            'prob_high': 'mean',
            'prob_int': 'mean',
            'prob_low': 'mean'
        }).reset_index()
        
        platform_conf = evaluate_conformal_on_platform(drug_summary, q_hat_11)
        conformal_test_results[platform] = platform_conf
        
        print(f"\n--- Platform: {platform.upper()} Conformal Performance (q_hat={q_hat_11:.6f}) ---")
        print(f"  Empirical Coverage:     {platform_conf['covered_count']} ({platform_conf['empirical_coverage']*100:.1f}%)")
        print(f"  Average Set Size:       {platform_conf['avg_set_size']:.3f}")
        print(f"  Ambiguous Sets (|S|>1): {platform_conf['num_ambiguous']}")
        print(f"  Empty Sets (OOD, |S|=0):{platform_conf['num_empty_ood']}")
        print(f"  Under-triage Count:     {platform_conf['num_under_triage']}")
        print("  Detailed Drug Prediction Sets:")
        for d_rec in platform_conf['drug_details']:
            print(f"    {d_rec['drug_name']:<16} | True: {d_rec['true_tier']:<12} | Set: {str(d_rec['prediction_set']):<25} | Action: {d_rec['action_tier']:<12} | {d_rec['outcome']}")

    # Also test with hard-coded Q_HAT = 0.6275 from app.py to compare!
    print(f"\n--- COMPARISON WITH HARD-CODED Q_HAT = 0.6275 ---")
    conf_hardcoded = {}
    for platform in platforms:
        X_test, y_test, groups_test, _ = load_rev(platform, 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
        row_probs = deployed_model.predict_proba(X_test.values)
        df_row = pd.DataFrame(row_probs, columns=['prob_high', 'prob_int', 'prob_low'])
        df_row['drug_name'] = groups_test.values
        df_row['true_label'] = y_test.values
        drug_summary = df_row.groupby('drug_name').agg({
            'true_label': 'first',
            'prob_high': 'mean',
            'prob_int': 'mean',
            'prob_low': 'mean'
        }).reset_index()
        conf_hardcoded[platform] = evaluate_conformal_on_platform(drug_summary, 0.6275)
        print(f"  {platform.upper():<10}: EmpCov={conf_hardcoded[platform]['covered_count']}, AvgSize={conf_hardcoded[platform]['avg_set_size']:.3f}, Ambiguous={conf_hardcoded[platform]['num_ambiguous']}, Empty={conf_hardcoded[platform]['num_empty_ood']}, UnderTriage={conf_hardcoded[platform]['num_under_triage']}")

    output_data = {
        'lodo_scores_table': df_lodo_sorted.to_dict(orient='records'),
        'recalibrated_rank11': {
            'q_hat': q_hat_11,
            'alpha': alpha_11,
            'nominal_coverage': nom_cov_11,
            'inclusion_threshold': thresh_11,
            'rank_11_drug': rank_11_row['drug_name']
        },
        'test_evaluation_recalibrated': conformal_test_results,
        'test_evaluation_hardcoded_06275': conf_hardcoded
    }
    
    out_json = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval', 'task5_conformal_results.json')
    with open(out_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved Task 5 results to {out_json}")

if __name__ == '__main__':
    run_task5()
