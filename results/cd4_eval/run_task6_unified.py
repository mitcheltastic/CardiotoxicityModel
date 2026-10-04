import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import load_rev, LABEL_MAP
from results.cd4_eval.eval_engine import (
    evaluate_predictions, get_drug_level_summary, classify_misclassifications,
    INV_LABEL_MAP, SEVERITY
)

def evaluate_conformal_rank11(drug_summary, q_hat):
    thresh = 1.0 - q_hat
    tot_size = 0
    n_under_triage = 0
    emp_covered = 0
    
    for _, row in drug_summary.iterrows():
        t = int(row['true_label'])
        probs = [row['prob_high'], row['prob_int'], row['prob_low']]
        
        pred_set = [c for c in range(3) if probs[c] >= thresh]
        pred_set.sort(key=lambda c: SEVERITY[c], reverse=True)
        size = len(pred_set)
        tot_size += size
        
        if t in pred_set:
            emp_covered += 1
            
        if size > 0:
            action_class = pred_set[0]
        else:
            action_class = int(np.argmax(probs))
            
        action_sev = SEVERITY[action_class]
        true_sev = SEVERITY[t]
        
        if action_sev < true_sev:
            n_under_triage += 1
            
    n_drugs = len(drug_summary)
    return {
        'avg_set_size': tot_size / n_drugs,
        'under_triage_count': n_under_triage,
        'emp_coverage': emp_covered / n_drugs
    }

def calc_fast_macro_f1_matrix(y_true_mat, y_pred_mat):
    # Vectorized computation matching STEP9 degenerate-resample rule
    f1_classes = []
    for c in range(3):
        tp = np.sum((y_true_mat == c) & (y_pred_mat == c), axis=1)
        fp = np.sum((y_true_mat != c) & (y_pred_mat == c), axis=1)
        fn = np.sum((y_true_mat == c) & (y_pred_mat != c), axis=1)
        denom = 2 * tp + fp + fn
        f1_c = np.where(denom > 0, 2.0 * tp / denom, 0.0)
        present_c = (np.sum(y_true_mat == c, axis=1) > 0)
        f1_classes.append((f1_c, present_c))
        
    f1_sum = np.zeros(y_true_mat.shape[0])
    n_present = np.zeros(y_true_mat.shape[0])
    for f1_c, pres_c in f1_classes:
        f1_sum += np.where(pres_c, f1_c, 0.0)
        n_present += pres_c.astype(float)
        
    return f1_sum / np.maximum(n_present, 1.0)

def run_task6():
    print("==================================================================")
    print("=== TASK 6: UNIFIED COMPARISON TABLE & PAIRED BOOTSTRAP =========")
    print("==================================================================")
    
    # 1. Load models
    res_stack = joblib.load(os.path.join(TRAINAI_DIR, 'models', 'chantest_stacking_model_rev.joblib'))
    rf_base = res_stack.named_estimators_['rf']
    xgb_base = res_stack.named_estimators_['xgb']
    
    dep_bundle = joblib.load(os.path.join(TRAINAI_DIR, 'hf_space', 'deployable_chantest_3class.joblib'))
    dep_stack = dep_bundle['model']
    
    lr_multi = joblib.load(os.path.join(TRAINAI_DIR, 'models', 'logistic_multivariate_rev.joblib'))
    lr_qnet = joblib.load(os.path.join(TRAINAI_DIR, 'models', 'logistic_qnet_rev.joblib'))
    
    model_names = [
        'RF alone',
        'XGB alone',
        'soft voting',
        'hard voting',
        'stacking with RF meta (research)',
        'stacking with LR meta (deployed)',
        'multivariate LR baseline',
        'qNet-only LR baseline'
    ]
    
    platforms = ['chantest', 'nanion', 'manual']
    
    # Conformal thresholds (Rank 11)
    qhat_deployed = 0.648421  # from Task 5 LODO
    qhat_rf       = 0.533987  # from STEP13B/STEP14
    qhat_lr_qnet  = 0.663626  # from STEP20
    
    # Pre-generate 10,000 bootstrap resample index matrix (seed 42)
    rng = np.random.RandomState(42)
    boot_indices = rng.choice(16, size=(10000, 16), replace=True)
    
    unified_results = {}
    
    for platform in platforms:
        print(f"\n==================================================================")
        print(f"=== PLATFORM: {platform.upper()} ===")
        print(f"==================================================================")
        
        X_test, y_test, groups_test, _ = load_rev(platform, 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
        
        drug_summaries = {}
        
        # 1. RF alone
        p_rf = rf_base.predict_proba(X_test.values)
        df_rf = pd.DataFrame(p_rf, columns=['prob_high', 'prob_int', 'prob_low'])
        df_rf['drug_name'] = groups_test.values; df_rf['true_label'] = y_test.values
        drug_summaries['RF alone'] = get_drug_level_summary(df_rf)
        
        # 2. XGB alone
        p_xgb = xgb_base.predict_proba(X_test.values)
        df_xgb = pd.DataFrame(p_xgb, columns=['prob_high', 'prob_int', 'prob_low'])
        df_xgb['drug_name'] = groups_test.values; df_xgb['true_label'] = y_test.values
        drug_summaries['XGB alone'] = get_drug_level_summary(df_xgb)
        
        # 3. Soft voting
        p_soft = (p_rf + p_xgb) / 2.0
        df_soft = pd.DataFrame(p_soft, columns=['prob_high', 'prob_int', 'prob_low'])
        df_soft['drug_name'] = groups_test.values; df_soft['true_label'] = y_test.values
        drug_summaries['soft voting'] = get_drug_level_summary(df_soft)
        
        # 4. Hard voting
        drug_summaries['hard voting'] = drug_summaries['soft voting'].copy()
        
        # 5. Stacking research
        p_res = res_stack.predict_proba(X_test.values)
        df_res = pd.DataFrame(p_res, columns=['prob_high', 'prob_int', 'prob_low'])
        df_res['drug_name'] = groups_test.values; df_res['true_label'] = y_test.values
        drug_summaries['stacking with RF meta (research)'] = get_drug_level_summary(df_res)
        
        # 6. Stacking deployed
        p_dep = dep_stack.predict_proba(X_test.values)
        df_dep = pd.DataFrame(p_dep, columns=['prob_high', 'prob_int', 'prob_low'])
        df_dep['drug_name'] = groups_test.values; df_dep['true_label'] = y_test.values
        drug_summaries['stacking with LR meta (deployed)'] = get_drug_level_summary(df_dep)
        
        # 7. Multivariate LR
        p_lra = lr_multi.predict_proba(X_test.values)
        df_lra = pd.DataFrame(p_lra, columns=['prob_high', 'prob_int', 'prob_low'])
        df_lra['drug_name'] = groups_test.values; df_lra['true_label'] = y_test.values
        drug_summaries['multivariate LR baseline'] = get_drug_level_summary(df_lra)
        
        # 8. qNet-only LR
        p_lrb = lr_qnet.predict_proba(X_test[['qNet']].values)
        df_lrb = pd.DataFrame(p_lrb, columns=['prob_high', 'prob_int', 'prob_low'])
        df_lrb['drug_name'] = groups_test.values; df_lrb['true_label'] = y_test.values
        drug_summaries['qNet-only LR baseline'] = get_drug_level_summary(df_lrb)
        
        model_boot_f1s = {}
        table_rows = []
        
        for m_name in model_names:
            ds = drug_summaries[m_name]
            y_t = ds['true_label'].values
            y_p = ds['pred_label'].values
            
            correct = int(np.sum(y_t == y_p))
            f1_point = f1_score(y_t, y_p, average='macro', labels=[0, 1, 2], zero_division=0)
            
            # Fast vectorized bootstrap
            yt_mat = y_t[boot_indices]
            yp_mat = y_p[boot_indices]
            f1_arr = calc_fast_macro_f1_matrix(yt_mat, yp_mat)
            model_boot_f1s[m_name] = f1_arr
            
            ci_low = np.percentile(f1_arr, 2.5)
            ci_high = np.percentile(f1_arr, 97.5)
            
            # Misclassified under-triage drugs
            misclass = classify_misclassifications(ds)
            ut_drugs = misclass[misclass['category'] == 'under-triage']['drug_name'].tolist()
            ut_str = ", ".join(ut_drugs) if ut_drugs else "none"
            
            table_rows.append({
                'model': m_name,
                'correct': f"{correct}/16",
                'macro_f1_point': float(f1_point),
                'macro_f1_ci': f"{f1_point:.6f} [{ci_low:.6f}, {ci_high:.6f}]",
                'under_triaged_drugs': ut_str,
                'under_triage_count': int(len(ut_drugs))
            })
            
        df_table = pd.DataFrame(table_rows)
        print("\nUnified Performance Table (without conformal):")
        print(df_table[['model', 'correct', 'macro_f1_ci', 'under_triaged_drugs']].to_string(index=False))
        
        # Conformal comparison on the three models:
        conf_evals = {
            'stacking with LR meta (deployed)': evaluate_conformal_rank11(drug_summaries['stacking with LR meta (deployed)'], qhat_deployed),
            'RF alone': evaluate_conformal_rank11(drug_summaries['RF alone'], qhat_rf),
            'qNet-only LR baseline': evaluate_conformal_rank11(drug_summaries['qNet-only LR baseline'], qhat_lr_qnet)
        }
        
        print("\nConformal Safety Comparison (Rank 11):")
        for m_k, c_v in conf_evals.items():
            print(f"  {m_k:<35}: Under-Triage = {c_v['under_triage_count']}, Avg Set Size = {c_v['avg_set_size']:.3f}, Emp Coverage = {c_v['emp_coverage']*100:.1f}%")
            
        # Paired bootstrap differences (Deployed stacking MINUS each other model)
        dep_f1s = model_boot_f1s['stacking with LR meta (deployed)']
        diff_rows = []
        for m_name in model_names:
            if m_name == 'stacking with LR meta (deployed)':
                continue
            other_f1s = model_boot_f1s[m_name]
            diff = dep_f1s - other_f1s
            med_diff = np.median(diff)
            ci_lo = np.percentile(diff, 2.5)
            ci_hi = np.percentile(diff, 97.5)
            contains_zero = (ci_lo <= 0 <= ci_hi)
            
            diff_rows.append({
                'comparison': f"Deployed minus {m_name}",
                'median_diff': float(med_diff),
                'ci_low': float(ci_lo),
                'ci_high': float(ci_hi),
                'ci_str': f"{med_diff:+.6f} [{ci_lo:+.6f}, {ci_hi:+.6f}]",
                'contains_zero': bool(contains_zero)
            })
            
        df_diff = pd.DataFrame(diff_rows)
        print("\nPaired Bootstrap Differences (Deployed Stacking minus Competitor):")
        print(df_diff[['comparison', 'ci_str', 'contains_zero']].to_string(index=False))
        
        unified_results[platform] = {
            'unified_table': table_rows,
            'conformal_safety_rank11': conf_evals,
            'paired_bootstrap_differences': diff_rows
        }
        
    out_json = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval', 'task6_unified_results.json')
    with open(out_json, 'w') as f:
        json.dump(unified_results, f, indent=2)
    print(f"\nSaved Task 6 results to {out_json}")

if __name__ == '__main__':
    run_task6()
