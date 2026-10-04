import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import load_rev
from results.cd4_eval.eval_engine import (
    evaluate_predictions, get_drug_level_summary, classify_misclassifications,
    calc_ordinal_aucs_drug, calc_ordinal_aucs_row, run_drug_bootstrap, run_cipa_bootstrap
)

def compute_voting_probs_and_preds(rf_model, xgb_model, X_vals):
    # Predict probabilities from RF and XGB
    p_rf = rf_model.predict_proba(X_vals)
    p_xgb = xgb_model.predict_proba(X_vals)
    
    # Soft voting: mean of predict_proba
    p_soft = (p_rf + p_xgb) / 2.0
    pred_soft = np.argmax(p_soft, axis=1)
    
    # Hard voting: majority vote with ties broken by summed probability
    # For each sample:
    pred_rf = np.argmax(p_rf, axis=1)
    pred_xgb = np.argmax(p_xgb, axis=1)
    
    # If pred_rf == pred_xgb, majority is pred_rf
    # If pred_rf != pred_xgb (1-1 tie), tie broken by argmax(p_rf + p_xgb) = argmax(p_soft)!
    # Thus, with two learners, hard voting with summed probability tie-breaking
    # is mathematically IDENTICAL to soft voting whenever a tie occurs!
    is_tie = (pred_rf != pred_xgb)
    pred_hard = np.where(is_tie, pred_soft, pred_rf)
    
    # Note: verify if pred_hard is identical to pred_soft everywhere
    # If pred_rf == pred_xgb = c, then p_rf[c] is max in p_rf and p_xgb[c] is max in p_xgb,
    # so (p_rf[c] + p_xgb[c])/2 is also strictly max, meaning argmax(p_soft) == c!
    # Therefore, pred_hard is IDENTICAL to pred_soft in all cases!
    return p_soft, pred_soft, pred_hard, p_rf, p_xgb

def run_task4():
    print("==================================================================")
    print("=== TASK 4: VOTING COMPARISON (DEPLOYED VS RESEARCH BASE LEARNERS) ===")
    print("==================================================================")
    
    # 1. Load models
    deployed_bundle = joblib.load(os.path.join(TRAINAI_DIR, 'hf_space', 'deployable_chantest_3class.joblib'))
    deployed_stack = deployed_bundle['model']
    dep_rf = deployed_stack.estimators_[0]
    dep_xgb = deployed_stack.estimators_[1]
    
    research_stack = joblib.load(os.path.join(TRAINAI_DIR, 'models', 'chantest_stacking_model_rev.joblib'))
    res_rf = research_stack.named_estimators_['rf']
    res_xgb = research_stack.named_estimators_['xgb']
    
    model_sources = {
        'deployed_base': (dep_rf, dep_xgb, "Base learners from hf_space/deployable_chantest_3class.joblib"),
        'research_base': (res_rf, res_xgb, "Base learners from models/chantest_stacking_model_rev.joblib")
    }
    
    platforms = ['chantest', 'nanion', 'manual']
    voting_results = {}
    
    for src_name, (rf_m, xgb_m, desc) in model_sources.items():
        print(f"\n==================================================================")
        print(f"=== SOURCE: {src_name} ({desc}) ===")
        print(f"==================================================================")
        voting_results[src_name] = {}
        
        for platform in platforms:
            print(f"\n--- Platform: {platform.upper()} ---")
            X_test, y_test, groups_test, _ = load_rev(platform, 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
            
            p_soft, pred_soft, pred_hard, p_rf, p_xgb = compute_voting_probs_and_preds(rf_m, xgb_m, X_test.values)
            
            # Check hard vs soft equivalence
            identical_preds = np.array_equal(pred_soft, pred_hard)
            print(f"Are row-level hard voting and soft voting predictions identical? {identical_preds}")
            
            # We evaluate both soft and hard voting
            # Note: For drug-level aggregation, soft voting averages p_soft per drug then argmax.
            # For hard voting, does it average p_soft or take majority of drug-level predictions?
            # Prompt states: "(a) soft voting = mean of their predict_proba, argmax; (b) hard voting = majority of their predicted classes with ties broken by summed probability (note in the report if hard voting with two learners reduces to soft voting on ties)."
            
            # Row-level metrics
            row_metrics_soft = evaluate_predictions(y_test.values, pred_soft)
            row_metrics_hard = evaluate_predictions(y_test.values, pred_hard)
            auc1_row, auc2_row = calc_ordinal_aucs_row(y_test.values, p_soft)
            
            # Drug-level aggregation for soft voting:
            df_row_probs = pd.DataFrame(p_soft, columns=['prob_high', 'prob_int', 'prob_low'])
            df_row_probs['drug_name'] = groups_test.values
            df_row_probs['true_label'] = y_test.values
            
            drug_summary_soft = get_drug_level_summary(df_row_probs)
            drug_y_true = drug_summary_soft['true_label'].values
            drug_y_pred_soft = drug_summary_soft['pred_label'].values
            
            drug_metrics_soft = evaluate_predictions(drug_y_true, drug_y_pred_soft)
            auc1_drug, auc2_drug = calc_ordinal_aucs_drug(drug_summary_soft)
            
            correct_drugs_soft = int(np.sum(drug_y_true == drug_y_pred_soft))
            total_drugs = len(drug_y_true)
            misclass_soft = classify_misclassifications(drug_summary_soft)
            
            # Drug bootstrap
            boot_drug_soft = run_drug_bootstrap(drug_summary_soft, n_iter=10000, seed=42)
            
            # CiPA replicate bootstrap
            cipa_boot_soft = run_cipa_bootstrap(groups_test, y_test, p_soft, n_bootstrap=10000, seed=42)
            
            # For hard voting: if applied at drug level directly to base learners' drug predictions:
            # Let's also compute drug predictions for RF alone and XGB alone
            df_rf = pd.DataFrame(p_rf, columns=['prob_high', 'prob_int', 'prob_low'])
            df_rf['drug_name'] = groups_test.values
            df_rf['true_label'] = y_test.values
            ds_rf = get_drug_level_summary(df_rf)
            
            df_xgb = pd.DataFrame(p_xgb, columns=['prob_high', 'prob_int', 'prob_low'])
            df_xgb['drug_name'] = groups_test.values
            df_xgb['true_label'] = y_test.values
            ds_xgb = get_drug_level_summary(df_xgb)
            
            # If RF and XGB agree at drug level, that's majority. If they disagree, tie broken by summed drug-level probability:
            drug_rf_pred = ds_rf['pred_label'].values
            drug_xgb_pred = ds_xgb['pred_label'].values
            drug_hard_tie = (drug_rf_pred != drug_xgb_pred)
            drug_summed_prob = (ds_rf[['prob_high', 'prob_int', 'prob_low']].values + ds_xgb[['prob_high', 'prob_int', 'prob_low']].values) / 2.0
            drug_hard_pred = np.where(drug_hard_tie, np.argmax(drug_summed_prob, axis=1), drug_rf_pred)
            
            drug_hard_identical_to_soft = np.array_equal(drug_hard_pred, drug_y_pred_soft)
            print(f"Are drug-level hard voting and soft voting predictions identical? {drug_hard_identical_to_soft}")
            
            drug_metrics_hard = evaluate_predictions(drug_y_true, drug_hard_pred)
            misclass_hard = classify_misclassifications(pd.DataFrame({
                'drug_name': drug_summary_soft['drug_name'],
                'true_label': drug_y_true,
                'pred_label': drug_hard_pred,
                'prob_high': drug_summary_soft['prob_high'],
                'prob_int': drug_summary_soft['prob_int'],
                'prob_low': drug_summary_soft['prob_low']
            }))
            
            res_platform = {
                'platform': platform,
                'soft_voting': {
                    'correct_drugs': f"{correct_drugs_soft}/{total_drugs}",
                    'drug_macro_f1': drug_metrics_soft['macro_f1'],
                    'drug_accuracy': drug_metrics_soft['accuracy'],
                    'drug_precision': drug_metrics_soft['precision'].tolist(),
                    'drug_recall': drug_metrics_soft['recall'].tolist(),
                    'drug_f1': drug_metrics_soft['f1'].tolist(),
                    'drug_cm': drug_metrics_soft['confusion_matrix'].tolist(),
                    'misclassifications': misclass_soft.to_dict(orient='records'),
                    'row_macro_f1': row_metrics_soft['macro_f1'],
                    'row_accuracy': row_metrics_soft['accuracy'],
                    'row_cm': row_metrics_soft['confusion_matrix'].tolist(),
                    'ordinal_auc_drug': {'risk_gt_low': auc1_drug, 'risk_gt_int': auc2_drug},
                    'ordinal_auc_row': {'risk_gt_low': auc1_row, 'risk_gt_int': auc2_row},
                    'drug_bootstrap_macro_f1': {
                        'median': boot_drug_soft['median'],
                        'ci_low': boot_drug_soft['ci_low'],
                        'ci_high': boot_drug_soft['ci_high'],
                        'ci_str': f"{boot_drug_soft['median']:.6f} [{boot_drug_soft['ci_low']:.6f}, {boot_drug_soft['ci_high']:.6f}]"
                    },
                    'cipa_bootstrap_ovr_auc': cipa_boot_soft
                },
                'hard_voting': {
                    'correct_drugs': f"{int(np.sum(drug_y_true == drug_hard_pred))}/{total_drugs}",
                    'drug_macro_f1': drug_metrics_hard['macro_f1'],
                    'drug_accuracy': drug_metrics_hard['accuracy'],
                    'drug_precision': drug_metrics_hard['precision'].tolist(),
                    'drug_recall': drug_metrics_hard['recall'].tolist(),
                    'drug_f1': drug_metrics_hard['f1'].tolist(),
                    'drug_cm': drug_metrics_hard['confusion_matrix'].tolist(),
                    'misclassifications': misclass_hard.to_dict(orient='records'),
                    'row_macro_f1': row_metrics_hard['macro_f1'],
                    'row_accuracy': row_metrics_hard['accuracy'],
                    'row_cm': row_metrics_hard['confusion_matrix'].tolist(),
                    'identical_to_soft_row': identical_preds,
                    'identical_to_soft_drug': drug_hard_identical_to_soft
                }
            }
            
            voting_results[src_name][platform] = res_platform
            
            print(f"Soft Voting - Correct: {correct_drugs_soft}/{total_drugs} | Macro-F1: {drug_metrics_soft['macro_f1']:.6f}")
            print(f"Soft Voting - Bootstrap CI: {res_platform['soft_voting']['drug_bootstrap_macro_f1']['ci_str']}")
            print(f"Hard Voting - Correct: {res_platform['hard_voting']['correct_drugs']} | Macro-F1: {drug_metrics_hard['macro_f1']:.6f}")
            print(f"Misclassified Drugs (Soft Voting, {len(misclass_soft)}):")
            for _, m in misclass_soft.iterrows():
                print(f"  {m['drug_name']:<16}: {m['true_tier']:<12} -> {m['pred_tier']:<12} ({m['category']})")

    out_json = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval', 'task4_voting_results.json')
    with open(out_json, 'w') as f:
        json.dump(voting_results, f, indent=2)
    print(f"\nSaved Task 4 results to {out_json}")

if __name__ == '__main__':
    run_task4()
