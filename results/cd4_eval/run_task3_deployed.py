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
    calc_ordinal_aucs_drug, calc_ordinal_aucs_row, run_drug_bootstrap, run_cipa_bootstrap,
    INV_LABEL_MAP
)

def run_task3():
    print("==================================================================")
    print("=== TASK 3: EVALUATE DEPLOYED MODEL ON ALL THREE PLATFORMS ======")
    print("==================================================================")
    
    bundle_path = os.path.join(TRAINAI_DIR, 'hf_space', 'deployable_chantest_3class.joblib')
    bundle = joblib.load(bundle_path)
    model = bundle['model']
    print(f"Loaded deployed model from {bundle_path}")
    print(f"Meta-learner: {model.final_estimator_}")
    
    platforms = ['chantest', 'nanion', 'manual']
    all_platform_results = {}
    
    for platform in platforms:
        print(f"\n--- Platform: {platform.upper()} ---")
        X_test, y_test, groups_test, r_test = load_rev(platform, 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
        print(f"Loaded {len(X_test)} rows across {groups_test.nunique()} test drugs.")
        
        # Row-level predict proba
        row_probs = model.predict_proba(X_test.values)
        y_test_pred_row = np.argmax(row_probs, axis=1)
        
        # Row-level metrics
        row_metrics = evaluate_predictions(y_test.values, y_test_pred_row)
        auc1_row, auc2_row = calc_ordinal_aucs_row(y_test.values, row_probs)
        
        # Drug-level summary
        df_row_probs = pd.DataFrame(row_probs, columns=['prob_high', 'prob_int', 'prob_low'])
        df_row_probs['drug_name'] = groups_test.values
        df_row_probs['true_label'] = y_test.values
        
        drug_summary = get_drug_level_summary(df_row_probs)
        drug_y_true = drug_summary['true_label'].values
        drug_y_pred = drug_summary['pred_label'].values
        
        drug_metrics = evaluate_predictions(drug_y_true, drug_y_pred)
        auc1_drug, auc2_drug = calc_ordinal_aucs_drug(drug_summary)
        
        correct_drugs = int(np.sum(drug_y_true == drug_y_pred))
        total_drugs = len(drug_y_true)
        
        misclass_df = classify_misclassifications(drug_summary)
        
        # Drug bootstrap
        print("Running 10,000 drug-level bootstrap iterations...")
        boot_drug = run_drug_bootstrap(drug_summary, n_iter=10000, seed=42)
        
        # CiPA replicate bootstrap
        print("Running 10,000 CiPA replicate bootstrap iterations...")
        cipa_boot = run_cipa_bootstrap(groups_test, y_test, row_probs, n_bootstrap=10000, seed=42)
        
        res = {
            'platform': platform,
            'correct_drugs': f"{correct_drugs}/{total_drugs}",
            'drug_macro_f1': drug_metrics['macro_f1'],
            'drug_accuracy': drug_metrics['accuracy'],
            'drug_precision': drug_metrics['precision'].tolist(),
            'drug_recall': drug_metrics['recall'].tolist(),
            'drug_f1': drug_metrics['f1'].tolist(),
            'drug_cm': drug_metrics['confusion_matrix'].tolist(),
            'misclassifications': misclass_df.to_dict(orient='records'),
            'row_macro_f1': row_metrics['macro_f1'],
            'row_accuracy': row_metrics['accuracy'],
            'row_cm': row_metrics['confusion_matrix'].tolist(),
            'ordinal_auc_drug': {'risk_gt_low': auc1_drug, 'risk_gt_int': auc2_drug},
            'ordinal_auc_row': {'risk_gt_low': auc1_row, 'risk_gt_int': auc2_row},
            'drug_bootstrap_macro_f1': {
                'median': boot_drug['median'],
                'ci_low': boot_drug['ci_low'],
                'ci_high': boot_drug['ci_high'],
                'ci_str': f"{boot_drug['median']:.6f} [{boot_drug['ci_low']:.6f}, {boot_drug['ci_high']:.6f}]"
            },
            'cipa_bootstrap_ovr_auc': cipa_boot
        }
        
        all_platform_results[platform] = res
        
        print(f"Correct: {correct_drugs}/{total_drugs} ({drug_metrics['accuracy']*100:.1f}%)")
        print(f"Drug Macro-F1: {drug_metrics['macro_f1']:.6f}")
        print(f"Drug Bootstrap Macro-F1 [95% CI]: {res['drug_bootstrap_macro_f1']['ci_str']}")
        print(f"Confusion Matrix (Drug):\n{drug_metrics['confusion_matrix']}")
        print(f"Misclassified Drugs ({len(misclass_df)}):")
        for _, m in misclass_df.iterrows():
            print(f"  {m['drug_name']:<16}: {m['true_tier']:<12} -> {m['pred_tier']:<12} ({m['category']})")
        print(f"Row Macro-F1: {row_metrics['macro_f1']:.6f} | Row Acc: {row_metrics['accuracy']:.6f}")
        print(f"Ordinal AUC (Drug): risk>Low={auc1_drug:.4f}, risk>Int={auc2_drug:.4f}")
        print(f"Ordinal AUC (Row):  risk>Low={auc1_row:.4f}, risk>Int={auc2_row:.4f}")
        print(f"CiPA Replicate Bootstrap OvR AUC: High={cipa_boot['High']['str']}, Int={cipa_boot['Int']['str']}, Low={cipa_boot['Low']['str']}")

    out_json = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval', 'task3_deployed_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_platform_results, f, indent=2)
    print(f"\nSaved Task 3 results to {out_json}")

if __name__ == '__main__':
    run_task3()
