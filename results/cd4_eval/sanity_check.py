import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS, LABEL_MAP

def run_sanity_check():
    print("=== SANITY CHECK: Reproduce Research Model ChanTest Drug-Level Macro-F1 ===")
    model_path = os.path.join(trainai_dir, 'models', 'chantest_stacking_model_rev.joblib')
    model = joblib.load(model_path)
    
    X_test, y_test, groups_test, r_test = load_rev('chantest', 'testing', data_dir=os.path.join(trainai_dir, 'data'))
    print(f"Loaded X_test: shape={X_test.shape}, unique drugs={groups_test.nunique()}")
    
    # Predict probabilities
    y_test_proba = model.predict_proba(X_test)
    
    # Drug-level evaluation
    df_eval = pd.DataFrame(y_test_proba, columns=['prob_high', 'prob_int', 'prob_low'])
    df_eval['drug_name'] = groups_test.values
    df_eval['true_label'] = y_test.values
    
    drug_summary = df_eval.groupby('drug_name').agg({
        'true_label': 'first',
        'prob_high': 'mean',
        'prob_int': 'mean',
        'prob_low': 'mean'
    }).reset_index()
    
    drug_summary['pred_label'] = np.argmax(
        drug_summary[['prob_high', 'prob_int', 'prob_low']].values, axis=1
    )
    
    drug_y_true = drug_summary['true_label'].values
    drug_y_pred = drug_summary['pred_label'].values
    
    drug_macro_f1 = f1_score(drug_y_true, drug_y_pred, average='macro')
    drug_acc = accuracy_score(drug_y_true, drug_y_pred)
    drug_cm = confusion_matrix(drug_y_true, drug_y_pred, labels=[0, 1, 2])
    
    print(f"Drug-Level Macro-F1: {drug_macro_f1:.6f} ({drug_macro_f1})")
    print(f"Drug-Level Accuracy: {drug_acc:.6f} ({int(drug_acc * len(drug_y_true))}/{len(drug_y_true)})")
    print(f"Confusion Matrix:\n{drug_cm}")
    
    diff = abs(drug_macro_f1 - 0.6944444444444443)
    if diff < 1e-5:
        print("[SUCCESS] Successfully reproduced research model drug-level macro-F1: 0.694444!")
    else:
        print(f"[FAILURE] Macro-F1 mismatch: expected 0.694444, got {drug_macro_f1:.6f}")

if __name__ == '__main__':
    run_sanity_check()
