import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import deploy_utils

print("\n===== Cross-Instrument Generalization Diagnostic =====")

biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
model_path = "./artifacts/deployable_chantest_3class.joblib"
bundle = joblib.load(model_path)
model = bundle['model']
scaler = bundle['scaler']

label_map = {'high': 0, 'intermediate': 1, 'low': 2}
inverse_label_map = {0: 'High', 1: 'Intermediate', 2: 'Low'}

datasets = ['chantest', 'nanion', 'manual']

results = []

for ds in datasets:
    print(f"\n--- DATASET: {ds.upper()} ---")
    
    test_file = f"data/{ds}_testing.csv" if ds != "manual" else f"data/{ds}_testing(in).csv"
    df_test = pd.read_csv(test_file).dropna()
    
    X_test = df_test[biomarkers].values
    y_test_str = df_test['label'].values
    y_test = np.array([label_map[y.lower()] if isinstance(y, str) else y for y in y_test_str])
    drugs_test = df_test['drug_name'].values
    
    X_test_scaled = scaler.transform(X_test)
    
    # (a) Sample-level 3x3 confusion matrix
    y_pred = model.predict(X_test_scaled)
    cm_sample = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print("Sample-level 3x3 Confusion Matrix (High=0, Int=1, Low=2):")
    print(cm_sample)
    
    # Drug-level aggregation
    y_proba = model.predict_proba(X_test_scaled)
    unique_drugs = sorted(list(set(drugs_test)))
    
    y_test_drug = []
    y_pred_drug = []
    
    drug_predictions = {}
    
    for dr in unique_drugs:
        idx = (drugs_test == dr)
        true_label = y_test[idx][0]
        y_test_drug.append(true_label)
        
        # argmax of mean probability
        mean_prob = y_proba[idx].mean(axis=0)
        avg_pred = mean_prob.argmax()
        y_pred_drug.append(avg_pred)
        
        drug_predictions[dr] = {'true': true_label, 'pred': avg_pred}
    
    # (b) Drug-level 3x3 confusion matrix
    cm_drug = confusion_matrix(y_test_drug, y_pred_drug, labels=[0, 1, 2])
    print("\nDrug-level 3x3 Confusion Matrix (Argmax of Mean Probabilities):")
    print(cm_drug)
    
    # (c) Macro-F1 and per-class recall
    macro_f1 = f1_score(y_test_drug, y_pred_drug, average='macro', labels=[0, 1, 2])
    recalls = recall_score(y_test_drug, y_pred_drug, average=None, labels=[0, 1, 2], zero_division=0)
    
    print(f"\nDrug-level Macro-F1: {macro_f1:.4f}")
    print(f"Per-class recall (High, Int, Low): {recalls}")
    
    # (d) Explicit splits and critical misses
    true_high_drugs = [dr for dr, v in drug_predictions.items() if v['true'] == 0]
    true_int_drugs = [dr for dr, v in drug_predictions.items() if v['true'] == 1]
    
    high_preds = [drug_predictions[dr]['pred'] for dr in true_high_drugs]
    int_preds = [drug_predictions[dr]['pred'] for dr in true_int_drugs]
    
    print("\nHigh Drugs Split (True: 4):")
    print(f"Pred High: {high_preds.count(0)}, Pred Int: {high_preds.count(1)}, Pred Low: {high_preds.count(2)}")
    
    print("Intermediate Drugs Split (True: 7):")
    print(f"Pred High: {int_preds.count(0)}, Pred Int: {int_preds.count(1)}, Pred Low: {int_preds.count(2)}")
    
    any_high_low = any(p == 2 for p in high_preds)
    any_int_low = any(p == 2 for p in int_preds)
    print(f"Any High drug predicted Low? {any_high_low}")
    print(f"Any Int drug predicted Low? {any_int_low}")
    critical_miss_count = high_preds.count(2) + int_preds.count(2)
    
    # Bonus: specific misclassified drugs
    print("\nMisclassified Drugs:")
    misclassified = []
    for dr, v in drug_predictions.items():
        if v['true'] != v['pred']:
            print(f"- {dr}: True = {inverse_label_map[v['true']]}, Predicted = {inverse_label_map[v['pred']]}")
            misclassified.append(f"{dr} ({inverse_label_map[v['true']]}->{inverse_label_map[v['pred']]})")
    
    if not misclassified:
        print("None!")
    
    results.append({
        'Dataset': ds.upper(),
        'Drug Macro-F1': f"{macro_f1:.4f}",
        'Int Recall': f"{recalls[1]:.4f}",
        'Crit Miss (Pred Low)': critical_miss_count,
        'Misclassified Drugs': ", ".join(misclassified) if misclassified else "None"
    })

print("\n\n===== COMPACT SUMMARY TABLE =====")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
