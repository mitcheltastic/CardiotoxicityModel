import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import deploy_utils
import sys

print("===== STRICT DETERMINISTIC CROSS-INSTRUMENT DIAGNOSTIC =====")

expected_test_drugs = ['astemizole', 'azimilide', 'clarithromycin', 'clozapine', 'disopyramide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', 'metoprolol', 'nifedipine', 'nitrendipine', 'pimozide', 'risperidone', 'tamoxifen', 'vandetanib']
train_drugs = ['bepridil', 'chlorpromazine', 'cisapride', 'diltiazem', 'dofetilide', 'mexiletine', 'ondansetron', 'quinidine', 'ranolazine', 'sotalol', 'terfenadine', 'verapamil']

biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
model_path = "./artifacts/deployable_chantest_3class.joblib"
bundle = joblib.load(model_path)
model = bundle['model']
scaler = bundle['scaler']

label_map = {'high': 0, 'intermediate': 1, 'low': 2}
inv_label_map = {0: 'High', 1: 'Int', 2: 'Low'}

print("\n[GUARD] Scaler Object Verification:")
print(f"Scaler type: {type(scaler)}")
print(f"Scaler mean_: {scaler.mean_}")
print(f"Scaler scale_: {scaler.scale_}")

datasets = ['chantest', 'nanion', 'manual']

for ds in datasets:
    print(f"\n=======================================================")
    print(f"DATASET: {ds.upper()}")
    print(f"=======================================================")
    
    test_file = f"data/{ds}_testing.csv" if ds != "manual" else f"data/{ds}_testing(in).csv"
    df_test = pd.read_csv(test_file).dropna()
    
    X_test = df_test[biomarkers].values
    y_test_str = df_test['label'].values
    y_test = np.array([label_map[y.lower()] if isinstance(y, str) else y for y in y_test_str])
    drugs_test = df_test['drug_name'].values
    
    unique_drugs = []
    for d in drugs_test:
        if d not in unique_drugs:
            unique_drugs.append(d)
    
    print("\n[GUARD] Test Drug List Verification:")
    print(f"Unique test drugs in row order:\n{unique_drugs}")
    
    assert set(unique_drugs) == set(expected_test_drugs), "Assertion Failed: Test drugs do not match the expected 16 test drugs!"
    assert len(set(unique_drugs).intersection(set(train_drugs))) == 0, "Assertion Failed: Training drugs leaked into test set!"
    print("Assertion Passed: Exactly 16 test drugs, no training drugs leaked.")
    
    print(f"\n[GUARD] Applying EXACT SAME scaler object to {ds.upper()}...")
    X_test_scaled = scaler.transform(X_test)
    
    qnet_idx = biomarkers.index('qNet')
    qnet_mean = np.mean(X_test_scaled[:, qnet_idx])
    qnet_std = np.std(X_test_scaled[:, qnet_idx])
    
    print(f"Scaled {ds.upper()} qNet - Mean: {qnet_mean:.4f}, Std: {qnet_std:.4f}")
    if abs(qnet_mean) > 1.0:
        print(f"-> REAL SHIFT DETECTED for {ds.upper()}!")
    else:
        print(f"-> Centered near 0. No extreme scale shift.")
        
    y_proba = model.predict_proba(X_test_scaled)
    
    y_true_drug = []
    y_pred_drug = []
    
    print("\nPer-Drug Ground Truth & Prediction Table:")
    print(f"{'Drug Name':<16} | {'True':<4} | {'Pred':<4} | {'P(High)':<8} | {'P(Int)':<8} | {'P(Low)':<8}")
    print("-" * 65)
    
    for dr in unique_drugs:
        idx = (drugs_test == dr)
        
        true_labels = y_test[idx]
        assert len(set(true_labels)) == 1, f"Assertion Failed: Drug {dr} has multiple true labels!"
        dr_true = true_labels[0]
        
        dr_probs = y_proba[idx]
        mean_prob = dr_probs.mean(axis=0)
        dr_pred = mean_prob.argmax()
        
        y_true_drug.append(dr_true)
        y_pred_drug.append(dr_pred)
        
        print(f"{dr:<16} | {inv_label_map[dr_true]:<4} | {inv_label_map[dr_pred]:<4} | {mean_prob[0]:.4f}   | {mean_prob[1]:.4f}   | {mean_prob[2]:.4f}")
        
    print("\nDrug-Level Confusion Matrix:")
    print(confusion_matrix(y_true_drug, y_pred_drug, labels=[0, 1, 2]))
    
    macro_f1 = f1_score(y_true_drug, y_pred_drug, average='macro', labels=[0, 1, 2])
    print(f"\nDrug-Level Macro-F1: {macro_f1:.4f}")
    
    print("\nClassification Report (Drug Level):")
    print(classification_report(y_true_drug, y_pred_drug, labels=[0, 1, 2], target_names=['High(0)', 'Int(1)', 'Low(2)'], zero_division=0))
