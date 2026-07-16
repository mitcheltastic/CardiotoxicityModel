import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from sklearn.metrics import confusion_matrix
import deploy_utils

print("\n===== Q3: Drug-level confusion for the deployable model =====")
biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
model_path = "./artifacts/deployable_chantest_3class.joblib"
bundle = joblib.load(model_path)
model = bundle['model']
scaler = bundle['scaler']

df_test = pd.read_csv("data/chantest_testing.csv").dropna()
X_test = df_test[biomarkers].values

label_map = {'high': 0, 'intermediate': 1, 'low': 2}
y_test_str = df_test['label'].values
y_test = np.array([label_map[y.lower()] if isinstance(y, str) else y for y in y_test_str])

drugs_test = df_test['drug_name'].values

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
cm_sample = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print("Sample-level 3x3 Confusion Matrix (High=0, Int=1, Low=2):")
print(cm_sample)
print(f"Exact count of true-high samples predicted low: {cm_sample[0, 2]}")

y_proba = model.predict_proba(X_test_scaled)

unique_drugs = sorted(list(set(drugs_test)))
y_test_drug = []
y_pred_maj = []
y_pred_avg = []

for dr in unique_drugs:
    idx = (drugs_test == dr)
    true_label = y_test[idx][0]
    y_test_drug.append(true_label)
    
    preds = y_pred[idx]
    maj_pred = np.bincount(preds).argmax()
    y_pred_maj.append(maj_pred)
    
    mean_prob = y_proba[idx].mean(axis=0)
    avg_pred = mean_prob.argmax()
    y_pred_avg.append(avg_pred)

cm_maj = confusion_matrix(y_test_drug, y_pred_maj, labels=[0, 1, 2])
cm_avg = confusion_matrix(y_test_drug, y_pred_avg, labels=[0, 1, 2])

print("\nDrug-level 3x3 Confusion Matrix (Majority Vote):")
print(cm_maj)

print("\nDrug-level 3x3 Confusion Matrix (Argmax of Mean Probabilities):")
print(cm_avg)

true_int_count = sum(1 for y in y_test_drug if y == 1)
print(f"\nHow many of the 16 test drugs are truly intermediate: {true_int_count}")

true_int_idx = [i for i, y in enumerate(y_test_drug) if y == 1]
maj_splits = [y_pred_maj[i] for i in true_int_idx]
avg_splits = [y_pred_avg[i] for i in true_int_idx]

print(f"How they split (Majority Vote) - High: {maj_splits.count(0)}, Int: {maj_splits.count(1)}, Low: {maj_splits.count(2)}")
print(f"How they split (Mean Probs) - High: {avg_splits.count(0)}, Int: {avg_splits.count(1)}, Low: {avg_splits.count(2)}")

true_high_idx = [i for i, y in enumerate(y_test_drug) if y == 0]
high_pred_low_maj = any(y_pred_maj[i] == 2 for i in true_high_idx)
high_pred_low_avg = any(y_pred_avg[i] == 2 for i in true_high_idx)
print(f"Any truly-high drug predicted low? Majority Vote: {high_pred_low_maj}, Mean Probs: {high_pred_low_avg}")

print("\n===== Q4: CV splitter swappability =====")
with open('plus_ultra.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if ("Phase 3:" in source or "sweep" in source.lower()) and "StratifiedGroupKFold" in source:
            print(f"\n--- Code Cell {i} snippets ---")
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'KFold' in line or 'cv=' in line or 'cv_scheme' in line or 'Group' in line:
                    print(f"L{j}: {line.strip()}")
