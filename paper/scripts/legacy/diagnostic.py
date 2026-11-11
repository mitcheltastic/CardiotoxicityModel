import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from sklearn.metrics import confusion_matrix
import deploy_utils

print("===== Q1: Drug-name overlap across datasets =====")

datasets = ["chantest", "nanion", "manual"]
drug_sets_train = {}
drug_sets_test = {}

for d in datasets:
    train_file = f"data/{d}_training.csv" if d != "manual" else f"data/{d}_training(in).csv"
    test_file = f"data/{d}_testing.csv" if d != "manual" else f"data/{d}_testing(in).csv"
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    train_drugs = set(df_train['drug_name'].unique())
    test_drugs = set(df_test['drug_name'].unique())
    
    drug_sets_train[d] = train_drugs
    drug_sets_test[d] = test_drugs
    
    print(f"{d} train ({len(train_drugs)}):", sorted(list(train_drugs)))
    print(f"{d} test ({len(test_drugs)}):", sorted(list(test_drugs)))

def compute_intersections(d_sets, name="train"):
    d1, d2, d3 = d_sets["chantest"], d_sets["nanion"], d_sets["manual"]
    print(f"\n{name.capitalize()} Pairwise intersections:")
    print(f"chantest & nanion: {len(d1.intersection(d2))}")
    print(f"chantest & manual: {len(d1.intersection(d3))}")
    print(f"nanion & manual: {len(d2.intersection(d3))}")
    
    three_way = d1.intersection(d2).intersection(d3)
    print(f"{name.capitalize()} 3-way intersection: {len(three_way)}")
    
    all_drugs = d1.union(d2).union(d3)
    not_in_all = all_drugs - three_way
    if not_in_all:
        print(f"Drugs NOT shared by all three in {name}: {sorted(list(not_in_all))}")
    else:
        print(f"All {name} drugs are identical across the three datasets.")

compute_intersections(drug_sets_train, "train")
compute_intersections(drug_sets_test, "test")


print("\n===== Q2: Cross-dataset distribution comparability =====")
biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']

stats = {}
for d in datasets:
    train_file = f"data/{d}_training.csv" if d != "manual" else f"data/{d}_training(in).csv"
    df = pd.read_csv(train_file)
    stats[d] = df[biomarkers].agg(['mean', 'std', 'min', 'max']).T

for bm in biomarkers:
    print(f"\n--- {bm} ---")
    row_data = []
    means = []
    stds = []
    for d in datasets:
        m = stats[d].loc[bm, 'mean']
        s = stats[d].loc[bm, 'std']
        means.append(m)
        stds.append(s)
        print(f"{d:>10}: mean={m:10.4f}, std={s:10.4f}, min={stats[d].loc[bm, 'min']:10.4f}, max={stats[d].loc[bm, 'max']:10.4f}")
    
    mean_ratio = max(np.abs(means)) / min(np.abs(means)) if min(np.abs(means)) != 0 else float('inf')
    std_ratio = max(stds) / min(stds) if min(stds) != 0 else float('inf')
    print(f"Ratio of largest to smallest mean: {mean_ratio:.4f}")
    print(f"Ratio of largest to smallest std: {std_ratio:.4f}")

print("\n===== Q3: Drug-level confusion for the deployable model =====")
model_path = "./artifacts/deployable_chantest_3class.joblib"
model = joblib.load(model_path)
df_test = pd.read_csv("data/chantest_testing.csv").dropna()
X_test = df_test[biomarkers]
y_test = df_test['label'].values
drugs_test = df_test['drug_name'].values

y_pred = model.predict(X_test)
cm_sample = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print("Sample-level 3x3 Confusion Matrix (High=0, Int=1, Low=2):")
print(cm_sample)
print(f"Exact count of true-high samples predicted low: {cm_sample[0, 2]}")

y_proba = model.predict_proba(X_test)

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
        if "cv" in source and "KFold" in source and "RandomizedSearchCV" in source:
            print(f"--- Found CV usage in cell {i} ---")
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'KFold' in line or 'cv=' in line or 'cv =' in line:
                    print(f"L{j}: {line.strip()}")
