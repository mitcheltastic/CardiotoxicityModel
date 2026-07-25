# NOTE: this script echoes summary prose from STEP8 verbatim; that prose is retracted.
import os
import sys
import numpy as np
import pandas as pd

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
trainai_dir = os.path.join(base_dir, "TrainAI")
sys.path.append(os.path.join(trainai_dir, 'src'))

from data_loader_rev import load_rev

output_lines = []
def log(msg=""):
    print(msg)
    output_lines.append(msg)

log("==========================================")
log("=== STEP 10 — VREST OVERLAP RECHECK ===")
log("==========================================")

# 1. Load Data
X_tr, y_tr, g_tr, _ = load_rev('chantest', 'training')
X_te, y_te, g_te, _ = load_rev('chantest', 'testing')

df_tr = pd.DataFrame(X_tr)
df_tr['drug_name'] = g_tr
df_tr['label'] = y_tr

df_te = pd.DataFrame(X_te)
df_te['drug_name'] = g_te
df_te['label'] = y_te

features = ['vrest', 'qNet', 'carest', 'dvdtmax']
splits = [('TRAINING (12 drugs)', df_tr, 66), ('TESTING (16 drugs)', df_te, 120)]

summary_records = []

for feat in features:
    log(f"\n##########################################")
    log(f"### FEATURE: {feat}")
    log(f"##########################################")
    
    for split_name, df_split, total_pairs in splits:
        log(f"\n--- SPLIT: {split_name} ---")
        drugs = sorted(df_split['drug_name'].unique())
        
        stats = {}
        for d in drugs:
            vals = df_split[df_split['drug_name'] == d][feat].values
            stats[d] = {
                'min': np.min(vals),
                'max': np.max(vals),
                'mean': np.mean(vals),
                'std': np.std(vals),
                'label': df_split[df_split['drug_name'] == d]['label'].iloc[0]
            }
            log(f"  {d:15s} (Label: {stats[d]['label']}) | min: {stats[d]['min']:.10f} | max: {stats[d]['max']:.10f} | mean: {stats[d]['mean']:.10f} | std: {stats[d]['std']:.10f}")
            
        overlapping_pairs = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                dA, dB = drugs[i], drugs[j]
                minA, maxA = stats[dA]['min'], stats[dA]['max']
                minB, maxB = stats[dB]['min'], stats[dB]['max']
                
                # Condition: max(minA, minB) <= min(maxA, maxB)
                if max(minA, minB) <= min(maxA, maxB):
                    overlapping_pairs.append((dA, dB, minA, maxA, minB, maxB))
                    
        overlap_cnt = len(overlapping_pairs)
        pct = (overlap_cnt / total_pairs) * 100.0
        log(f"\nTotal Pairs: {total_pairs} | Overlapping Pairs: {overlap_cnt} ({pct:.2f}%)")
        
        summary_records.append({
            'feature': feat,
            'split': split_name.split()[0],
            'overlap_count': overlap_cnt,
            'total_pairs': total_pairs,
            'pct': pct
        })
        
        log(f"Up to 10 overlapping pairs with ranges:")
        for k, (dA, dB, minA, maxA, minB, maxB) in enumerate(overlapping_pairs[:10]):
            log(f"  {k+1:2d}. {dA:12s} [{minA:.10f}, {maxA:.10f}] vs {dB:12s} [{minB:.10f}, {maxB:.10f}]")

log("\n==========================================")
log("=== SUMMARY TABLE: FEATURE x SPLIT ===")
log("==========================================")
log(f"{'Feature':10s} | {'Split':10s} | {'Overlap Count':15s} | {'Total Pairs':12s} | {'Percentage':10s}")
log("-" * 65)
for r in summary_records:
    log(f"{r['feature']:10s} | {r['split']:10s} | {r['overlap_count']:15d} | {r['total_pairs']:12d} | {r['pct']:6.2f}%")

vrest_tr = [r['overlap_count'] for r in summary_records if r['feature'] == 'vrest' and r['split'] == 'TRAINING'][0]
vrest_te = [r['overlap_count'] for r in summary_records if r['feature'] == 'vrest' and r['split'] == 'TESTING'][0]

log("\n--- VERDICT ON READINGS ---")
log(f"Recomputed vrest overlaps: Training = {vrest_tr}, Testing = {vrest_te}.")
if vrest_tr == 0 and vrest_te == 0:
    log("The recomputed vrest numbers support the reading of ZERO overlapping pairs.")
elif vrest_tr == 40 and vrest_te == 108:
    log("The recomputed vrest numbers support the reading of 40 (train) and 108 (test) overlapping pairs.")
else:
    log(f"The recomputed vrest numbers support NEITHER reading (Actual: train={vrest_tr}, test={vrest_te}).")

log("\n==========================================")
log("=== GREP docs/STEP8_SHAP_DIAGNOSTIC.md FOR OVERLAP SECTION ===")
log("==========================================")
step8_path = os.path.join(base_dir, "docs", "STEP8_SHAP_DIAGNOSTIC.md")
with open(step8_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_overlap_sec = False
step8_verbatim = []
for line in lines:
    if "Overlap" in line or "overlap" in line or "Analysis for" in line:
        in_overlap_sec = True
    if "STEP 3" in line:
        in_overlap_sec = False
    if in_overlap_sec:
        step8_verbatim.append(line.rstrip())
        log(line.rstrip())

disagrees = False
# Check if step8 counts match recomputed values
# Let's see if step8 text contains the exact numbers
log("\n--- COMPARISON WITH STEP8 TEXT ---")
for r in summary_records:
    f_str = f"Overlapping pairs:"
    # We will verify during script execution

doc_out_path = os.path.join(base_dir, "docs", "STEP10_VREST_OVERLAP_RECHECK.md")
trainai_doc_out_path = os.path.join(trainai_dir, "docs", "STEP10_VREST_OVERLAP_RECHECK.md")

with open(doc_out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

with open(trainai_doc_out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

print(f"\nWritten to {doc_out_path}")
