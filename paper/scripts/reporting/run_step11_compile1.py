import os
import sys
import numpy as np
import pandas as pd
import json
import hashlib

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
trainai_dir = os.path.join(base_dir, "TrainAI")

# STEP 1
bootstrap_md_path = os.path.join(base_dir, "docs", "STEP9_BOOTSTRAP.md")
with open(bootstrap_md_path, 'r', encoding='utf-8') as f:
    bootstrap_text = f.read()

orig_str = "Rank-11 Under-triage: 0.000000 [95% CI: 0.000000, 0.000000]"
rule_of_three_val = 3.0 / 16.0
new_str = f"Rank-11 Under-triage: 0 of 16 under-triaged drugs observed; approximate 95% upper bound on the true rate = {rule_of_three_val:.6f} ({rule_of_three_val*100:.2f}%)."
rep_str = f"~~{orig_str}~~ (SUPERSEDED)\n{new_str}"

print("=== STEP 1: REPLACE DEGENERATE UNDER-TRIAGE ===")
print("BEFORE:")
print(orig_str)
print("AFTER:")
print(new_str)

bootstrap_text = bootstrap_text.replace(orig_str, rep_str)
with open(bootstrap_md_path, 'w', encoding='utf-8') as f:
    f.write(bootstrap_text)

# STEP 2
print("\n=== STEP 2: DEGENERATE RESAMPLE DISCLOSURE ===")
lines = bootstrap_text.split("\n")
for i, line in enumerate(lines):
    if "DEGENERATE-RESAMPLE HANDLING" in line:
        for j in range(i, i+6):
            print(lines[j])
        break

# STEP 3
print("\n=== STEP 3: VERIFY IDENTICAL PLATFORM CIs ===")
SEVERITY = {0: 2, 1: 1, 2: 0}
Q_HAT_11 = 0.762830
THRESH_11 = 1 - Q_HAT_11

platforms = ['chantest', 'nanion', 'manual']
outcomes = {}
for p in platforms:
    fpath = os.path.join(trainai_dir, 'results', f'cross_platform_probs_{p}_rev.csv')
    df = pd.read_csv(fpath)
    df = df[df['split'] == 'testing'].reset_index(drop=True)
    
    p_outcomes = []
    for i in range(len(df)):
        drug = df['drug_name'].iloc[i]
        p_vec = df[['prob_high', 'prob_int', 'prob_low']].values[i]
        t_lbl = df['true_label'].iloc[i]
        t_sev = SEVERITY[t_lbl]
        
        cset = np.where(p_vec >= THRESH_11)[0].tolist()
        if len(cset) == 0:
            cset = [np.argmax(p_vec)]
            
        m_sev = max([SEVERITY[c] for c in cset])
        if m_sev == t_sev:
            p_outcomes.append("correct")
        elif m_sev > t_sev:
            p_outcomes.append("over-called")
        else:
            p_outcomes.append("under-triaged")
    outcomes[p] = p_outcomes
    
    cor = p_outcomes.count("correct")
    ovr = p_outcomes.count("over-called")
    und = p_outcomes.count("under-triaged")
    
    print(f"\nPlatform: {p}")
    for d, o in zip(df['drug_name'].values, p_outcomes):
        print(f"  {d:15s}: {o}")
    print(f"  Counts -> Correct: {cor}, Over: {ovr}, Under: {und}")

if outcomes['chantest'] == outcomes['nanion'] == outcomes['manual']:
    print("\nPASS (explained): The three outcome vectors are perfectly identical drug-for-drug. The differing coverage comes from correct sets having different sizes across platforms, even though the max severity (action) stays the same.")
else:
    print("\nERROR: platform outcome arrays are not identical.")
