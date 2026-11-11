import os
import pandas as pd
import numpy as np

base_dir = r"d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah"
results_dir = os.path.join(base_dir, "TrainAI", "results")

Q_HAT_11 = 0.762830
THRESH = 1.0 - Q_HAT_11 # 0.237170
platforms = ['chantest', 'nanion', 'manual']

for p in platforms:
    print(f"\n==========================================")
    print(f"=== PLATFORM: {p.upper()} ===")
    print(f"==========================================")
    df = pd.read_csv(os.path.join(results_dir, f'cross_platform_probs_{p}_rev.csv'))
    df = df[df['split'] == 'testing'].reset_index(drop=True)
    
    misses = []
    clears = []
    
    p_cols = ['prob_high', 'prob_int', 'prob_low']
    
    for i in range(len(df)):
        drug = df['drug_name'].iloc[i]
        t = int(df['true_label'].iloc[i])
        p_vec = df[p_cols].values[i]
        p_true = p_vec[t]
        
        cset = np.where(p_vec >= THRESH)[0].tolist()
        if len(cset) == 0:
            cset = [np.argmax(p_vec)]
            
        status = "CLEARS" if (t in cset) else "MISSES"
        if status == "MISSES":
            misses.append(drug)
        else:
            clears.append(drug)
            
        print(f"{drug:15s} | True: {t} | p_mean[true]: {p_true:.6f} | Thresh: {THRESH:.6f} | Status: {status}")
        
    print(f"\nSummary for {p}: Total = {len(df)}, Clears = {len(clears)}, Misses = {len(misses)}")
    print(f"Miss list: {misses}")
