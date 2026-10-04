import os
import sys
import json
import numpy as np
import pandas as pd

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS

def run_task7():
    print("==================================================================")
    print("=== TASK 7: BIOMARKER RANGES ACROSS PLATFORMS ===================")
    print("==================================================================")
    
    # Load cleaned training sets
    X_chan, _, _, _ = load_rev('chantest', 'training', data_dir=os.path.join(TRAINAI_DIR, 'data'))
    X_nan,  _, _, _ = load_rev('nanion',   'training', data_dir=os.path.join(TRAINAI_DIR, 'data'))
    X_man,  _, _, _ = load_rev('manual',   'training', data_dir=os.path.join(TRAINAI_DIR, 'data'))
    
    results = []
    
    for bm in CANONICAL_BIOMARKERS:
        s_chan = X_chan[bm]
        s_nan  = X_nan[bm]
        s_man  = X_man[bm]
        
        c_min = float(s_chan.min())
        c_p1  = float(np.percentile(s_chan, 1))
        c_med = float(s_chan.median())
        c_p99 = float(np.percentile(s_chan, 99))
        c_max = float(s_chan.max())
        
        n_min = float(s_nan.min())
        n_max = float(s_nan.max())
        
        m_min = float(s_man.min())
        m_max = float(s_man.max())
        
        results.append({
            'biomarker': bm,
            'chantest_min': c_min,
            'chantest_p1': c_p1,
            'chantest_median': c_med,
            'chantest_p99': c_p99,
            'chantest_max': c_max,
            'nanion_min': n_min,
            'nanion_max': n_max,
            'manual_min': m_min,
            'manual_max': m_max
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    out_json = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval', 'task7_biomarker_ranges.json')
    df_res.to_json(out_json, orient='records', indent=2)
    print(f"\nSaved Task 7 results to {out_json}")

if __name__ == '__main__':
    run_task7()
