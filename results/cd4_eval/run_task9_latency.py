import os
import sys
import time
import json
import joblib
import numpy as np
import shap
import platform
import subprocess

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import CANONICAL_BIOMARKERS

def get_cpu_info():
    try:
        cmd = "wmic cpu get name"
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        lines = [line.strip() for line in out.split('\n') if line.strip() and 'Name' not in line]
        if lines:
            return lines[0]
    except Exception as e:
        pass
    return platform.processor() or "Unknown CPU"

def run_task9():
    print("==================================================================")
    print("=== TASK 9: LOCAL LATENCY BENCHMARK ==============================")
    print("==================================================================")
    
    cpu_model = get_cpu_info()
    print(f"CPU Model: {cpu_model}")
    
    hf_dir = os.path.join(TRAINAI_DIR, 'hf_space')
    bundle = joblib.load(os.path.join(hf_dir, 'deployable_chantest_3class.joblib'))
    model = bundle['model']
    background = joblib.load(os.path.join(hf_dir, 'shap_background.joblib'))
    
    # 1 input row (e.g. azimilide from chantest test set)
    # Azimilide: [43.637118, 72.87939091, 41.82645396, -88.01230706, 338.5000001, 441.25, -0.238296891, 0.000254324, 7.60e-05, 337.4999999, 767.2500001]
    input_row = np.array([[43.637118, 72.87939091, 41.82645396, -88.01230706, 338.5000001, 441.25, -0.238296891, 0.000254324, 7.60e-05, 337.4999999, 767.2500001]])
    
    # Warmup
    for _ in range(5):
        _ = model.predict_proba(input_row)
        
    # 1. predict_proba latency (100 runs)
    print("\nBenchmarking predict_proba (100 runs)...")
    latencies_predict = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict_proba(input_row)
        t1 = time.perf_counter()
        latencies_predict.append((t1 - t0) * 1000.0) # ms
        
    arr_pred = np.array(latencies_predict)
    pred_stats = {
        'mean_ms': float(np.mean(arr_pred)),
        'p50_ms': float(np.percentile(arr_pred, 50)),
        'p95_ms': float(np.percentile(arr_pred, 95)),
        'max_ms': float(np.max(arr_pred))
    }
    print(f"predict_proba (ms): Mean={pred_stats['mean_ms']:.3f}, P50={pred_stats['p50_ms']:.3f}, P95={pred_stats['p95_ms']:.3f}, Max={pred_stats['max_ms']:.3f}")
    
    # 2. SHAP explain latency (30 runs)
    # Settings from app.py: KernelExplainer, background from shap_background.joblib, nsamples=500, predicted class only
    print("\nInitializing KernelExplainer...")
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Warmup SHAP (JIT compile)
    print("Warmup SHAP run...")
    _ = explainer.shap_values(input_row, nsamples=500, silent=True)
    
    print("Benchmarking SHAP explain (30 runs, nsamples=500, predicted class only)...")
    latencies_shap = []
    for i in range(30):
        t0 = time.perf_counter()
        pred_cls = int(model.predict(input_row)[0])
        sv = explainer.shap_values(input_row, nsamples=500, silent=True)
        if isinstance(sv, list):
            _ = sv[pred_cls][0]
        else:
            if sv.ndim == 3:
                _ = sv[0, :, pred_cls]
            else:
                _ = sv[0]
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies_shap.append(lat_ms)
        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/30: {lat_ms:.2f} ms")
            
    arr_shap = np.array(latencies_shap)
    shap_stats = {
        'mean_ms': float(np.mean(arr_shap)),
        'p50_ms': float(np.percentile(arr_shap, 50)),
        'p95_ms': float(np.percentile(arr_shap, 95)),
        'max_ms': float(np.max(arr_shap))
    }
    print(f"SHAP explain (ms): Mean={shap_stats['mean_ms']:.3f}, P50={shap_stats['p50_ms']:.3f}, P95={shap_stats['p95_ms']:.3f}, Max={shap_stats['max_ms']:.3f}")
    
    results = {
        'cpu_model': cpu_model,
        'predict_proba': pred_stats,
        'shap_explain': shap_stats
    }
    
    out_json = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval', 'task9_latency_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Task 9 results to {out_json}")

if __name__ == '__main__':
    run_task9()
