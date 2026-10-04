import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import platform
from sklearn.model_selection import train_test_split

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS, LABEL_MAP
from results.cd4_eval.eval_engine import (
    evaluate_predictions, get_drug_level_summary, classify_misclassifications,
    calc_ordinal_aucs_drug, calc_ordinal_aucs_row, run_drug_bootstrap, run_cipa_bootstrap,
    INV_LABEL_MAP
)

RF_OUT_DIR = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval', 'rf')
os.makedirs(RF_OUT_DIR, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 10000

def get_cpu_info():
    try:
        cmd = "wmic cpu get name"
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        lines = [line.strip() for line in out.split('\n') if line.strip() and 'Name' not in line]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"

def main():
    print("==================================================================")
    print("=== RANDOM FOREST BASE LEARNER COMPREHENSIVE CD-4 EVALUATION ===")
    print("==================================================================")
    
    # 0. Load and Verify Model Equivalence
    dep_bundle = joblib.load(os.path.join(TRAINAI_DIR, 'hf_space', 'deployable_chantest_3class.joblib'))
    rf_model = dep_bundle['model'].estimators_[0]
    
    res_bundle = joblib.load(os.path.join(TRAINAI_DIR, 'models', 'chantest_stacking_model_rev.joblib'))
    rf_res = res_bundle.named_estimators_['rf']
    
    print(f"Loaded RF from deployable bundle: {type(rf_model)}")
    print(f"Number of estimators: {len(rf_model.estimators_)}")
    print(f"Criterion: {rf_model.criterion}, Max Depth: {rf_model.max_depth}")
    
    # Verify predictions on all 3 platforms
    for p in ['chantest', 'nanion', 'manual']:
        X, y, g, r = load_rev(p, 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
        diff = np.max(np.abs(rf_model.predict_proba(X.values) - rf_res.predict_proba(X.values)))
        print(f"Prediction max diff on {p:<8}: {diff:.8f}")
        assert diff == 0.0, f"RF models not identical on {p}!"
    print("--> CONFIRMED: RF in deployable bundle produces 100% IDENTICAL predictions to models/chantest_stacking_model_rev.joblib.")

    # ==================================================================
    # 1. EVALUATION ON CHANTEST, NANION, MANUAL TEST SETS
    # ==================================================================
    print("\n==================================================================")
    print("=== 1. TEST SET PERFORMANCE (10,000 BOOTSTRAP, SEED 42) ==========")
    print("==================================================================")
    
    platforms = ['chantest', 'nanion', 'manual']
    test_results = {}
    
    for platform in platforms:
        print(f"\n--- Platform: {platform.upper()} ---")
        X_test, y_test, groups_test, r_test = load_rev(platform, 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
        print(f"Loaded {len(X_test)} rows across {groups_test.nunique()} test drugs.")
        
        # Row-level predict proba
        row_probs = rf_model.predict_proba(X_test.values)
        y_test_pred_row = np.argmax(row_probs, axis=1)
        
        # Row-level metrics
        row_metrics = evaluate_predictions(y_test.values, y_test_pred_row)
        auc1_row, auc2_row = calc_ordinal_aucs_row(y_test.values, row_probs)
        
        # Drug-level summary
        df_row_probs = pd.DataFrame(row_probs, columns=['prob_high', 'prob_int', 'prob_low'])
        df_row_probs['drug_name'] = groups_test.values
        df_row_probs['true_label'] = y_test.values
        
        drug_summary = get_drug_level_summary(df_row_probs)
        drug_y_true = drug_summary['true_label'].values
        drug_y_pred = drug_summary['pred_label'].values
        
        drug_metrics = evaluate_predictions(drug_y_true, drug_y_pred)
        auc1_drug, auc2_drug = calc_ordinal_aucs_drug(drug_summary)
        
        correct_drugs = int(np.sum(drug_y_true == drug_y_pred))
        total_drugs = len(drug_y_true)
        
        misclass_df = classify_misclassifications(drug_summary)
        
        # Drug bootstrap
        print("Running 10,000 drug-level bootstrap iterations...")
        boot_drug = run_drug_bootstrap(drug_summary, n_iter=N_BOOTSTRAP, seed=SEED)
        
        # CiPA replicate bootstrap
        print("Running 10,000 CiPA replicate bootstrap iterations...")
        cipa_boot = run_cipa_bootstrap(groups_test, y_test, row_probs, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        
        res = {
            'platform': platform,
            'correct_drugs': f"{correct_drugs}/{total_drugs}",
            'drug_macro_f1': drug_metrics['macro_f1'],
            'drug_accuracy': drug_metrics['accuracy'],
            'drug_precision': drug_metrics['precision'].tolist(),
            'drug_recall': drug_metrics['recall'].tolist(),
            'drug_f1': drug_metrics['f1'].tolist(),
            'drug_cm': drug_metrics['confusion_matrix'].tolist(),
            'misclassifications': misclass_df.to_dict(orient='records'),
            'row_macro_f1': row_metrics['macro_f1'],
            'row_accuracy': row_metrics['accuracy'],
            'row_cm': row_metrics['confusion_matrix'].tolist(),
            'ordinal_auc_drug': {'risk_gt_low': auc1_drug, 'risk_gt_int': auc2_drug},
            'ordinal_auc_row': {'risk_gt_low': auc1_row, 'risk_gt_int': auc2_row},
            'drug_bootstrap_macro_f1': {
                'median': boot_drug['median'],
                'ci_low': boot_drug['ci_low'],
                'ci_high': boot_drug['ci_high'],
                'ci_str': f"{boot_drug['median']:.6f} [{boot_drug['ci_low']:.6f}, {boot_drug['ci_high']:.6f}]"
            },
            'cipa_bootstrap_ovr_auc': cipa_boot,
            'drug_summary': drug_summary.to_dict(orient='records')
        }
        test_results[platform] = res
        
        print(f"Correct: {correct_drugs}/{total_drugs} ({drug_metrics['accuracy']*100:.2f}%)")
        print(f"Drug Macro-F1: {drug_metrics['macro_f1']:.6f}")
        print(f"Drug Bootstrap 95% CI: [{boot_drug['ci_low']:.6f}, {boot_drug['ci_high']:.6f}]")
        print(f"Per-class Precision: High={drug_metrics['precision'][0]:.4f}, Int={drug_metrics['precision'][1]:.4f}, Low={drug_metrics['precision'][2]:.4f}")
        print(f"Per-class Recall:    High={drug_metrics['recall'][0]:.4f}, Int={drug_metrics['recall'][1]:.4f}, Low={drug_metrics['recall'][2]:.4f}")
        print(f"Per-class F1:        High={drug_metrics['f1'][0]:.4f}, Int={drug_metrics['f1'][1]:.4f}, Low={drug_metrics['f1'][2]:.4f}")
        print(f"Drug Confusion Matrix [H, I, L]:\n{drug_metrics['confusion_matrix']}")
        print(f"Misclassified Drugs ({len(misclass_df)}):")
        for _, m in misclass_df.iterrows():
            print(f"  {m['drug_name']:<16}: {m['true_tier']:<12} -> {m['pred_tier']:<12} ({m['category']})")
        print(f"Row Macro-F1: {row_metrics['macro_f1']:.6f} | Row Accuracy: {row_metrics['accuracy']:.6f}")
        print(f"Ordinal AUC (Drug): risk>Low={auc1_drug:.4f}, risk>Int={auc2_drug:.4f}")
        print(f"Ordinal AUC (Row):  risk>Low={auc1_row:.4f}, risk>Int={auc2_row:.4f}")
        print(f"CiPA Bootstrap OvR AUC: High={cipa_boot['High']['str']}, Int={cipa_boot['Int']['str']}, Low={cipa_boot['Low']['str']}")

    with open(os.path.join(RF_OUT_DIR, 'rf_test_metrics.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    # ==================================================================
    # 2. FULL RANK-11 LODO CONFORMAL TABLE FOR RF
    # ==================================================================
    print("\n==================================================================")
    print("=== 2. FULL RANK-11 LODO CONFORMAL TABLE & PLATFORM METRICS =====")
    print("==================================================================")
    
    # 12 Training drugs LODO data from verified STEP13B
    lodo_calibration_rf = [
        {'drug_name': 'cisapride',      'true_class': 'intermediate', 'true_idx': 1, 'prob_high': 0.023487, 'prob_int': 0.976482, 'prob_low': 0.000030, 's_d': 0.023518},
        {'drug_name': 'dofetilide',     'true_class': 'high',         'true_idx': 0, 'prob_high': 0.974995, 'prob_int': 0.024734, 'prob_low': 0.000271, 's_d': 0.025005},
        {'drug_name': 'terfenadine',    'true_class': 'intermediate', 'true_idx': 1, 'prob_high': 0.058741, 'prob_int': 0.897190, 'prob_low': 0.044069, 's_d': 0.102810},
        {'drug_name': 'mexiletine',     'true_class': 'low',          'true_idx': 2, 'prob_high': 0.006357, 'prob_int': 0.096546, 'prob_low': 0.897096, 's_d': 0.102904},
        {'drug_name': 'bepridil',       'true_class': 'high',         'true_idx': 0, 'prob_high': 0.895961, 'prob_int': 0.002966, 'prob_low': 0.101073, 's_d': 0.104039},
        {'drug_name': 'chlorpromazine', 'true_class': 'intermediate', 'true_idx': 1, 'prob_high': 0.001469, 'prob_int': 0.757775, 'prob_low': 0.240756, 's_d': 0.242225},
        {'drug_name': 'quinidine',      'true_class': 'high',         'true_idx': 0, 'prob_high': 0.726190, 'prob_int': 0.018756, 'prob_low': 0.255054, 's_d': 0.273810},
        {'drug_name': 'ranolazine',     'true_class': 'low',          'true_idx': 2, 'prob_high': 0.043715, 'prob_int': 0.279179, 'prob_low': 0.677106, 's_d': 0.322894},
        {'drug_name': 'verapamil',      'true_class': 'low',          'true_idx': 2, 'prob_high': 0.018283, 'prob_int': 0.350695, 'prob_low': 0.631023, 's_d': 0.368977},
        {'drug_name': 'ondansetron',    'true_class': 'intermediate', 'true_idx': 1, 'prob_high': 0.264494, 'prob_int': 0.621936, 'prob_low': 0.113570, 's_d': 0.378064},
        {'drug_name': 'sotalol',        'true_class': 'high',         'true_idx': 0, 'prob_high': 0.466013, 'prob_int': 0.479129, 'prob_low': 0.054859, 's_d': 0.533987},
        {'drug_name': 'diltiazem',      'true_class': 'low',          'true_idx': 2, 'prob_high': 0.000000, 'prob_int': 0.904169, 'prob_low': 0.095831, 's_d': 0.904169}
    ]
    
    df_lodo = pd.DataFrame(lodo_calibration_rf)
    df_lodo['rank'] = range(1, 13)
    
    q_hat_11 = 0.533987
    alpha_11 = 2.0 / 13.0 # 0.153846
    nom_cov_11 = 11.0 / 13.0 # 0.846154
    thresh_11 = 1.0 - q_hat_11 # 0.466013
    
    print(f"Rank 11 Calibration Drug: sotalol (s_d = {q_hat_11:.6f})")
    print(f"Quantile q_hat:           {q_hat_11:.6f}")
    print(f"Alpha:                    {alpha_11:.6f}")
    print(f"Nominal Coverage:         {nom_cov_11:.6f} ({nom_cov_11*100:.3f}%)")
    print(f"Inclusion Threshold:      {thresh_11:.6f}")
    
    # Evaluate across test platforms
    severity_map = {0: 2, 1: 1, 2: 0} # High=2, Int=1, Low=0
    conformal_platform_eval = {}
    
    for platform in platforms:
        d_summary = pd.DataFrame(test_results[platform]['drug_summary'])
        records = []
        covered_count = 0
        total_size = 0
        n_ambiguous = 0
        n_empty = 0
        n_under_triage = 0
        
        for _, row in d_summary.iterrows():
            t = int(row['true_label'])
            probs = [row['prob_high'], row['prob_int'], row['prob_low']]
            
            # Prediction set
            pred_set = [c for c in range(3) if probs[c] >= thresh_11]
            pred_set.sort(key=lambda c: severity_map[c], reverse=True)
            
            size = len(pred_set)
            total_size += size
            
            is_cov = (t in pred_set)
            if is_cov:
                covered_count += 1
            if size > 1:
                n_ambiguous += 1
            elif size == 0:
                n_empty += 1
                
            # Decision action:
            # If set not empty: most severe class in pred_set
            # If set empty (abstain): fallback to argmax of probs or full set
            if size > 0:
                action_class = pred_set[0]
                action_sev = severity_map[action_class]
            else:
                action_class = int(np.argmax(probs))
                action_sev = severity_map[action_class]
                
            true_sev = severity_map[t]
            if action_sev < true_sev:
                outcome = "under-triage"
                n_under_triage += 1
            elif action_sev > true_sev:
                outcome = "over-call"
            else:
                outcome = "correct"
                
            records.append({
                'drug_name': row['drug_name'],
                'true_tier': INV_LABEL_MAP[t],
                'p_high': row['prob_high'],
                'p_int': row['prob_int'],
                'p_low': row['prob_low'],
                'prediction_set': [INV_LABEL_MAP[c] for c in pred_set],
                'set_size': size,
                'action_tier': INV_LABEL_MAP[action_class],
                'outcome': outcome,
                'is_covered': is_cov
            })
            
        conformal_platform_eval[platform] = {
            'empirical_coverage': covered_count / len(d_summary),
            'covered_ratio': f"{covered_count}/{len(d_summary)}",
            'avg_set_size': total_size / len(d_summary),
            'ambiguous_drugs_count': n_ambiguous,
            'ambiguous_drugs_list': [r['drug_name'] for r in records if r['set_size'] > 1],
            'empty_sets_count': n_empty,
            'empty_sets_list': [r['drug_name'] for r in records if r['set_size'] == 0],
            'under_triage_count': n_under_triage,
            'under_triage_list': [r['drug_name'] for r in records if r['outcome'] == 'under-triage'],
            'drug_records': records
        }
        
        print(f"\nPlatform: {platform.upper()}")
        print(f"  Empirical Coverage: {covered_count}/{len(d_summary)} ({covered_count/len(d_summary)*100:.2f}%)")
        print(f"  Avg Set Size:       {total_size/len(d_summary):.4f}")
        print(f"  Ambiguous Drugs:    {n_ambiguous} ({[r['drug_name'] for r in records if r['set_size'] > 1]})")
        print(f"  Empty Sets:         {n_empty}")
        print(f"  Under-triage:       {n_under_triage}")

    conformal_output = {
        'rank_11_parameters': {
            'drug': 'sotalol',
            'q_hat': q_hat_11,
            'alpha': alpha_11,
            'nominal_coverage': nom_cov_11,
            'threshold': thresh_11
        },
        'lodo_calibration_table': df_lodo.to_dict(orient='records'),
        'platforms': conformal_platform_eval
    }
    with open(os.path.join(RF_OUT_DIR, 'rf_conformal_rank11.json'), 'w') as f:
        json.dump(conformal_output, f, indent=2)

    # ==================================================================
    # 3. GLOBAL SHAP FOR RF (TREEEXPLAINER, 320-ROW SUBSAMPLE)
    # ==================================================================
    print("\n==================================================================")
    print("=== 3. GLOBAL SHAP (TREEEXPLAINER, 320-ROW TEST SUBSAMPLE) ======")
    print("==================================================================")
    
    X_test_full, y_test_full, groups_test_full, _ = load_rev('chantest', 'testing', data_dir=os.path.join(TRAINAI_DIR, 'data'))
    _, X_test_sub, _, groups_test_sub = train_test_split(
        X_test_full, groups_test_full, test_size=320, random_state=SEED, stratify=groups_test_full
    )
    print(f"Subsampled test set shape: {X_test_sub.shape} across {groups_test_sub.nunique()} drugs.")
    
    print("Initializing shap.TreeExplainer on RF base learner...")
    tree_explainer = shap.TreeExplainer(rf_model)
    print("Computing TreeExplainer SHAP values on 320 rows...")
    t0_shap = time.time()
    shap_vals = tree_explainer.shap_values(X_test_sub)
    t1_shap = time.time()
    print(f"SHAP values computed in {t1_shap - t0_shap:.2f} s.")
    
    # Check shape: for multi-class, TreeExplainer returns list of 3 arrays or (320, 11, 3)
    if isinstance(shap_vals, list):
        mean_abs_high = np.abs(shap_vals[0]).mean(axis=0)
        mean_abs_int  = np.abs(shap_vals[1]).mean(axis=0)
        mean_abs_low  = np.abs(shap_vals[2]).mean(axis=0)
    else:
        mean_abs = np.abs(shap_vals).mean(axis=0)
        if mean_abs.ndim == 2 and mean_abs.shape[1] == 3:
            mean_abs_high = mean_abs[:, 0]
            mean_abs_int  = mean_abs[:, 1]
            mean_abs_low  = mean_abs[:, 2]
        else:
            mean_abs_high = mean_abs[0]
            mean_abs_int  = mean_abs[1]
            mean_abs_low  = mean_abs[2]
        
    mean_abs_agg = (mean_abs_high + mean_abs_int + mean_abs_low) / 3.0
    
    shap_records = []
    for i, bm in enumerate(CANONICAL_BIOMARKERS):
        shap_records.append({
            'biomarker': bm,
            'mean_abs_shap_high': float(mean_abs_high[i]),
            'mean_abs_shap_int': float(mean_abs_int[i]),
            'mean_abs_shap_low': float(mean_abs_low[i]),
            'mean_abs_shap_aggregated': float(mean_abs_agg[i])
        })
    df_shap = pd.DataFrame(shap_records).sort_values(by='mean_abs_shap_aggregated', ascending=False).reset_index(drop=True)
    df_shap['rank'] = df_shap.index + 1
    
    shap_csv_path = os.path.join(RF_OUT_DIR, 'rf_shap_importance.csv')
    df_shap.to_csv(shap_csv_path, index=False)
    print(f"Saved SHAP importance table to: {shap_csv_path}")
    print(df_shap[['rank', 'biomarker', 'mean_abs_shap_high', 'mean_abs_shap_int', 'mean_abs_shap_low', 'mean_abs_shap_aggregated']].to_string(index=False))
    
    # Generate 300-dpi figure matching results/cd4_eval/fig_shap_global_fixed.png
    biomarker_label_map = {
        'qNet': 'qNet',
        'vrest': r'$V_{m,rest}$',
        'max_dv': r'$dV_m/dt_{repol}$',
        'APD50': r'APD$_{50}$',
        'APD90': r'APD$_{90}$',
        'vmax': r'$V_{m,peak}$',
        'dvdtmax': r'$dV_m/dt_{max}$',
        'camax': r'Ca$_{peak}$',
        'CaTD50': r'CaD$_{50}$',
        'CaTD90': r'CaD$_{90}$',
        'carest': r'Ca$_{rest}$'
    }
    
    plot_df = df_shap.sort_values(by='mean_abs_shap_aggregated', ascending=True).reset_index(drop=True)
    
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif']
    })
    
    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=300)
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df['mean_abs_shap_aggregated'], color='#3B6EA5', edgecolor='#1F4E79', height=0.68)
    
    ax.set_yticks(y_pos)
    labels = [biomarker_label_map.get(bm, bm) for bm in plot_df['biomarker']]
    ax.set_yticklabels(labels)
    
    ax.set_xlabel(r'Global Mean Absolute SHAP Value ($\mathrm{mean}|\mathrm{SHAP}|$)')
    ax.set_title('Global Feature Importance (Random Forest, ChanTest Test Split)', fontsize=11, fontweight='bold', pad=12)
    
    max_val = plot_df['mean_abs_shap_aggregated'].max()
    ax.set_xlim(0, max_val * 1.18)
    
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        ax.text(w + max_val * 0.015, y, f'{w:.4f}', va='center', ha='left', fontsize=8.5, color='#222222')
        
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    
    fig_png = os.path.join(RF_OUT_DIR, 'fig_shap_global_rf.png')
    fig_pdf = os.path.join(RF_OUT_DIR, 'fig_shap_global_rf.pdf')
    plt.savefig(fig_png, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved figure to:\n  PNG (300 dpi): {fig_png}\n  PDF: {fig_pdf}")

    # ==================================================================
    # 4. LOCAL LATENCY BENCHMARK ON ONE ROW
    # ==================================================================
    print("\n==================================================================")
    print("=== 4. LOCAL LATENCY BENCHMARK ON ONE ROW =======================")
    print("==================================================================")
    cpu_model = get_cpu_info()
    print(f"CPU Model: {cpu_model}")
    
    # 1 input row (Azimilide from chantest test set)
    input_row = np.array([[43.637118, 72.87939091, 41.82645396, -88.01230706, 338.5000001, 441.25, -0.238296891, 0.000254324, 7.60e-05, 337.4999999, 767.2500001]])
    
    # A. RF predict_proba (100 runs)
    print("\nBenchmarking RF predict_proba (100 runs)...")
    for _ in range(10):
        _ = rf_model.predict_proba(input_row)
        
    latencies_pred = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = rf_model.predict_proba(input_row)
        t1 = time.perf_counter()
        latencies_pred.append((t1 - t0) * 1000.0)
    arr_pred = np.array(latencies_pred)
    pred_stats = {
        'mean_ms': float(np.mean(arr_pred)),
        'p50_ms': float(np.percentile(arr_pred, 50)),
        'p95_ms': float(np.percentile(arr_pred, 95)),
        'max_ms': float(np.max(arr_pred))
    }
    print(f"RF predict_proba (ms): Mean={pred_stats['mean_ms']:.4f}, P50={pred_stats['p50_ms']:.4f}, P95={pred_stats['p95_ms']:.4f}, Max={pred_stats['max_ms']:.4f}")
    
    # B. TreeExplainer explain (30 runs)
    print("\nBenchmarking TreeExplainer explain (30 runs)...")
    _ = tree_explainer.shap_values(input_row)
    
    latencies_tree = []
    for i in range(30):
        t0 = time.perf_counter()
        _ = tree_explainer.shap_values(input_row)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies_tree.append(lat_ms)
        if (i + 1) % 10 == 0:
            print(f"  Run {i+1}/30: {lat_ms:.2f} ms")
    arr_tree = np.array(latencies_tree)
    tree_stats = {
        'mean_ms': float(np.mean(arr_tree)),
        'p50_ms': float(np.percentile(arr_tree, 50)),
        'p95_ms': float(np.percentile(arr_tree, 95)),
        'max_ms': float(np.max(arr_tree))
    }
    print(f"TreeExplainer explain (ms): Mean={tree_stats['mean_ms']:.4f}, P50={tree_stats['p50_ms']:.4f}, P95={tree_stats['p95_ms']:.4f}, Max={tree_stats['max_ms']:.4f}")
    
    # C. KernelExplainer with 15 k-means centroids and nsamples=500 (30 runs)
    print("\nBenchmarking KernelExplainer with 15 k-means centroids, nsamples=500 (30 runs)...")
    bg_centroids = joblib.load(os.path.join(TRAINAI_DIR, 'hf_space', 'shap_background.joblib'))
    kernel_explainer = shap.KernelExplainer(rf_model.predict_proba, bg_centroids)
    
    print("KernelExplainer warmup run...")
    _ = kernel_explainer.shap_values(input_row, nsamples=500, silent=True)
    
    latencies_kernel = []
    for i in range(30):
        t0 = time.perf_counter()
        pred_cls = int(rf_model.predict(input_row)[0])
        sv = kernel_explainer.shap_values(input_row, nsamples=500, silent=True)
        if isinstance(sv, list):
            _ = sv[pred_cls][0]
        else:
            if sv.ndim == 3:
                _ = sv[0, :, pred_cls]
            else:
                _ = sv[0]
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies_kernel.append(lat_ms)
        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/30: {lat_ms:.2f} ms")
    arr_kernel = np.array(latencies_kernel)
    kernel_stats = {
        'mean_ms': float(np.mean(arr_kernel)),
        'p50_ms': float(np.percentile(arr_kernel, 50)),
        'p95_ms': float(np.percentile(arr_kernel, 95)),
        'max_ms': float(np.max(arr_kernel))
    }
    print(f"KernelExplainer explain (ms): Mean={kernel_stats['mean_ms']:.4f}, P50={kernel_stats['p50_ms']:.4f}, P95={kernel_stats['p95_ms']:.4f}, Max={kernel_stats['max_ms']:.4f}")
    
    latency_results = {
        'cpu_model': cpu_model,
        'rf_predict_proba': pred_stats,
        'tree_explainer': tree_stats,
        'kernel_explainer_15centroids': kernel_stats
    }
    with open(os.path.join(RF_OUT_DIR, 'rf_latency_benchmark.json'), 'w') as f:
        json.dump(latency_results, f, indent=2)
    print(f"\nSaved latency benchmark to {os.path.join(RF_OUT_DIR, 'rf_latency_benchmark.json')}")
    print("\nALL TASKS COMPLETE SUCCESSFULLY!")

if __name__ == '__main__':
    main()
