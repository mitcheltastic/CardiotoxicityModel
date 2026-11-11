import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# 1. Environment & Paths Setup
TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
RESULTS_DIR = os.path.join(TRAINAI_DIR, 'results')

MODELS = ['Stacking', 'RF', 'LR', 'LR-qNet']
PLATFORMS = ['chantest', 'nanion', 'manual']

FILE_MAP = {
    'Stacking': {p: os.path.join(RESULTS_DIR, f'cross_platform_probs_{p}_rev.csv') for p in PLATFORMS},
    'RF': {p: os.path.join(RESULTS_DIR, f'rf_probs_{p}_rev.csv') for p in PLATFORMS},
    'LR': {p: os.path.join(RESULTS_DIR, f'logistic_multivariate_probs_{p}_rev.csv') for p in PLATFORMS},
    'LR-qNet': {p: os.path.join(RESULTS_DIR, f'logistic_qnet_probs_{p}_rev.csv') for p in PLATFORMS}
}

# 2. Read 12 CSV files, filter split == 'testing' (16 rows each)
datasets = {}
for model in MODELS:
    datasets[model] = {}
    for platform in PLATFORMS:
        fpath = FILE_MAP[model][platform]
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")
        df = pd.read_csv(fpath)
        df_test = df[df['split'] == 'testing'].reset_index(drop=True)
        if len(df_test) != 16:
            raise ValueError(f"Expected 16 testing rows in {fpath}, got {len(df_test)}")
        datasets[model][platform] = {
            'true_label': df_test['true_label'].values.astype(int),
            'prob_high': df_test['prob_high'].values.astype(float),
            'prob_int': df_test['prob_int'].values.astype(float),
            'prob_low': df_test['prob_low'].values.astype(float),
            'probs': df_test[['prob_high', 'prob_int', 'prob_low']].values.astype(float)
        }

# 3. Bootstrap 10,000 iterations (seed 42)
N_BOOTSTRAP = 10000
SEED = 42

np.random.seed(SEED)
boot_indices = np.random.choice(16, size=(N_BOOTSTRAP, 16), replace=True)

def calc_auc_fast(pos_mat, score_mat):
    # Vectorized calculation equivalent to sklearn.metrics.roc_auc_score
    is_pos = pos_mat[:, :, None]
    is_neg = (~pos_mat)[:, None, :]
    s_pos = score_mat[:, :, None]
    s_neg = score_mat[:, None, :]
    
    pair_mask = is_pos & is_neg
    win = (s_pos > s_neg).astype(float) + 0.5 * (s_pos == s_neg).astype(float)
    
    n_pos = np.sum(is_pos, axis=(1, 2))
    n_neg = np.sum(is_neg, axis=(1, 2))
    n_pairs = np.sum(pair_mask, axis=(1, 2))
    
    valid = (n_pos > 0) & (n_neg > 0) & (n_pairs > 0)
    res = np.full(pos_mat.shape[0], np.nan)
    res[valid] = np.sum(win * pair_mask, axis=(1, 2))[valid] / n_pairs[valid]
    return res

def format_result(med, lo, hi):
    if np.isnan(med):
        return "N/A"
    return f"{med:.3f} ({lo:.3f}-{hi:.3f})"

metrics_list = [
    'macro_f1',
    'prec_class0_high', 'rec_class0_high', 'f1_class0_high',
    'prec_class1_int',  'rec_class1_int',  'f1_class1_int',
    'prec_class2_low',  'rec_class2_low',  'f1_class2_low',
    'auc_thresh1', 'auc_thresh2'
]

results_records = []

print("==========================================================")
print("  STEP 21 SIMPLE - BOOTSTRAP RESULTS (10,000 ITERATIONS)  ")
print("==========================================================")

for model in MODELS:
    for platform in PLATFORMS:
        data = datasets[model][platform]
        y_true = data['true_label']
        p_high = data['prob_high']
        p_int = data['prob_int']
        p_low = data['prob_low']
        probs = data['probs']
        
        # Verify point AUC using sklearn.metrics.roc_auc_score directly
        pos1_point = (y_true == 0) | (y_true == 1)
        score1_point = p_high + p_int
        auc1_sk = roc_auc_score(pos1_point, score1_point)
        
        pos2_point = (y_true == 0)
        score2_point = p_high
        auc2_sk = roc_auc_score(pos2_point, score2_point)

        # Vectorized resampling over 10,000 iterations
        yt_mat = y_true[boot_indices]         # (10000, 16)
        ph_mat = p_high[boot_indices]         # (10000, 16)
        pi_mat = p_int[boot_indices]          # (10000, 16)
        pl_mat = p_low[boot_indices]          # (10000, 16)
        probs_mat = probs[boot_indices]       # (10000, 16, 3)
        yp_mat = np.argmax(probs_mat, axis=2) # (10000, 16)

        # Vectorized 3x3 confusion matrix per row
        counts_mat = np.zeros((N_BOOTSTRAP, 9), dtype=int)
        for c_t in range(3):
            for c_p in range(3):
                counts_mat[:, c_t * 3 + c_p] = np.sum((yt_mat == c_t) & (yp_mat == c_p), axis=1)
        
        cm_mat = counts_mat.reshape(N_BOOTSTRAP, 3, 3)
        tp_mat = np.diagonal(cm_mat, axis1=1, axis2=2) # (10000, 3)
        fp_mat = np.sum(cm_mat, axis=1) - tp_mat       # (10000, 3)
        fn_mat = np.sum(cm_mat, axis=2) - tp_mat       # (10000, 3)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            prec_mat = np.where(tp_mat + fp_mat > 0, tp_mat / (tp_mat + fp_mat), np.nan)
            rec_mat  = np.where(tp_mat + fn_mat > 0, tp_mat / (tp_mat + fn_mat), np.nan)
            f1_mat   = np.where((~np.isnan(prec_mat)) & (~np.isnan(rec_mat)) & (prec_mat + rec_mat > 0),
                                2 * prec_mat * rec_mat / (prec_mat + rec_mat), 0.0)
            
            # Macro-F1: mean over present classes in each bootstrap row
            macro_f1 = np.zeros(N_BOOTSTRAP)
            for b in range(N_BOOTSTRAP):
                present = np.unique(yt_mat[b])
                macro_f1[b] = np.mean(f1_mat[b, present])

        # AUC Ambang 1 & 2
        pos1_mat = (yt_mat == 0) | (yt_mat == 1)
        score1_mat = ph_mat + pi_mat
        auc_t1 = calc_auc_fast(pos1_mat, score1_mat)

        pos2_mat = (yt_mat == 0)
        score2_mat = ph_mat
        auc_t2 = calc_auc_fast(pos2_mat, score2_mat)

        metric_arrays = {
            'macro_f1': macro_f1,
            'prec_class0_high': prec_mat[:, 0],
            'rec_class0_high': rec_mat[:, 0],
            'f1_class0_high': f1_mat[:, 0],
            'prec_class1_int': prec_mat[:, 1],
            'rec_class1_int': rec_mat[:, 1],
            'f1_class1_int': f1_mat[:, 1],
            'prec_class2_low': prec_mat[:, 2],
            'rec_class2_low': rec_mat[:, 2],
            'f1_class2_low': f1_mat[:, 2],
            'auc_thresh1': auc_t1,
            'auc_thresh2': auc_t2
        }
        
        print(f"\n--- Model: {model:<10} | Platform: {platform:<8} ---")
        for m_name in metrics_list:
            arr = metric_arrays[m_name]
            med = np.nanmedian(arr)
            lo  = np.nanpercentile(arr, 2.5)
            hi  = np.nanpercentile(arr, 97.5)
            res_str = format_result(med, lo, hi)
            
            print(f"  {m_name:<20}: {res_str}")
            
            results_records.append({
                'model': model,
                'platform': platform,
                'metric': m_name,
                'median': med,
                'ci_lower': lo,
                'ci_upper': hi,
                'result_str': res_str
            })

df_out = pd.DataFrame(results_records)
out_csv = os.path.join(RESULTS_DIR, 'ci_perclass_rev.csv')
df_out.to_csv(out_csv, index=False)
print(f"\nSaved all results to: {out_csv}")
