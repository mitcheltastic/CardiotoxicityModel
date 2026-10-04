import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if TRAINAI_DIR not in sys.path:
    sys.path.insert(0, TRAINAI_DIR)

from src.data_loader_rev import load_rev, CANONICAL_BIOMARKERS, LABEL_MAP

SEVERITY = {0: 2, 1: 1, 2: 0} # 0=High (Sev 2), 1=Int (Sev 1), 2=Low (Sev 0)
INV_LABEL_MAP = {0: 'high', 1: 'intermediate', 2: 'low'}

def evaluate_predictions(y_true, y_pred, prob_matrix=None):
    """
    Computes macro-F1, accuracy, per-class P/R/F1, and confusion matrix.
    Labels are 0: High, 1: Intermediate, 2: Low.
    """
    labels = [0, 1, 2]
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'macro_f1': macro_f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

def get_drug_level_summary(df_row_probs):
    """
    Aggregates row-level probabilities to drug-level via mean probability followed by argmax.
    df_row_probs must have columns: drug_name, true_label, prob_high, prob_int, prob_low.
    """
    drug_summary = df_row_probs.groupby('drug_name').agg({
        'true_label': 'first',
        'prob_high': 'mean',
        'prob_int': 'mean',
        'prob_low': 'mean'
    }).reset_index()
    
    drug_summary['pred_label'] = np.argmax(
        drug_summary[['prob_high', 'prob_int', 'prob_low']].values, axis=1
    )
    return drug_summary

def classify_misclassifications(drug_summary):
    """
    Identifies misclassified drugs and labels them as under-triage or over-call.
    """
    records = []
    for _, row in drug_summary.iterrows():
        t = int(row['true_label'])
        p = int(row['pred_label'])
        if t != p:
            sev_t = SEVERITY[t]
            sev_p = SEVERITY[p]
            category = "under-triage" if sev_p < sev_t else "over-call"
            records.append({
                'drug_name': row['drug_name'],
                'true_tier': INV_LABEL_MAP[t],
                'pred_tier': INV_LABEL_MAP[p],
                'category': category,
                'prob_high': row['prob_high'],
                'prob_int': row['prob_int'],
                'prob_low': row['prob_low']
            })
    return pd.DataFrame(records)

def calc_ordinal_aucs_drug(drug_summary):
    """
    Calculates ordinal binary AUCs at drug level:
    Threshold 1: P(risk > Low) = p_high + p_int, true binary = 1 if High or Int else 0
    Threshold 2: P(risk > Intermediate) = p_high, true binary = 1 if High else 0
    """
    y_true = drug_summary['true_label'].values
    p_high = drug_summary['prob_high'].values
    p_int  = drug_summary['prob_int'].values
    
    # Threshold 1
    s1 = p_high + p_int
    y1 = np.where((y_true == 0) | (y_true == 1), 1, 0)
    auc1 = roc_auc_score(y1, s1) if len(np.unique(y1)) > 1 else np.nan
    
    # Threshold 2
    s2 = p_high
    y2 = np.where(y_true == 0, 1, 0)
    auc2 = roc_auc_score(y2, s2) if len(np.unique(y2)) > 1 else np.nan
    
    return auc1, auc2

def calc_ordinal_aucs_row(y_row, prob_row):
    """
    Calculates ordinal binary AUCs at row level.
    prob_row shape (N, 3): col 0=high, col 1=int, col 2=low.
    """
    p_high = prob_row[:, 0]
    p_int  = prob_row[:, 1]
    
    s1 = p_high + p_int
    y1 = np.where((y_row == 0) | (y_row == 1), 1, 0)
    auc1 = roc_auc_score(y1, s1) if len(np.unique(y1)) > 1 else np.nan
    
    s2 = p_high
    y2 = np.where(y_row == 0, 1, 0)
    auc2 = roc_auc_score(y2, s2) if len(np.unique(y2)) > 1 else np.nan
    
    return auc1, auc2

def run_drug_bootstrap(drug_summary, n_iter=10000, seed=42):
    """
    Runs 10,000-iteration drug-level bootstrap for macro-F1 (STEP9 protocol).
    Returns median, 2.5%, 97.5% percentiles.
    """
    np.random.seed(seed)
    n_drugs = len(drug_summary)
    f1_list = []
    
    y_t_all = drug_summary['true_label'].values
    y_p_all = drug_summary['pred_label'].values
    
    for _ in range(n_iter):
        idx = np.random.choice(n_drugs, size=n_drugs, replace=True)
        y_t = y_t_all[idx]
        y_p = y_p_all[idx]
        present = np.unique(y_t)
        f1 = f1_score(y_t, y_p, labels=present, average='macro', zero_division=0)
        f1_list.append(f1)
        
    f1_arr = np.array(f1_list)
    return {
        'median': np.median(f1_arr),
        'ci_low': np.percentile(f1_arr, 2.5),
        'ci_high': np.percentile(f1_arr, 97.5),
        'distributions': f1_arr
    }

def calc_auc_fast(pos_mat, score_mat):
    """
    Vectorized AUC calculation matching scripts/reporting/run_step22_cipa_bootstrap.py
    """
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

def run_cipa_bootstrap(groups, y, row_probs, n_bootstrap=10000, seed=42):
    """
    Runs the exact CiPA replicate bootstrap protocol from STEP22.
    Samples 1 replicate row per drug across 10,000 iterations.
    Computes One-vs-Rest AUC per class with 95% CI.
    """
    unique_drugs = groups.unique()
    drug_row_indices = {d: np.where(groups.values == d)[0] for d in unique_drugs}
    
    rng = np.random.RandomState(seed)
    n_drugs = len(unique_drugs)
    boot_indices = np.zeros((n_bootstrap, n_drugs), dtype=int)
    for b in range(n_bootstrap):
        for i, d in enumerate(unique_drugs):
            boot_indices[b, i] = rng.choice(drug_row_indices[d])
            
    y_16 = np.array([y.values[drug_row_indices[d][0]] for d in unique_drugs])
    ytrue_10k = np.tile(y_16, (n_bootstrap, 1))
    
    probs_10k = row_probs[boot_indices] # (10000, 16, 3)
    
    class_names = ['High', 'Int', 'Low']
    auc_results = {}
    
    for c in range(3):
        pos_c = (ytrue_10k == c)
        score_c = probs_10k[:, :, c]
        auc_10k = calc_auc_fast(pos_c, score_c)
        
        valid_vals = auc_10k[np.isfinite(auc_10k)]
        med = np.median(valid_vals)
        ci_lo = np.percentile(valid_vals, 2.5)
        ci_hi = np.percentile(valid_vals, 97.5)
        auc_results[class_names[c]] = {
            'median': med,
            'ci_low': ci_lo,
            'ci_high': ci_hi,
            'str': f"{med:.3f} ({ci_lo:.3f}-{ci_hi:.3f})"
        }
    return auc_results
