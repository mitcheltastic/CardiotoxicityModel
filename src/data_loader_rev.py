import os
import pandas as pd
import numpy as np

# CANONICAL 11 BIOMARKER FEATURES (in exact model input order)
# Derived from plus_ultra.ipynb Cell 24 (Line 29), hf_space/app.py (Line 57), conformal_diag.py (Line 14)
CANONICAL_BIOMARKERS = [
    'qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90',
    'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90'
]

# LABEL ENCODING MAPPING
# Derived from plus_ultra.ipynb Cell 30 (Line 72), hf_space/app.py (Line 67), conformal_diag.py (Line 17), context.md (Line 103)
# File & Line Source: plus_ultra.ipynb Cell 30 Line 72: `'class_map': {'high': 0, 'intermediate': 1, 'low': 2}`
LABEL_MAP = {'high': 0, 'intermediate': 1, 'low': 2}

SENTINELS = [
    '#VALUE!', '#DIV/0!', '#N/A', '#REF!', '#NAME?', '#NULL!', '#NUM!', '',
    'NA', 'N/A', 'NaN', 'nan', 'null', 'None', 'inf', '-inf'
]

def load_rev(platform: str, split: str, data_dir: str = None):
    """
    Load and clean _rev dataset in-memory without scaling or disk modification.
    
    CRITICAL METHODOLOGICAL NOTE:
    NO SCALING IS APPLIED IN THIS LOADER.
    The _rev datasets are already control-normalized against control simulations by the data provider.
    Applying StandardScaler or any normalization transform would corrupt pre-scaled features.
    """
    platform_norm = platform.lower().strip()
    split_norm = split.lower().strip()
    
    if platform_norm not in ['chantest', 'nanion', 'manual']:
        raise ValueError(f"ERROR: Invalid platform '{platform}'. Expected 'chantest', 'nanion', or 'manual'.")
        
    if split_norm in ['train', 'training']:
        split_name = 'training'
    elif split_norm in ['test', 'testing']:
        split_name = 'testing'
    else:
        raise ValueError(f"ERROR: Invalid split '{split}'. Expected 'training' or 'testing'.")
        
    fname = f"{platform_norm}_{split_name}_scaled_rev.csv"
    
    if data_dir is None:
        # Default to TrainAI/data directory relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"ERROR: Data file not found at '{fpath}'")
        
    # Read with dtype=str and keep_default_na=False to prevent silent pandas coercion
    df_raw = pd.read_csv(fpath, dtype=str, keep_default_na=False)
    rows_before = len(df_raw)
    
    # Identify label column
    label_col = 'label' if 'label' in df_raw.columns else 'Class'
    if label_col not in df_raw.columns:
        raise KeyError(f"ERROR: Label column missing in {fname}")
        
    if 'drug_name' not in df_raw.columns:
        raise KeyError(f"ERROR: 'drug_name' column missing in {fname}")
        
    # Find offending rows containing any sentinel in the 11 biomarker columns
    dropped_logs = []
    dropped_row_indices = []
    
    for r_idx, row in df_raw.iterrows():
        offending = [bm for bm in CANONICAL_BIOMARKERS if row[bm] in SENTINELS]
        if offending:
            dropped_row_indices.append(r_idx)
            dropped_logs.append({
                'platform': platform_norm,
                'split': split_name,
                'row_index': r_idx,
                'Sample_ID': row.get('Sample_ID', 'N/A'),
                'drug_name': row['drug_name'],
                'label': row[label_col],
                'offending_columns': ";".join(offending)
            })
            
    # Append to logs/DROPPED_ROWS_REV.csv audit trail
    if dropped_logs:
        log_dir = os.path.join(os.path.dirname(data_dir), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'DROPPED_ROWS_REV.csv')
        df_dropped = pd.DataFrame(dropped_logs)
        
        file_exists = os.path.exists(log_file)
        df_dropped.to_csv(log_file, mode='a', header=not file_exists, index=False)
        
    # Drop offending rows in-memory
    df_clean = df_raw.drop(index=dropped_row_indices).copy()
    rows_dropped = len(dropped_row_indices)
    rows_after = len(df_clean)
    
    # Extract groups and raw labels
    groups = df_clean['drug_name'].reset_index(drop=True)
    raw_labels = df_clean[label_col].str.lower().str.strip()
    
    # Encode labels
    y = raw_labels.map(LABEL_MAP).reset_index(drop=True)
    
    # Prepare feature matrix X (convert to float64 explicitly)
    X = df_clean[CANONICAL_BIOMARKERS].astype(np.float64).reset_index(drop=True)
    
    # Build load-time diagnostic report
    drug_counts = groups.value_counts().to_dict()
    sample_ids = df_clean['Sample_ID'].tolist() if 'Sample_ID' in df_clean.columns else []
    
    report = {
        'platform': platform_norm,
        'split': split_name,
        'fname': fname,
        'rows_before': rows_before,
        'rows_dropped': rows_dropped,
        'rows_after': rows_after,
        'drug_counts': drug_counts,
        'min_rows_per_drug': min(drug_counts.values()) if drug_counts else 0,
        'median_rows_per_drug': float(np.median(list(drug_counts.values()))) if drug_counts else 0.0,
        'max_rows_per_drug': max(drug_counts.values()) if drug_counts else 0,
        'sample_ids': sample_ids,
        'dropped_records': dropped_logs
    }
    
    # ASSERTION GUARDS (Raise immediately on failure)
    expected_drug_count = 12 if split_name == 'training' else 16
    unique_drugs = groups.nunique()
    if unique_drugs != expected_drug_count:
        raise AssertionError(f"ERROR: Drug count is {unique_drugs}, expected {expected_drug_count} for {split_name}")
        
    if list(X.columns) != CANONICAL_BIOMARKERS:
        raise AssertionError(f"ERROR: X columns {list(X.columns)} do not match canonical order {CANONICAL_BIOMARKERS}")
        
    if X.shape[1] != 11:
        raise AssertionError(f"ERROR: X has {X.shape[1]} columns, expected 11")
        
    for col in X.columns:
        if X[col].dtype != np.float64:
            raise AssertionError(f"ERROR: Column '{col}' has dtype {X[col].dtype}, expected float64")
            
    if X.isna().sum().sum() > 0:
        raise AssertionError(f"ERROR: X contains {X.isna().sum().sum()} NaN values after cleaning")
        
    if np.isinf(X.values).sum() > 0:
        raise AssertionError(f"ERROR: X contains Inf values after cleaning")
        
    if y.isna().sum() > 0:
        unmapped = raw_labels[y.isna()].unique()
        raise AssertionError(f"ERROR: Unmapped labels found: {unmapped}")
        
    if len(X) != len(y) or len(y) != len(groups):
        raise AssertionError(f"ERROR: Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(groups)={len(groups)}")
        
    # Check each drug carries exactly one label
    drug_label_counts = df_clean.groupby('drug_name')[label_col].nunique()
    mixed_drugs = drug_label_counts[drug_label_counts > 1]
    if not mixed_drugs.empty:
        raise AssertionError(f"ERROR: Drug(s) carry mixed labels: {mixed_drugs.to_dict()}")
        
    # Check minimum rows per drug >= 1900
    if report['min_rows_per_drug'] < 1900:
        raise AssertionError(f"ERROR: Drug with fewer than 1900 rows detected: min={report['min_rows_per_drug']}")
        
    return X, y, groups, report
