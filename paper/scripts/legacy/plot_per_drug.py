import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')

from deploy_utils import PreFittedEstimatorWrapper, FittedHolder, FinalEstimatorWrapper

sns.set_theme(style='whitegrid', context='talk')

print("Loading deployed model...")
deployable = joblib.load('./artifacts/deployable_chantest_3class.joblib')
model = deployable['model']
scaler = deployable['scaler']
biomarkers = deployable['biomarkers']

datasets = ['chantest', 'nanion', 'manual']
label_map = {'high': 0, 'intermediate': 1, 'low': 2}
inv_label_map = {0: 'High', 1: 'Int', 2: 'Low'}

for ds in datasets:
    test_file = f"data/{ds}_testing.csv" if ds != "manual" else f"data/{ds}_testing(in).csv"
    df = pd.read_csv(test_file).dropna()
    
    X = df[biomarkers]
    X_scaled = scaler.transform(X)
    
    probs = model.predict_proba(X_scaled)
    df['prob_high'] = probs[:, 0]
    df['prob_int'] = probs[:, 1]
    df['prob_low'] = probs[:, 2]
    
    # Average per drug
    drug_stats = df.groupby('drug_name').agg({
        'label': 'first',
        'prob_high': 'mean',
        'prob_int': 'mean',
        'prob_low': 'mean'
    }).reset_index()
    
    # Map True Class to numeric for sorting
    drug_stats['true_numeric'] = drug_stats['label'].str.lower().map(label_map)
    # Sort by true class (High -> Int -> Low) and then by drug_name alphabetically
    drug_stats = drug_stats.sort_values(['true_numeric', 'drug_name'], ascending=[True, True]).reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.8
    indices = np.arange(len(drug_stats))
    
    p1 = plt.bar(indices, drug_stats['prob_high'], bar_width, color='#d62728', label='P(High)')
    p2 = plt.bar(indices, drug_stats['prob_int'], bar_width, bottom=drug_stats['prob_high'], color='#ff7f0e', label='P(Int)')
    p3 = plt.bar(indices, drug_stats['prob_low'], bar_width, bottom=drug_stats['prob_high'] + drug_stats['prob_int'], color='#1f77b4', label='P(Low)')
    
    plt.xticks(indices, drug_stats['drug_name'], rotation=45, ha='right', fontsize=10)
    plt.ylabel('Mean Predicted Probability', fontsize=12)
    plt.title(f'Per-Drug Mean Predicted Probabilities ({ds.upper()})', fontsize=14, pad=20, fontweight='bold')
    plt.ylim(0, 1.25)
    
    # Add True Class text inside the top of the bar
    for idx, row in drug_stats.iterrows():
        plt.text(idx, 0.98, inv_label_map[row['true_numeric']], ha='center', va='top', fontsize=9, fontweight='bold', color='white', rotation=90)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Highlight specific drugs
    conf_disag_idx = []
    for idx, row in drug_stats.iterrows():
        d_name = row['drug_name']
        if d_name in ['domperidone', 'pimozide']:
            conf_disag_idx.append(idx)
            
        if ds == 'manual' and d_name == 'clarithromycin':
            # Uncertainty
            plt.annotate('Uncertainty\n(0.47 vs 0.52)', xy=(idx, 1.01), 
                         xytext=(idx, 1.15), arrowprops=dict(color='#333333', arrowstyle='->', shrinkB=2),
                         ha='center', va='bottom', fontsize=10, color='blue', annotation_clip=False)

    if conf_disag_idx:
        mid_idx = sum(conf_disag_idx) / len(conf_disag_idx)
        plt.text(mid_idx, 1.15, 'Confident disagreement\nwith label', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
        for idx in conf_disag_idx:
            plt.annotate('', xy=(idx, 1.01), xytext=(mid_idx, 1.14),
                         arrowprops=dict(color='#333333', arrowstyle='->', shrinkB=2, shrinkA=2),
                         annotation_clip=False)

    plt.tight_layout()
    plt.savefig(f'./artifacts/figures/per_drug_probabilities_{ds}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Per-drug probability plots generated.")
