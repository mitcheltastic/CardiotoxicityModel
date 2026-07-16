import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')

from deploy_utils import PreFittedEstimatorWrapper, FittedHolder, FinalEstimatorWrapper

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#111111',
    'axes.labelcolor': '#111111',
    'axes.titlecolor': '#111111',
    'figure.titlesize': 18,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

os.makedirs('./artifacts/figures', exist_ok=True)

deployable = joblib.load('./artifacts/deployable_chantest_3class.joblib')
reloaded_model = deployable['model']

bootstrap = joblib.load('./artifacts/phase4_bootstrap.joblib')
sweep_results = pd.read_csv('./artifacts/phase3_protocolA_results.csv')

print('Loading dataset...')
df_test = pd.read_csv('data/chantest_testing.csv').dropna()
biomarkers = deployable['biomarkers']
X_test = deployable['scaler'].transform(df_test[biomarkers])
label_map = {'high': 0, 'intermediate': 1, 'low': 2}
y_test = df_test['label'].str.lower().map(label_map).values

print('Generating Figure 1: Confusion Matrix...')
preds = reloaded_model.predict(X_test)
cm = confusion_matrix(y_test, preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels = ['high', 'intermediate', 'low']
plt.figure(figsize=(7, 6))
sns.heatmap(cm_normalized, annot=False, cmap='Blues', cbar=True,
            xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        plt.text(j + 0.5, i + 0.5, f'{cm[i, j]:,}\n({cm_normalized[i, j]*100:.1f}%)',
                 horizontalalignment='center', verticalalignment='center',
                 color=color, fontsize=12, weight='bold')

plt.ylabel('True Risk Category', fontsize=12, fontweight='bold', labelpad=10)
plt.xlabel('Predicted Risk Category', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Sample-Level Confusion Matrix: Deployable Stacking Model on Test Split\n(Chantest 3-Class)', 
          fontsize=12, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('./artifacts/figures/confusion_matrix_chantest.png', dpi=300, bbox_inches='tight')
plt.close()


print('Generating Figure 2: Bootstrap Macro-F1 Distributions...')
plt.figure(figsize=(9, 5))
colors = {'chantest': '#1f77b4', 'nanion': '#ff7f0e', 'manual': '#2ca02c'}

for ds in ['chantest', 'nanion', 'manual']:
    f1_arr = bootstrap[ds]['f1']
    median = np.median(f1_arr)
    ci_low = np.percentile(f1_arr, 2.5)
    ci_high = np.percentile(f1_arr, 97.5)
    
    sns.kdeplot(f1_arr, label=f'{ds} (median={median:.3f})', color=colors[ds], fill=True, alpha=0.25, linewidth=2)
    plt.axvline(median, color=colors[ds], linestyle='-', linewidth=1.5)
    plt.axvline(ci_low, color=colors[ds], linestyle='--', linewidth=1)
    plt.axvline(ci_high, color=colors[ds], linestyle='--', linewidth=1)

plt.xlabel('Bootstrap Test Macro-F1', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Bootstrap Test Macro-F1 Distributions (drug-level resampling, 1 row/drug/iteration)\nSolid lines = Median, Dashed lines = 95% CI', fontsize=12, pad=15, fontweight='bold')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./artifacts/figures/bootstrap_macrof1_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print('Generating Figure 3: Per-class AUC Comparison...')
datasets_list = ['chantest', 'nanion', 'manual']
classes = ['high', 'intermediate', 'low']
x_indices = np.arange(len(datasets_list))
bar_width = 0.25

plt.figure(figsize=(9, 6))

for c_idx, c_name in enumerate(classes):
    medians = []
    yerr_low = []
    yerr_high = []
    
    for ds in datasets_list:
        auc_arr = bootstrap[ds][f'auc_{c_name}']
        med = np.median(auc_arr)
        low = np.percentile(auc_arr, 2.5)
        high = np.percentile(auc_arr, 97.5)
        
        medians.append(med)
        yerr_low.append(med - low)
        yerr_high.append(high - med)
        
    yerr = np.array([yerr_low, yerr_high])
    plt.bar(x_indices + c_idx * bar_width, medians, bar_width, 
            yerr=yerr, capsize=4, label=f'{c_name.upper()} Class', 
            error_kw=dict(lw=1.2, capthick=1.2, ecolor='#333333'))

plt.xticks(x_indices + bar_width, [d.capitalize() for d in datasets_list], fontsize=11)
plt.ylabel('One-vs-Rest Area Under ROC (AUC)', fontsize=12)
plt.xlabel('Dataset Split', fontsize=12)
plt.title('Per-Class AUC Comparison with 95% Bootstrap Confidence Intervals\n(drug-level resampling, 1 row/drug/iteration)', fontsize=12, pad=15, fontweight='bold')
plt.ylim(0.5, 1.05)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('./artifacts/figures/per_class_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print('Generating Figure 5: Ensemble Family & Paradigm Comparison...')
sweep_c3 = sweep_results[(sweep_results['dataset'] == 'chantest') & (sweep_results['scenario'] == '3class')]
best_stacking = sweep_c3[sweep_c3['ensemble_family'] == 'stacking']['test_macro_f1'].max()
best_voting = sweep_c3[sweep_c3['ensemble_family'] == 'voting']['test_macro_f1'].max()
best_ml = sweep_c3[sweep_c3['paradigm'] == 'ML']['test_macro_f1'].max()
best_dl = sweep_c3[sweep_c3['paradigm'] == 'DL']['test_macro_f1'].max()
best_mixed = sweep_c3[sweep_c3['paradigm'] == 'mixed']['test_macro_f1'].max()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.bar(['Stacking', 'Voting'], [best_stacking, best_voting], color=['#1f77b4', '#aec7e8'], edgecolor='#111111', width=0.5)
ax1.set_ylabel('Best Test Macro-F1', fontsize=11)
ax1.set_ylim(0.7, 0.85)
ax1.set_title('Stacking vs. Voting Ensembles', fontsize=11, fontweight='bold')
ax2.bar(['ML', 'DL (ANNs)', 'Mixed'], [best_ml, best_dl, best_mixed], color=['#2ca02c', '#d62728', '#9467bd'], edgecolor='#111111', width=0.6)
ax2.set_ylabel('Best Test Macro-F1', fontsize=11)
ax2.set_ylim(0.2, 0.85)
ax2.set_title('Pure-ML vs. Pure-DL vs. Mixed', fontsize=11, fontweight='bold')
plt.suptitle('Sample-Level Combinatorial Ensemble Sweep Comparison (Chantest 3-Class)', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('./artifacts/figures/ensemble_family_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print('Relabeling complete.')
