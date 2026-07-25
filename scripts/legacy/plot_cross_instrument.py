import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style='whitegrid', context='talk')

datasets = ['Chantest', 'Nanion', 'Manual']
f1_scores = [0.878, 0.819, 0.756]
critical_misses = [0, 0, 1]

# Distinguish by color
colors = ['#1f77b4', '#ff7f0e', '#ff7f0e'] # Blue for in-dist, orange for domain-shifted

plt.figure(figsize=(9, 6))

indices = np.arange(len(datasets))
bars = plt.bar(indices, f1_scores, color=colors, edgecolor='#111111', width=0.6)

# Custom legend for the distinction
import matplotlib.patches as mpatches
in_dist_patch = mpatches.Patch(color='#1f77b4', label='In-Distribution (Calibration/Training Source)')
out_dist_patch = mpatches.Patch(color='#ff7f0e', label='Unseen Instrument (Domain-Shifted)')
plt.legend(handles=[in_dist_patch, out_dist_patch], loc='lower left', fontsize=11)

plt.ylabel('Drug-Level Macro-F1', fontsize=12)
plt.title('Cross-Instrument Generalization (Drug Level)\nchantest-trained model applied unchanged', fontsize=14, pad=20, fontweight='bold')
plt.ylim(0, 1.0)
plt.xticks(indices, datasets, fontsize=12)

# Annotations
for i, bar in enumerate(bars):
    height = bar.get_height()
    # F1 score text
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, 
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    # Critical misses text inside the bar
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2,
             f'Critical\nMisses:\n{critical_misses[i]}', ha='center', va='center', 
             color='white' if i == 0 else '#111111', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./artifacts/figures/cross_instrument_macrof1.png', dpi=300, bbox_inches='tight')
plt.close()
