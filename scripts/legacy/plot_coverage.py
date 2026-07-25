import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

sns.set_theme(style='whitegrid', context='talk')

datasets = ['Chantest\n(In-Distribution)', 'Nanion\n(Domain-Shifted)', 'Manual\n(Domain-Shifted)']
coverages = [0.875, 0.8125, 0.8125]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(8, 5))
indices = np.arange(len(datasets))
plt.bar(indices, coverages, color=colors, edgecolor='#111111', width=0.6)

plt.axhline(0.846, color='#d62728', linestyle='--', linewidth=2, label=r'Nominal Coverage ($1 - \alpha = 0.846$)')

plt.xticks(indices, datasets, fontsize=12)
plt.ylabel('Empirical Coverage', fontsize=12)
plt.title(r'Empirical Coverage vs. Nominal ($\alpha=0.154$)', fontsize=14, pad=20, fontweight='bold')
plt.ylim(0, 1.05)

for i, v in enumerate(coverages):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=11)

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('./artifacts/figures/empirical_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

print("Empirical coverage plotted.")
