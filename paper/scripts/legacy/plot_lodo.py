import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import subprocess
import os

sns.set_theme(style='whitegrid', context='talk')

print("Running conformal_diag.py to get LODO scores...")
result = subprocess.run([sys.executable, 'conformal_diag.py'], capture_output=True, text=True)
lines = result.stdout.split('\n')

scores = []
drugs = []
parsing = False
for line in lines:
    if line.startswith("Drug Name"):
        parsing = True
        continue
    if parsing:
        if line.startswith("--- Step 2"):
            break
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 6 and not parts[0].startswith('-'):
            drug = parts[0]
            score = float(parts[5])
            drugs.append(drug)
            scores.append(score)

# Sort by score
sorted_indices = np.argsort(scores)
sorted_scores = np.array(scores)[sorted_indices]
sorted_drugs = np.array(drugs)[sorted_indices]

plt.figure(figsize=(10, 5))
indices = np.arange(len(sorted_scores))
plt.bar(indices, sorted_scores, color='#1f77b4', edgecolor='#111111')
plt.xticks(indices, sorted_drugs, rotation=45, ha='right', fontsize=11)
plt.ylabel('Nonconformity Score $s$', fontsize=12)
plt.title('LODO Calibration Nonconformity Scores (12 Training Drugs)', fontsize=14, pad=20, fontweight='bold')

# Alpha lines
plt.axhline(0.6275, color='#d62728', linestyle='--', linewidth=2, label=r'$\hat{q} = 0.6275 \ (\alpha = 0.154)$')
plt.axhline(0.9898, color='#ff7f0e', linestyle='-.', linewidth=2, label=r'$\hat{q} = 0.9898 \ (\alpha = 0.077)$')

# Annotate diltiazem
dilt_idx = np.where(sorted_drugs == 'diltiazem')[0][0]
dilt_score = sorted_scores[dilt_idx]
plt.annotate('diltiazem\n(Outlier setting $\\alpha=0.077$ limit)', 
             xy=(dilt_idx, dilt_score), xytext=(dilt_idx - 2, dilt_score - 0.2),
             arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', fontsize=10)

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('./artifacts/figures/lodo_calibration_scores.png', dpi=300, bbox_inches='tight')
plt.close()

print("LODO calibration scores plotted.")
