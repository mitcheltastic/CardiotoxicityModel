import os
import sys
import hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set TrueType fonts and global font size to 8 pt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

base_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
csv_path = os.path.join(base_dir, 'results', 'shap_global_xgb_rev.csv')

if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at {csv_path}")
    sys.exit(1)

# Assert SHA-256 of input CSV
with open(csv_path, 'rb') as f:
    csv_sha = hashlib.sha256(f.read()).hexdigest().upper()

EXPECTED_SHA = '3CF4A7A301C00CE088857C4BF4507D36D103A557C618B44AE7EA9AEE26D84142'
if csv_sha != EXPECTED_SHA:
    print(f"ERROR: SHA-256 mismatch for {csv_path}")
    print(f"  Got:      {csv_sha}")
    print(f"  Expected: {EXPECTED_SHA}")
    sys.exit(1)
else:
    print(f"Input CSV SHA-256 assertion PASSED: {csv_sha}")

df = pd.read_csv(csv_path)

# Label mapping using matplotlib mathtext
biomarker_map = {
    'qNet': 'qNet',
    'vrest': r'$V_{m,rest}$',
    'APD50': r'APD$_{50}$',
    'APD90': r'APD$_{90}$',
    'max_dv': r'$dV_m/dt_{repol}$',
    'camax': r'Ca$_{peak}$',
    'vmax': r'$V_{m,peak}$',
    'dvdtmax': r'$dV_m/dt_{max}$',
    'CaTD90': r'CaD$_{90}$',
    'CaTD50': r'CaD$_{50}$',
    'carest': r'Ca$_{rest}$'
}

# Sort descending for importance table check (top to bottom ranking)
df_desc = df.sort_values(by='mean_abs_shap_aggregated', ascending=False).reset_index(drop=True)

print("\n=== RANKED BIOMARKER LIST (TOP TO BOTTOM) ===")
for idx, row in df_desc.iterrows():
    raw_bm = row['biomarker']
    mapped_label = biomarker_map.get(raw_bm, raw_bm)
    val = row['mean_abs_shap_aggregated']
    print(f"{idx+1:2d}. {raw_bm:<10} -> {mapped_label:<20} | mean|SHAP| = {val:.6f}")

top_bm = df_desc.loc[0, 'biomarker']
top_val = df_desc.loc[0, 'mean_abs_shap_aggregated']

if top_bm == 'qNet' and abs(top_val - 2.313963) < 1e-4:
    print(f"\nAssertion (top feature is qNet ~ 2.314): PASS ({top_bm} = {top_val:.6f})")
else:
    print(f"\nAssertion (top feature is qNet ~ 2.314): FAIL ({top_bm} = {top_val:.6f})")
    sys.exit(1)

# Color specification: Steel/Medium Blue fill with Dark Blue edge
BAR_FILL_HEX = '#3B6EA5'
BAR_EDGE_HEX = '#1F4E79'

print(f"\nApplying Blue Color Scheme: Fill = '{BAR_FILL_HEX}', Edge = '{BAR_EDGE_HEX}'")

# For horizontal bar plot, sort ascending so the highest importance appears on top
plot_df = df_desc.sort_values(by='mean_abs_shap_aggregated', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Manual margins to accommodate y-axis labels
fig.subplots_adjust(left=0.32, right=0.95, top=0.95, bottom=0.18)

y_pos = np.arange(len(plot_df))
ax.barh(y_pos, plot_df['mean_abs_shap_aggregated'], color=BAR_FILL_HEX, edgecolor=BAR_EDGE_HEX, height=0.7)

ax.set_yticks(y_pos)
mapped_labels = [biomarker_map.get(bm, bm) for bm in plot_df['biomarker']]
ax.set_yticklabels(mapped_labels, fontsize=8)
ax.set_xlabel('mean|SHAP| (log-odds)', fontsize=8)

pdf_path = os.path.join(base_dir, 'results', 'fig_shap_xgb.pdf')
png_path = os.path.join(base_dir, 'results', 'fig_shap_xgb.png')

plt.savefig(pdf_path, format='pdf', bbox_inches=None)
plt.savefig(png_path, format='png', dpi=200, bbox_inches=None)

with open(pdf_path, 'rb') as f:
    pdf_sha = hashlib.sha256(f.read()).hexdigest().upper()

with open(png_path, 'rb') as f:
    png_sha = hashlib.sha256(f.read()).hexdigest().upper()

pdf_size_kb = os.path.getsize(pdf_path) / 1024.0
print(f"\nSaved Vector PDF: {pdf_path} ({pdf_size_kb:.2f} KB) | SHA-256: {pdf_sha}")
print(f"Saved PNG Preview: {png_path} ({os.path.getsize(png_path)/1024.0:.2f} KB) | SHA-256: {png_sha}")

# Check raster image presence in PDF
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

has_raster = b'/Subtype /Image' in pdf_bytes or b'/Subtype/Image' in pdf_bytes
print(f"Raster Image Present in PDF (/Subtype /Image): {has_raster}")
