import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set font size globally to 8 pt
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

base_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
csv_path = os.path.join(base_dir, 'results', 'shap_global_importance_rev.csv')

if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Label mapping using matplotlib mathtext
biomarker_map = {
    'qNet': 'qNet',
    'dvdtmax': r'$dV_m/dt_{max}$',
    'vmax': r'$V_{m,peak}$',
    'vrest': r'$V_{m,rest}$',
    'APD50': r'APD$_{50}$',
    'APD90': r'APD$_{90}$',
    'max_dv': r'$dV_m/dt_{repol}$',
    'camax': r'Ca$_{peak}$',
    'carest': r'Ca$_{rest}$',
    'CaTD50': r'CaD$_{50}$',
    'CaTD90': r'CaD$_{90}$'
}

# Sort descending for importance table check
df_desc = df.sort_values(by='mean_abs_shap_aggregated', ascending=False).reset_index(drop=True)

print("=== FINAL AXIS LABELS AND MEAN|SHAP| VALUES (DESCENDING ORDER) ===")
for idx, row in df_desc.iterrows():
    raw_bm = row['biomarker']
    mapped_label = biomarker_map.get(raw_bm, raw_bm)
    val = row['mean_abs_shap_aggregated']
    print(f"{idx+1:2d}. {raw_bm:<10} -> {mapped_label:<20} | mean|SHAP| = {val:.6f}")

top_bm = df_desc.loc[0, 'biomarker']
top_val = df_desc.loc[0, 'mean_abs_shap_aggregated']

if top_bm == 'qNet' and round(top_val, 6) == 0.145534:
    print(f"\nAssertion (top feature is qNet = 0.145534): PASS ({top_bm} = {top_val:.6f})")
else:
    print(f"\nAssertion (top feature is qNet = 0.145534): FAIL ({top_bm} = {top_val:.6f})")
    print("ERROR: nilai tidak cocok dengan CSV")
    sys.exit(1)

# For horizontal bar plot, sort ascending so the highest importance appears on top
plot_df = df_desc.sort_values(by='mean_abs_shap_aggregated', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Manual margins to accommodate y-axis labels without tight_layout/bbox_inches=None clipping
fig.subplots_adjust(left=0.32, right=0.95, top=0.95, bottom=0.18)

y_pos = np.arange(len(plot_df))
ax.barh(y_pos, plot_df['mean_abs_shap_aggregated'], color='gray', edgecolor='black', height=0.7)

ax.set_yticks(y_pos)
mapped_labels = [biomarker_map.get(bm, bm) for bm in plot_df['biomarker']]
ax.set_yticklabels(mapped_labels, fontsize=8)
ax.set_xlabel('mean|SHAP|', fontsize=8)

pdf_path = os.path.join(base_dir, 'results', 'fig_shap_global.pdf')
png_path = os.path.join(base_dir, 'results', 'fig_shap_global.png')

plt.savefig(pdf_path, format='pdf', bbox_inches=None)
plt.savefig(png_path, format='png', dpi=200, bbox_inches=None)

pdf_size_kb = os.path.getsize(pdf_path) / 1024.0
print(f"\nSaved PDF: {pdf_path} ({pdf_size_kb:.2f} KB)")
print(f"Saved PNG: {png_path} ({os.path.getsize(png_path)/1024.0:.2f} KB)")

# Measure PDF MediaBox dimensions
try:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    box = reader.pages[0].mediabox
    w_in = float(box.width) / 72.0
    h_in = float(box.height) / 72.0
    print(f"PDF MediaBox Dimensions: {w_in:.2f} x {h_in:.2f} inches")
except Exception as e:
    with open(pdf_path, 'rb') as f:
        content = f.read(2048).decode('latin1')
        if '/MediaBox' in content:
            idx = content.find('/MediaBox')
            box_str = content[idx:idx+50].split(']')[0]
            parts = box_str.replace('[', ' ').replace(']', ' ').split()
            w_in = float(parts[3]) / 72.0
            h_in = float(parts[4]) / 72.0
            print(f"PDF MediaBox Dimensions (fallback): {w_in:.2f} x {h_in:.2f} inches")

# Check raster image presence in PDF
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

has_raster = b'/Subtype /Image' in pdf_bytes or b'/Subtype/Image' in pdf_bytes
print(f"Raster Image Present in PDF (/Subtype /Image): {has_raster}")
