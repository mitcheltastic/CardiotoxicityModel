import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAINAI_DIR = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'

def generate_fixed_shap_figure():
    print("==================================================================")
    print("=== TASK 8: GENERATE CORRECTED SHAP GLOBAL IMPORTANCE FIGURE ====")
    print("==================================================================")
    
    csv_path = os.path.join(TRAINAI_DIR, 'results', 'shap_global_importance_rev.csv')
    df = pd.read_csv(csv_path)
    print(f"Loaded SHAP table from {csv_path} with {len(df)} rows.")
    
    # Mathematical axis label mapping
    biomarker_label_map = {
        'qNet': 'qNet',
        'vrest': r'$V_{m,rest}$',
        'max_dv': r'$dV_m/dt_{repol}$',
        'APD50': r'APD$_{50}$',
        'APD90': r'APD$_{90}$',
        'vmax': r'$V_{m,peak}$',
        'dvdtmax': r'$dV_m/dt_{max}$',
        'camax': r'Ca$_{peak}$',
        'CaTD50': r'CaD$_{50}$',
        'CaTD90': r'CaD$_{90}$',
        'carest': r'Ca$_{rest}$'
    }
    
    # Sort ascending for horizontal bar chart (so highest importance is at the top)
    plot_df = df.sort_values(by='mean_abs_shap_aggregated', ascending=True).reset_index(drop=True)
    
    # Style settings
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif']
    })
    
    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=300)
    
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df['mean_abs_shap_aggregated'], color='#3B6EA5', edgecolor='#1F4E79', height=0.68)
    
    ax.set_yticks(y_pos)
    labels = [biomarker_label_map.get(bm, bm) for bm in plot_df['biomarker']]
    ax.set_yticklabels(labels)
    
    ax.set_xlabel(r'Global Mean Absolute SHAP Value ($\mathrm{mean}|\mathrm{SHAP}|$)')
    ax.set_title('Global Feature Importance (Stacking Classifier, ChanTest Test Split)', fontsize=11, fontweight='bold', pad=12)
    
    # Add explicit value labels at the end of each bar
    max_val = plot_df['mean_abs_shap_aggregated'].max()
    ax.set_xlim(0, max_val * 1.18)
    
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        ax.text(w + max_val * 0.015, y, f'{w:.4f}', va='center', ha='left', fontsize=8.5, color='#222222')
        
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    out_dir = os.path.join(TRAINAI_DIR, 'results', 'cd4_eval')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'fig_shap_global_fixed.png')
    out_pdf = os.path.join(out_dir, 'fig_shap_global_fixed.pdf')
    
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved corrected figure to:")
    print(f"  PNG: {out_png}")
    print(f"  PDF: {out_pdf}")

if __name__ == '__main__':
    generate_fixed_shap_figure()
