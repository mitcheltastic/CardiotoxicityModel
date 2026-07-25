import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev

output_lines = []
def log(msg=""):
    output_lines.append(str(msg))
    print(msg)

SEED = 42

log("==========================================")
log("=== STEP 0 — HARD RULES AND SETUP ===")
log("==========================================")

scaler_modules = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'QuantileTransformer']
for sm in scaler_modules:
    if sm in globals() or sm in locals():
        log(f"ERROR: Scaler {sm} detected in script environment!")
        sys.exit(1)

model_path = os.path.join(trainai_dir, 'models', 'chantest_stacking_model_rev.joblib')
with open(model_path, 'rb') as f:
    h = hashlib.sha256(f.read()).hexdigest().upper()
expected_hash = "A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2"
if h == expected_hash:
    log("models/chantest_stacking_model_rev.joblib hash matches: PASS")
else:
    log(f"models/chantest_stacking_model_rev.joblib hash fails: FAIL. Expected {expected_hash}, got {h}")
    sys.exit(1)

stack_model = joblib.load(model_path)

meta_path = os.path.join(trainai_dir, 'models', 'chantest_stacking_metadata_rev.json')
with open(meta_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

canonical_biomarkers = metadata['canonical_biomarkers']
log(f"Canonical feature order loaded: {canonical_biomarkers}")


log("\n==========================================")
log("=== STEP 1 — DECLARE THE EXPLANATION TARGET ===")
log("==========================================")
log("Approach: StackingClassifier is not natively supported by TreeExplainer since it's an ensemble of base models (RF, XGB) feeding a meta-learner. Explaining the meta-learner alone would attribute importance to base-model probabilities rather than the 11 biomarkers, missing the biological link.")
log("Instead, we treat the FULL fitted stack's predict_proba as a black-box function and apply a model-agnostic explainer (KernelExplainer). This guarantees importance is correctly attributed to the input 11 biomarkers.")
log(f"SHAP Version: {shap.__version__}")
log("Explainer Class: shap.KernelExplainer")


log("\n==========================================")
log("=== STEP 2 — DATA FOR EXPLANATION ===")
log("==========================================")
X_train_full, y_train_full, groups_train_full, _ = load_rev('chantest', 'training', data_dir=os.path.join(trainai_dir, 'data'))
X_test_full, y_test_full, groups_test_full, _ = load_rev('chantest', 'testing', data_dir=os.path.join(trainai_dir, 'data'))

# Assert feature order matches exactly
if list(X_train_full.columns) != canonical_biomarkers:
    log("ERROR: Feature order does not match canonical order!")
    sys.exit(1)
if list(X_test_full.columns) != canonical_biomarkers:
    log("ERROR: Feature order does not match canonical order!")
    sys.exit(1)
log("Feature order verified against canonical order: PASS")

background_data = shap.sample(X_train_full, 100, random_state=SEED)
log(f"Background: A seed={SEED} random subsample of chantest TRAINING rows. Size: {len(background_data)}")

# Subsample test set for runtime, stratified by drug to ensure coverage
_, X_test_sub, _, groups_test_sub = train_test_split(
    X_test_full, groups_test_full, test_size=320, random_state=SEED, stratify=groups_test_full
)

log(f"Explained on: A seed={SEED} stratified subsample of chantest TESTING rows. Size: {len(X_test_sub)}")
drug_counts = groups_test_sub.value_counts().to_dict()
log(f"Per-drug row counts in the explained subsample:")
for d, c in sorted(drug_counts.items()):
    log(f"  {d}: {c}")

if len(drug_counts) != 16:
    missing = set(groups_test_full.unique()) - set(drug_counts.keys())
    log(f"ERROR: Subsample does not cover all 16 test drugs. Missing: {missing}")
    sys.exit(1)
log("Confirmed: The explained subsample covers all 16 test drugs.")


log("\n==========================================")
log("=== STEP 3 — GLOBAL IMPORTANCE ===")
log("==========================================")
log("Computing SHAP values using KernelExplainer (this may take a few minutes)...")
# Predict function that returns probability arrays
explainer = shap.KernelExplainer(stack_model.predict_proba, background_data)
# X_test_sub is 320 rows.
# Suppress tqdm if possible
shap_values = explainer.shap_values(X_test_sub, silent=True)

# shap_values from KernelExplainer might be a list or a 3D array
if isinstance(shap_values, list):
    mean_abs_shap_high = np.abs(shap_values[0]).mean(axis=0)
    mean_abs_shap_int = np.abs(shap_values[1]).mean(axis=0)
    mean_abs_shap_low = np.abs(shap_values[2]).mean(axis=0)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_abs_shap_high = mean_abs_shap[:, 0]
    mean_abs_shap_int = mean_abs_shap[:, 1]
    mean_abs_shap_low = mean_abs_shap[:, 2]

agg_rule = "Mean of mean|SHAP| across the three classes."
log(f"Aggregation rule: {agg_rule}")
mean_abs_shap_agg = (mean_abs_shap_high + mean_abs_shap_int + mean_abs_shap_low) / 3.0

global_importance = []
for i, bm in enumerate(canonical_biomarkers):
    global_importance.append({
        'biomarker': bm,
        'mean_abs_shap_high': mean_abs_shap_high[i],
        'mean_abs_shap_int': mean_abs_shap_int[i],
        'mean_abs_shap_low': mean_abs_shap_low[i],
        'mean_abs_shap_aggregated': mean_abs_shap_agg[i]
    })

gi_df = pd.DataFrame(global_importance)
gi_df = gi_df.sort_values(by='mean_abs_shap_aggregated', ascending=False).reset_index(drop=True)
gi_df['rank'] = gi_df.index + 1

log(f"\n{'Rank':<5} | {'Biomarker':<15} | {'High':<10} | {'Int':<10} | {'Low':<10} | Aggregated (mean|SHAP|)")
log("-" * 80)
qnet_row = None
for _, row in gi_df.iterrows():
    if row['biomarker'] == 'qNet':
        qnet_row = row
    log(f"{row['rank']:<5} | {row['biomarker']:<15} | {row['mean_abs_shap_high']:<10.6f} | {row['mean_abs_shap_int']:<10.6f} | {row['mean_abs_shap_low']:<10.6f} | {row['mean_abs_shap_aggregated']:.6f}")

log(f"\nqNet exactly: Rank {qnet_row['rank']}, mean|SHAP| = {qnet_row['mean_abs_shap_aggregated']:.6f}")


log("\n==========================================")
log("=== STEP 4 — DEGENERATE FEATURE CHECK ===")
log("==========================================")
log(f"{'Biomarker':<15} | {'Rank':<5} | {'mean|SHAP|':<10} | {'Std':<10} | CV")
log("-" * 65)

std_vals = []
cv_vals = []
for _, row in gi_df.iterrows():
    bm = row['biomarker']
    col_data = X_test_sub[bm]
    std = col_data.std()
    mean = col_data.mean()
    cv = std / mean if mean != 0 else np.nan
    
    # Store for CSV
    gi_df.loc[gi_df['biomarker'] == bm, 'std'] = std
    gi_df.loc[gi_df['biomarker'] == bm, 'cv'] = cv
    
    log(f"{bm:<15} | {row['rank']:<5} | {row['mean_abs_shap_aggregated']:<10.6f} | {std:<10.6f} | {cv:.6f}")

vrest_rank = gi_df[gi_df['biomarker'] == 'vrest']['rank'].iloc[0]
dvdt_rank = gi_df[gi_df['biomarker'] == 'dvdtmax']['rank'].iloc[0]
if vrest_rank >= 9 and dvdt_rank >= 9:
    log(f"\nvrest (Rank {vrest_rank}) and dvdtmax (Rank {dvdt_rank}) rank at or near the bottom.")
else:
    log(f"\nvrest (Rank {vrest_rank}) and dvdtmax (Rank {dvdt_rank}) do not both rank at the bottom.")


log("\n==========================================")
log("=== STEP 5 — FIGURE ===")
log("==========================================")
fig, ax = plt.subplots(figsize=(3.5, 3.5))

# Sort ascending for horizontal bar chart so highest is on top
plot_df = gi_df.sort_values(by='mean_abs_shap_aggregated', ascending=True)
y_pos = np.arange(len(plot_df))
ax.barh(y_pos, plot_df['mean_abs_shap_aggregated'], color='gray', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['biomarker'])
ax.set_xlabel('mean|SHAP|')

plt.tight_layout()

fig_pdf = os.path.join(trainai_dir, 'results', 'fig_shap_global.pdf')
fig_png = os.path.join(trainai_dir, 'results', 'fig_shap_global.png')

plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
plt.savefig(fig_png, format='png', dpi=200, bbox_inches='tight')

size_pdf = os.path.getsize(fig_pdf) / 1024.0
size_png = os.path.getsize(fig_png) / 1024.0
log(f"Saved {fig_pdf} ({size_pdf:.1f} KB)")
log(f"Saved {fig_png} ({size_png:.1f} KB)")


log("\n==========================================")
log("=== STEP 6 — PERSIST ===")
log("==========================================")
csv_path = os.path.join(trainai_dir, 'results', 'shap_global_importance_rev.csv')
gi_df.to_csv(csv_path, index=False)

with open(csv_path, 'rb') as f:
    h_csv = hashlib.sha256(f.read()).hexdigest().upper()
    
log(f"Saved CSV: {csv_path}")
log(f"Rows: {len(gi_df)}")
log(f"SHA-256: {h_csv}")

log("\n==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
flags_raised = False

if vrest_rank < 9 or dvdt_rank < 9:
    log(f"  - FLAG: Surprising rank for degenerate feature: vrest Rank {vrest_rank}, dvdtmax Rank {dvdt_rank} despite tiny CV.")
    flags_raised = True

# Near-uniform SHAP values check
max_high = gi_df['mean_abs_shap_high'].max()
min_high = gi_df['mean_abs_shap_high'].min()
if max_high - min_high < 1e-4:
    log("  - FLAG: High class SHAP values are near-uniform across features.")
    flags_raised = True

max_int = gi_df['mean_abs_shap_int'].max()
min_int = gi_df['mean_abs_shap_int'].min()
if max_int - min_int < 1e-4:
    log("  - FLAG: Intermediate class SHAP values are near-uniform across features.")
    flags_raised = True

if not flags_raised:
    log("  - No sanity flags raised.")


doc_path_repo = os.path.join(r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah', 'docs', 'STEP7_SHAP.md')
with open(doc_path_repo, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

log("\nWritten docs/STEP7_SHAP.md")
