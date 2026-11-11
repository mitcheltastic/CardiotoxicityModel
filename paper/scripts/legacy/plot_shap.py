import os
import joblib
import shap
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

print("Loading SHAP values...")
shap_data = joblib.load('./artifacts/shap_values.joblib')
shap_values = shap_data['shap_values'] # list of 3 (for each class)
X_test_shap = shap_data['eval_sample']

classes = ['high', 'intermediate', 'low']
for idx, c_name in enumerate(classes):
    print(f"Plotting SHAP summary for {c_name} class...")
    plt.figure()
    shap.summary_plot(shap_values[:, :, idx], X_test_shap, show=False)
    plt.title(f"SHAP Summary (Beeswarm) Plot - {c_name.upper()} Class", pad=20)
    plt.savefig(f'./artifacts/figures/shap_summary_{c_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("SHAP plots regenerated at 300 dpi.")
