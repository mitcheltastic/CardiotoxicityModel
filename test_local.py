import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from app import predict, explain
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

print("Testing High Risk (azimilide) Explain:")
exp_high = explain(43.637118, 72.87939091, 41.82645396, -88.01230706, 338.5000001, 441.25, -0.238296891, 0.000254324, 7.60e-05, 337.4999999, 767.2500001)
print(json.dumps(exp_high, indent=2))
sum_shap_high = sum([c['shap'] for c in exp_high['contributions']])
total_high = exp_high['base_value'] + sum_shap_high
print(f"Base + Sum(SHAP) = {total_high}")
qnet_high = next(c for c in exp_high['contributions'] if c['biomarker'] == 'qNet')
print(f"qNet contribution: {qnet_high['shap']}")

print("\nTesting Low Risk (loratadine) Explain:")
exp_low = explain(69.60795403, 72.89621306, 41.67183264, -88.01437399, 238, 290, -0.376227947, 0.000239128, 7.45e-05, 301, 726)
print(json.dumps(exp_low, indent=2))
sum_shap_low = sum([c['shap'] for c in exp_low['contributions']])
total_low = exp_low['base_value'] + sum_shap_low
print(f"Base + Sum(SHAP) = {total_low}")
qnet_low = next(c for c in exp_low['contributions'] if c['biomarker'] == 'qNet')
print(f"qNet contribution: {qnet_low['shap']}")
