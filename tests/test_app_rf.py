import os
import sys
import numpy as np

# Add TrainAI and hf_space to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_DIR = os.path.join(BASE_DIR, "hf_space")
if HF_DIR not in sys.path:
    sys.path.insert(0, HF_DIR)

import app

def test_a_abstention():
    print("=== Test (a): Abstention / Empty Set Test ===")
    probs = {"High": 0.40, "Intermediate": 0.35, "Low": 0.25}
    res = app.conformal_from_proba(probs)
    print(f"Input: {probs}")
    print(f"prediction_set:      {res['prediction_set']}")
    print(f"out_of_distribution: {res['out_of_distribution']}")
    print(f"recommended_action:  {repr(res['recommended_action'])}")
    print(f"set_size:            {res['set_size']}")
    print(f"is_ambiguous:        {res['is_ambiguous']}")
    print(f"warning:             {res.get('warning')}")

    assert res['prediction_set'] == [], f"Expected empty set, got {res['prediction_set']}"
    assert res['out_of_distribution'] is True, f"Expected out_of_distribution=True, got {res['out_of_distribution']}"
    assert res['recommended_action'] == "high", f"Expected recommended_action='high', got {res['recommended_action']}"
    assert res['set_size'] == 0, f"Expected set_size=0, got {res['set_size']}"
    print("[PASS] Test (a) PASSED successfully.\n")

def test_b_example1():
    print("=== Test (b): Example 1 (azimilide) Predict & Conformal ===")
    ex1 = [0.62631, 0.99975, 1.003687, -0.999976, 1.422269, 1.521552, -0.631328, 1.063894, 1.02, 1.121262, 1.056818]
    prob_dict, legacy_str, conformal_dict, ui_str = app.predict(*ex1)
    print(f"prob_dict:       {prob_dict}")
    print(f"legacy_str:      {legacy_str}")
    print(f"prediction_set:  {conformal_dict['prediction_set']}")
    print(f"recommended_act: {conformal_dict['recommended_action']}")
    print(f"set_size:        {conformal_dict['set_size']}")
    print(f"is_ambiguous:    {conformal_dict['is_ambiguous']}")
    print(f"q_hat:           {conformal_dict['q_hat']}")

    assert conformal_dict['prediction_set'] == ['high'], f"Expected ['high'], got {conformal_dict['prediction_set']}"
    assert conformal_dict['set_size'] == 1, f"Expected set_size=1, got {conformal_dict['set_size']}"
    assert np.isclose(conformal_dict['q_hat'], 0.533987), f"Expected q_hat=0.533987, got {conformal_dict['q_hat']}"
    assert conformal_dict['recommended_action'] == 'high'
    print("[PASS] Test (b) PASSED successfully.\n")

def test_c_example2():
    print("=== Test (c): Example 2 (clarithromycin) Predict & Conformal ===")
    ex2 = [0.918296, 0.999048, 0.990152, -1.000058, 1.115546, 1.14569, -0.838632, 0.937907, 0.99, 1.080565, 1.038912]
    prob_dict, legacy_str, conformal_dict, ui_str = app.predict(*ex2)
    print(f"prob_dict:       {prob_dict}")
    print(f"legacy_str:      {legacy_str}")
    print(f"prediction_set:  {conformal_dict['prediction_set']}")
    print(f"recommended_act: {conformal_dict['recommended_action']}")
    print(f"set_size:        {conformal_dict['set_size']}")
    print(f"is_ambiguous:    {conformal_dict['is_ambiguous']}")
    print(f"q_hat:           {conformal_dict['q_hat']}")

    assert conformal_dict['prediction_set'] == ['intermediate', 'low'], f"Expected ['intermediate', 'low'], got {conformal_dict['prediction_set']}"
    assert conformal_dict['recommended_action'] == 'intermediate', f"Expected action='intermediate', got {conformal_dict['recommended_action']}"
    assert conformal_dict['is_ambiguous'] is True
    print("[PASS] Test (c) PASSED successfully.\n")

def test_d_explain_example1():
    print("=== Test (d): Example 1 (azimilide) SHAP Explain ===")
    ex1 = [0.62631, 0.99975, 1.003687, -0.999976, 1.422269, 1.521552, -0.631328, 1.063894, 1.02, 1.121262, 1.056818]
    prob_dict, _, _, _ = app.predict(*ex1)
    exp = app.explain(*ex1)

    predicted_class = exp['predicted_class']
    base_val = exp['base_value']
    contribs = exp['contributions']
    sum_shap = sum(c['shap'] for c in contribs)
    pred_prob = prob_dict[predicted_class]
    reconstructed_prob = base_val + sum_shap
    diff = abs(reconstructed_prob - pred_prob)

    print(f"predicted_class:         {predicted_class}")
    print(f"num contributions:       {len(contribs)}")
    print(f"base_value:              {base_val:.8f}")
    print(f"sum(SHAP contributions): {sum_shap:.8f}")
    print(f"base + sum(SHAP):        {reconstructed_prob:.8f}")
    print(f"model predicted prob:    {pred_prob:.8f}")
    print(f"absolute difference:     {diff:.8e}")

    assert len(contribs) == 11, f"Expected 11 contributions, got {len(contribs)}"
    assert np.isclose(reconstructed_prob, pred_prob, atol=1e-6), f"Additivity check failed: diff={diff}"
    print("[PASS] Test (d) PASSED successfully.\n")

def test_i_explain_example2():
    print("=== Test (i): Example 2 (clarithromycin) SHAP Explain ===")
    ex2 = [0.918296, 0.999048, 0.990152, -1.000058, 1.115546, 1.14569, -0.838632, 0.937907, 0.99, 1.080565, 1.038912]
    prob_dict, _, _, _ = app.predict(*ex2)
    exp = app.explain(*ex2)

    predicted_class = exp['predicted_class']
    base_val = exp['base_value']
    contribs = exp['contributions']
    sum_shap = sum(c['shap'] for c in contribs)
    pred_prob = prob_dict[predicted_class]
    reconstructed_prob = base_val + sum_shap
    diff = abs(reconstructed_prob - pred_prob)

    print(f"predicted_class:         {predicted_class} (expected 'low')")
    print(f"num contributions:       {len(contribs)}")
    print(f"base_value:              {base_val:.8f}")
    print(f"sum(SHAP contributions): {sum_shap:.8f}")
    print(f"base + sum(SHAP):        {reconstructed_prob:.8f}")
    print(f"model predicted prob:    {pred_prob:.8f}")
    print(f"absolute difference:     {diff:.8e}")

    assert predicted_class == 'low', f"Expected predicted_class='low', got {predicted_class}"
    assert len(contribs) == 11, f"Expected 11 contributions, got {len(contribs)}"
    assert np.isclose(reconstructed_prob, pred_prob, atol=1e-6), f"Additivity check failed: diff={diff}"
    print("[PASS] Test (i) PASSED successfully.\n")

def test_e_requirements():
    print("=== Test (e): Requirements Check ===")
    req_path = os.path.join(HF_DIR, "requirements.txt")
    with open(req_path, "r") as f:
        reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    print(f"Requirements in hf_space/requirements.txt: {reqs}")
    # Verify imports needed by app.py:
    # app.py imports: joblib, numpy, shap, gradio (built-in to HF Spaces), deploy_utils (local)
    for pkg in ["joblib", "numpy", "shap"]:
        matched = any(r.split("==")[0] == pkg for r in reqs)
        print(f"  Package '{pkg}': {'FOUND' if matched else 'MISSING'}")
        assert matched, f"Missing required package {pkg} in requirements.txt"
    print("[PASS] Test (e) PASSED successfully.\n")

if __name__ == "__main__":
    test_a_abstention()
    test_b_example1()
    test_c_example2()
    test_d_explain_example1()
    test_i_explain_example2()
    test_e_requirements()
    print(">>> ALL TESTS a-e and i COMPLETED AND PASSED! <<<")
