import os
import gradio as gr
import joblib
import numpy as np
import shap
import deploy_utils  # Ensure it is imported so joblib can find custom classes if any

base_dir = os.path.dirname(os.path.abspath(__file__))
bundle_path = os.path.join(base_dir, 'deployable_chantest_rf.joblib')

bundle = joblib.load(bundle_path)
model = bundle['model']
explainer = shap.TreeExplainer(model)

EXPECTED_Q_HAT = 0.533987
EXPECTED_ALPHA = 2.0 / 13.0

Q_HAT = float(bundle.get('q_hat', EXPECTED_Q_HAT))
ALPHA = float(bundle.get('alpha', EXPECTED_ALPHA))

assert np.isclose(Q_HAT, EXPECTED_Q_HAT, atol=1e-5), f"Q_HAT mismatch: {Q_HAT} vs {EXPECTED_Q_HAT}"
assert np.isclose(ALPHA, EXPECTED_ALPHA, atol=1e-5), f"ALPHA mismatch: {ALPHA} vs {EXPECTED_ALPHA}"

# Validate and use inverse class map from bundle
inv_map = bundle['inverse_class_map']
assert inv_map == {0: "high", 1: "intermediate", 2: "low"}, f"Unexpected inverse_class_map: {inv_map}"

SEVERITY = {"High": 2, "Intermediate": 1, "Low": 0}
CALIBRATION_NOTE = (
    "Conformal threshold calibrated at drug level on probability\n"
    "vectors averaged over 2000 replicate samples per drug (leave-one-drug-out over 12\n"
    "training drugs). This response is based on a single input row; the coverage\n"
    "guarantee does not strictly transfer to single-row input."
)

EXPECTED_FEATURES = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
features = list(bundle['biomarkers'])
assert features == EXPECTED_FEATURES, f"Biomarker feature mismatch: {features} vs {EXPECTED_FEATURES}"

# Verify explainer output shape once at startup
_test_x = np.zeros((1, len(features)))
_test_sv = explainer.shap_values(_test_x)
if isinstance(_test_sv, list):
    print(f"TreeExplainer initialized. Multi-class SHAP output is a list of {len(_test_sv)} arrays with shape {_test_sv[0].shape}.")
else:
    print(f"TreeExplainer initialized. Multi-class SHAP output shape: {_test_sv.shape}")

# --- Full pipeline simulation (isolated add-on; failures never affect predict/explain) ---
try:
    from sim import sim_service
    sim_service.start_background_init()
    _SIM_AVAILABLE = True
except Exception as _sim_err:
    print(f"[sim] disabled: {_sim_err}", flush=True)
    _SIM_AVAILABLE = False

def conformal_from_proba(prob_dict):
    # Internal conformal rule uses Title-case
    proba = {k.capitalize(): float(v) for k, v in prob_dict.items()}
    threshold = 1.0 - Q_HAT
    pred_set = [t for t in ["High", "Intermediate", "Low"] if proba.get(t, 0.0) >= threshold]
    set_size = len(pred_set)
    is_ambiguous = set_size > 1

    if not pred_set:                 # model abstains
        ood, action = True, "High"   # most conservative tier
        warning_msg = (
            "No class met the conformal threshold. Result is not covered by the guarantee. "
            "The model abstains and recommends High risk monitoring."
        )
    else:
        ood, action = False, pred_set[0]   # ordered by severity
        warning_msg = None

    # Return prediction_set and recommended_action in lowercase per external API contract
    result = {
        "prediction_set": [t.lower() for t in pred_set],
        "recommended_action": action.lower(),
        "set_size": set_size,
        "is_ambiguous": is_ambiguous,
        "out_of_distribution": ood,
        "alpha": ALPHA,
        "q_hat": Q_HAT,
        "calibration_note": CALIBRATION_NOTE
    }
    if warning_msg:
        result["warning"] = warning_msg
    return result

def _preprocess(raw):
    if 'scaler' in bundle and bundle['scaler'] is not None:
        return bundle['scaler'].transform(raw)
    return raw

def predict(qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90):
    raw = np.array([[qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90]])
    Xs = _preprocess(raw)
    proba = model.predict_proba(Xs)[0]
    pred = int(np.argmax(proba))
    tier = inv_map[pred]

    # Return prob_dict with lowercase keys matching existing contract
    prob_dict = {inv_map[i]: float(proba[i]) for i in range(len(proba))}
    legacy_str = f"Predicted Risk Tier: {tier.capitalize()}"

    conformal_dict = conformal_from_proba(prob_dict)

    if conformal_dict.get("out_of_distribution"):
        ui_str = (
            f"### ⚠️ WARNING: Out of Distribution / Model Abstains\n"
            f"No class met the conformal threshold (1 - q̂ = {1.0 - Q_HAT:.4f}). "
            f"The model abstains and recommends: **{conformal_dict['recommended_action'].capitalize()}** risk monitoring."
        )
    elif conformal_dict["is_ambiguous"]:
        ui_str = (
            f"### Recommended Action: {conformal_dict['recommended_action'].capitalize()}\n"
            f"**Prediction Set:** {', '.join([c.capitalize() for c in conformal_dict['prediction_set']])}\n\n"
            f"*The model cannot rule out the more severe tier, so the more severe tier is recommended.*"
        )
    else:
        ui_str = f"### Recommended Action: {conformal_dict['recommended_action'].capitalize()}"

    return prob_dict, legacy_str, conformal_dict, ui_str

def explain(qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90):
    raw = np.array([[qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90]])
    Xs = _preprocess(raw)
    proba = model.predict_proba(Xs)[0]
    cls = int(np.argmax(proba))

    sv = explainer.shap_values(Xs)

    if isinstance(sv, list):
        contrib = sv[cls][0]
    elif sv.ndim == 3:
        contrib = sv[0, :, cls]
    else:
        contrib = sv[0]

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = float(explainer.expected_value[cls])
    else:
        base_value = float(explainer.expected_value)

    contributions = []
    for i, name in enumerate(features):
        contributions.append({
            "biomarker": name,
            "value": float(raw[0, i]),
            "shap": float(contrib[i])
        })

    return {
        "predicted_class": inv_map[cls],
        "base_value": float(base_value),
        "contributions": contributions
    }

with gr.Blocks(title="CiPA TdP Risk Classifier") as demo:
    gr.Markdown("# CiPA TdP Risk Classifier")
    gr.Markdown("Predicts the 3-tier Torsade de Pointes (TdP) risk (High, Intermediate, Low) based on 11 CiPA biomarkers (ratio to control).")
    
    with gr.Row():
        with gr.Column():
            inputs = [
                gr.Number(label="qNet"), gr.Number(label="dvdtmax"), gr.Number(label="vmax"),
                gr.Number(label="vrest"), gr.Number(label="APD50"), gr.Number(label="APD90"),
                gr.Number(label="max_dv"), gr.Number(label="camax"), gr.Number(label="carest"),
                gr.Number(label="CaTD50"), gr.Number(label="CaTD90")
            ]
            
            with gr.Row():
                predict_btn = gr.Button("Predict", variant="primary")
                explain_btn = gr.Button("Explain this prediction")
                
            gr.Examples(
                examples=[
                    [0.62631, 0.99975, 1.003687, -0.999976, 1.422269, 1.521552, -0.631328, 1.063894, 1.02, 1.121262, 1.056818],
                    [0.918296, 0.999048, 0.990152, -1.000058, 1.115546, 1.14569, -0.838632, 0.937907, 0.99, 1.080565, 1.038912]
                ],
                inputs=inputs,
                label="Example Drugs (High Risk = azimilide, Intermediate Risk = clarithromycin)"
            )
            
        with gr.Column():
            action_out = gr.Markdown()
            prob_out = gr.Label(label='Class probabilities')
            tier_out = gr.Textbox(label='Predicted TdP risk tier (Legacy)', visible=False)
            conformal_out = gr.JSON(label="Conformal Prediction Details")
            explain_out = gr.JSON(label="SHAP Explanation")
            
    predict_btn.click(fn=predict, inputs=inputs, outputs=[prob_out, tier_out, conformal_out, action_out], api_name="predict")
    explain_btn.click(fn=explain, inputs=inputs, outputs=explain_out, api_name="explain")

    # --- Full pipeline simulation endpoints (hidden from the UI, API only) ---
    if _SIM_AVAILABLE:
        with gr.Column(visible=False):
            sim_req = gr.Textbox(label="Request JSON")
            sim_key = gr.Textbox(label="Key", type="password")
            sim_out = gr.JSON()
            gr.Button("Simulate").click(sim_service.simulate, [sim_req, sim_key], sim_out,
                                        api_name="simulate", concurrency_limit=1, concurrency_id="sim_run")
            gr.Button("Drugs").click(sim_service.list_drugs, [sim_key], sim_out,
                                     api_name="sim_drugs", concurrency_id="sim_meta")
            gr.Button("Status").click(sim_service.status, None, sim_out,
                                      api_name="sim_status", concurrency_id="sim_meta")

if __name__ == '__main__':
    demo.launch()
