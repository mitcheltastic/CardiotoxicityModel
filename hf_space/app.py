import os
import gradio as gr
import joblib
import numpy as np
import shap
import deploy_utils  # Ensure it is imported so joblib can find custom classes if any

base_dir = os.path.dirname(os.path.abspath(__file__))
bundle_path = os.path.join(base_dir, 'deployable_chantest_3class.joblib')
bg_path = os.path.join(base_dir, 'shap_background.joblib')

bundle = joblib.load(bundle_path)
background = joblib.load(bg_path)
explainer = shap.KernelExplainer(bundle['model'].predict_proba, background)

ALPHA = 0.154
Q_HAT = 0.6275
SEVERITY = {"low": 0, "intermediate": 1, "high": 2}
CALIBRATION_NOTE = (
    "Conformal threshold calibrated at drug level on probability\n"
    "vectors averaged over 2000 replicate samples per drug (leave-one-drug-out over 12\n"
    "training drugs). This response is based on a single input row; the coverage\n"
    "guarantee does not strictly transfer to single-row input."
)

def conformal_from_proba(prob_dict):
    threshold = 1.0 - Q_HAT
    included_classes = []
    for c, p in prob_dict.items():
        if p >= threshold:
            included_classes.append(c)
    included_classes.sort(key=lambda c: SEVERITY[c.lower()], reverse=True)
    set_size = len(included_classes)
    is_ambiguous = set_size > 1
    
    if set_size == 0:
        argmax_class = max(prob_dict, key=prob_dict.get)
        return {
            "prediction_set": [],
            "recommended_action": argmax_class.lower(),
            "set_size": 0,
            "is_ambiguous": True,
            "out_of_distribution": True,
            "warning": "No class met the conformal threshold. Result is not covered by the guarantee.",
            "alpha": ALPHA,
            "q_hat": Q_HAT,
            "calibration_note": CALIBRATION_NOTE
        }
    
    recommended_action = included_classes[0]
    return {
        "prediction_set": included_classes,
        "recommended_action": recommended_action.lower(),
        "set_size": set_size,
        "is_ambiguous": is_ambiguous,
        "out_of_distribution": False,
        "alpha": ALPHA,
        "q_hat": Q_HAT,
        "calibration_note": CALIBRATION_NOTE
    }

features = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']

def _preprocess(raw):
    if 'scaler' in bundle and bundle['scaler'] is not None:
        return bundle['scaler'].transform(raw)
    return raw

def predict(qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90):
    raw = np.array([[qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90]])
    Xs = _preprocess(raw)
    pred = int(bundle['model'].predict(Xs)[0])
    proba = bundle['model'].predict_proba(Xs)[0]
    
    classes = [bundle['inverse_class_map'][i] for i in range(len(proba))]
    prob_dict = {classes[i]: float(proba[i]) for i in range(len(proba))}
    tier = bundle['inverse_class_map'][pred]
    legacy_str = f"Predicted Risk Tier: {tier.capitalize()}"
    
    conformal_dict = conformal_from_proba(prob_dict)
    
    if conformal_dict.get("out_of_distribution"):
        ui_str = (
            f"### ⚠️ WARNING: Out of Distribution\n"
            f"**{conformal_dict['warning']}**\n\n"
            f"Fallback Recommended Action: **{conformal_dict['recommended_action'].capitalize()}**"
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
    pred = int(bundle['model'].predict(Xs)[0])
    cls = pred
    
    sv = explainer.shap_values(Xs, nsamples=500)
    
    # Handle list of arrays (one per class) OR 3D array
    if isinstance(sv, list):
        shap_row = sv[cls][0]
    else:
        if sv.ndim == 3:
            shap_row = sv[0, :, cls]
        else:
            shap_row = sv[0]
            
    # expected_value can be a list or array
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[cls]
    else:
        base_value = explainer.expected_value
        
    contributions = []
    for i, name in enumerate(features):
        contributions.append({
            "biomarker": name,
            "value": float(raw[0, i]),
            "shap": float(shap_row[i])
        })
        
    return {
        "predicted_class": bundle['inverse_class_map'][pred],
        "base_value": float(base_value),
        "contributions": contributions
    }

with gr.Blocks(title="CiPA TdP Risk Classifier") as demo:
    gr.Markdown("# CiPA TdP Risk Classifier")
    gr.Markdown("Predicts the 3-tier Torsade de Pointes (TdP) risk (High, Intermediate, Low) based on 11 CiPA biomarkers.")
    
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
                    [43.637118, 72.87939091, 41.82645396, -88.01230706, 338.5000001, 441.25, -0.238296891, 0.000254324, 7.60e-05, 337.4999999, 767.2500001],
                    [69.60795403, 72.89621306, 41.67183264, -88.01437399, 238, 290, -0.376227947, 0.000239128, 7.45e-05, 301, 726]
                ],
                inputs=inputs,
                label="Example Drugs (High Risk = azimilide, Low Risk = loratadine)"
            )
            
        with gr.Column():
            action_out = gr.Markdown()
            prob_out = gr.Label(label='Class probabilities')
            tier_out = gr.Textbox(label='Predicted TdP risk tier (Legacy)', visible=False)
            conformal_out = gr.JSON(label="Conformal Prediction Details")
            explain_out = gr.JSON(label="SHAP Explanation")
            
    predict_btn.click(fn=predict, inputs=inputs, outputs=[prob_out, tier_out, conformal_out, action_out], api_name="predict")
    explain_btn.click(fn=explain, inputs=inputs, outputs=explain_out, api_name="explain")

if __name__ == '__main__':
    demo.launch()
