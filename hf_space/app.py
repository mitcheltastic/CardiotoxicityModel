import gradio as gr
import joblib
import numpy as np
import shap
import deploy_utils  # Ensure it is imported so joblib can find custom classes if any

bundle = joblib.load('deployable_chantest_3class.joblib')
background = joblib.load('shap_background.joblib')
explainer = shap.KernelExplainer(bundle['model'].predict_proba, background)

features = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']

def predict(qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90):
    raw = np.array([[qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90]])
    Xs = bundle['scaler'].transform(raw)
    pred = bundle['model'].predict(Xs)[0]
    proba = bundle['model'].predict_proba(Xs)[0]
    
    classes = [bundle['inverse_class_map'][i] for i in range(len(proba))]
    prob_dict = {classes[i]: float(proba[i]) for i in range(len(proba))}
    tier = bundle['inverse_class_map'][pred]
    
    return prob_dict, f"Predicted Risk Tier: {tier.capitalize()}"

def explain(qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90):
    raw = np.array([[qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90]])
    Xs = bundle['scaler'].transform(raw)
    pred = bundle['model'].predict(Xs)[0]
    cls = int(pred)
    
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
            prob_out = gr.Label(label='Class probabilities')
            tier_out = gr.Textbox(label='Predicted TdP risk tier')
            explain_out = gr.JSON(label="SHAP Explanation")
            
    predict_btn.click(fn=predict, inputs=inputs, outputs=[prob_out, tier_out], api_name="predict")
    explain_btn.click(fn=explain, inputs=inputs, outputs=explain_out, api_name="explain")

if __name__ == '__main__':
    demo.launch()
