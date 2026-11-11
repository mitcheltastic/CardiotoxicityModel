# TdP CiPA Screening — API Contract (v2.1 - Rev2 Production)

Integration reference for frontend applications (Web / Mobile). The model is deployed as a Gradio application on Hugging Face Spaces (`mitcheltastic/tdp-cipa-screening`), which auto-exposes REST and WebSocket/SSE endpoints.

---

## 1. Connection Details

| Field | Value |
|---|---|
| **Space** | `mitcheltastic/tdp-cipa-screening` |
| **Base URL** | `https://mitcheltastic-tdp-cipa-screening.hf.space` |
| **API Prefix** | `/gradio_api` (mandatory — omitting it returns HTTP 405) |
| **Auth** | None (Public Space). |
| **OpenAPI Spec** | `https://mitcheltastic-tdp-cipa-screening.hf.space/gradio_api/openapi.json` |

---

## 2. Input Contract (Predict & Explain Endpoints) ⚠️

Send **11 biomarkers (ratio to drug-free control)** as a JSON array of floats, in **exactly this order**:

| Index | Key | Description / Unit | Example Value (High Risk: Azimilide) | Example Value (Int Risk: Clarithromycin) |
|:---:|:---|:---|:---:|:---:|
| 1 | `qNet` | Net repolarization charge (normalized ratio) | `0.62631` | `0.918296` |
| 2 | `dvdtmax` | Maximum depolarization velocity ($dV_m/dt_{\max}$) | `0.99975` | `0.999048` |
| 3 | `vmax` | Peak action potential voltage ($V_{m,\text{peak}}$) | `1.003687` | `0.990152` |
| 4 | `vrest` | Resting membrane potential ($V_{m,\text{rest}}$) | `-0.999976` | `-1.000058` |
| 5 | `APD50` | Action potential duration at 50% repolarization | `1.422269` | `1.115546` |
| 6 | `APD90` | Action potential duration at 90% repolarization | `1.521552` | `1.145690` |
| 7 | `max_dv` | Maximum repolarization rate ($dV_m/dt_{\text{repol}}$) | `-0.631328` | `-0.838632` |
| 8 | `camax` | Peak intracellular calcium ($Ca_{\text{peak}}$) | `1.063894` | `0.937907` |
| 9 | `carest` | Resting intracellular calcium ($Ca_{\text{rest}}$) | `1.020000` | `0.990000` |
| 10 | `CaTD50` | Calcium transient duration at 50% recovery | `1.121262` | `1.080565` |
| 11 | `CaTD90` | Calcium transient duration at 90% recovery | `1.056818` | `1.038912` |

> **Important Note on Preprocessing**: Features are control-referenced ratio values derived upstream. **Do NOT apply StandardScaler or MinMaxScaler** on the client. Send the ratio values directly.

---

## 3. Endpoint `/predict` — Risk Triage & Conformal Safety

**Route:** `POST /gradio_api/call/predict`  
**Execution Time:** Fast (~35 ms)

Returns a `data` array containing **4 elements**:

```json
[
  // data[0]: Class Probabilities (Gradio Label format)
  {
    "label": "High",
    "confidences": [
      { "label": "high", "confidence": 0.885 },
      { "label": "intermediate", "confidence": 0.0925 },
      { "label": "low", "confidence": 0.0225 }
    ]
  },

  // data[1]: Legacy Tier String
  "Predicted Risk Tier: High",

  // data[2]: Conformal Prediction Safety Metadata Object
  {
    "prediction_set": ["high"],
    "recommended_action": "high",
    "set_size": 1,
    "is_ambiguous": false,
    "out_of_distribution": false,
    "alpha": 0.15384615384615385,
    "q_hat": 0.533987,
    "calibration_note": "Conformal threshold calibrated at drug level on probability vectors averaged over 2000 replicate samples per drug (leave-one-drug-out over 12 training drugs)..."
  },

  // data[3]: Formatted UI Markdown String
  "### Recommended Action: High"
]
```

### Handling Out-of-Distribution (OOD) / Model Abstention
When an input profile is corrupt, anomalous, or unconfident (no risk tier achieves $p \ge 1 - \hat{q} = 0.466013$):
- `data[2].out_of_distribution` will be `true`.
- `data[2].prediction_set` will be `[]` (empty set).
- `data[2].recommended_action` defaults conservatively to `"high"` (high-risk monitoring).
- `data[2].warning` will contain: `"No class met the conformal threshold. Result is not covered by the guarantee. The model abstains and recommends High risk monitoring."`

---

## 4. Endpoint `/explain` — Sub-2ms TreeSHAP Attribution

**Route:** `POST /gradio_api/call/explain`  
**Execution Time:** Ultra-fast (<2 ms via native `TreeExplainer`)  
**Input:** Same 11-value array as `/predict`.

Returns a `data` array where `data[0]` is the SHAP attribution object:

```json
{
  "predicted_class": "high",
  "base_value": 0.3347,
  "contributions": [
    { "biomarker": "qNet", "value": 0.62631, "shap": 0.3008 },
    { "biomarker": "APD90", "value": 1.521552, "shap": 0.1090 },
    { "biomarker": "APD50", "value": 1.422269, "shap": 0.0652 }
    // ... all 11 biomarkers
  ]
}
```

### Visualizing SHAP on the Client
- **Sort**: Order the contributions array by `|shap|` descending.
- **Direction**: `shap > 0` pushed the drug **toward** the predicted tier (red/orange); `shap < 0` pushed it **away** (blue/green).
- **Label**: Display the biomarker name along with the user's input ratio value.

---

## 5. Full Pipeline Biophysical Simulation Endpoints (FDA ORd Engine)

The deployed space also hosts full biophysical differential equation simulation endpoints that convert raw ion channel inhibition data ($IC_{50}$, Hill coefficients) and clinical $C_{\max}$ concentrations into cellular action potentials:

### 5.1 `/sim_drugs` — List Available Reference Compounds
- **Route:** `POST /gradio_api/call/sim_drugs`
- **Input:** `[key]` (string)
- **Output:** JSON list of 28 reference compounds with pre-configured channel affinities.

### 5.2 `/simulate` — Execute Biophysical Simulation
- **Route:** `POST /gradio_api/call/simulate`
- **Input:** `[request_json_str, key]`
- **Supported Payload Formats:**
  1. *Database Drug*: `{"drug": "vandetanib"}`
  2. *Custom Compounds*: `{"cmax_nM": 100, "herg": {"ic50_nM": 50, "hill": 1.0}, "ic50": {...}}`
- **Output:** Returns simulated action potential waveforms, calcium transient dynamics, and averaged biomarkers across $1\times\text{ to }4\times C_{\max}$.

### 5.3 `/sim_status` — Simulation Service Health Check
- **Route:** `POST /gradio_api/call/sim_status`
- **Output:** `{"available": true, "active_jobs": 0}`

---

## 6. Client Implementation Example (JavaScript / TypeScript)

```bash
npm install @gradio/client
```

```typescript
import { Client } from "@gradio/client";

// Initialize connection
const app = await Client.connect("mitcheltastic/tdp-cipa-screening");

// 11 Biomarker inputs in exact order
const biomarkers = [
  0.62631,   // qNet
  0.99975,   // dvdtmax
  1.003687,  // vmax
  -0.999976, // vrest
  1.422269,  // APD50
  1.521552,  // APD90
  -0.631328, // max_dv
  1.063894,  // camax
  1.020000,  // carest
  1.121262,  // CaTD50
  1.056818   // CaTD90
];

// 1. Instant Risk Triage + Conformal Safety
const predResponse = await app.predict("/predict", biomarkers);
const [probabilities, legacyTier, conformalMeta, uiMarkdown] = predResponse.data;

console.log("Recommended Action:", conformalMeta.recommended_action);
console.log("Prediction Set:", conformalMeta.prediction_set);
console.log("Is OOD / Abstain:", conformalMeta.out_of_distribution);

// 2. Sub-2ms TreeSHAP Explainability
const explainResponse = await app.predict("/explain", biomarkers);
const shapData = explainResponse.data[0];

console.log("Top biomarker:", shapData.contributions[0].biomarker, shapData.contributions[0].shap);
```

---

## 7. Raw HTTP / cURL Reference

### Step 1 — Initiate Prediction:
```bash
curl -X POST https://mitcheltastic-tdp-cipa-screening.hf.space/gradio_api/call/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [0.62631, 0.99975, 1.003687, -0.999976, 1.422269, 1.521552, -0.631328, 1.063894, 1.02, 1.121262, 1.056818]}'
# Output: {"event_id": "<EVENT_ID>"}
```

### Step 2 — Stream Result:
```bash
curl -N https://mitcheltastic-tdp-cipa-screening.hf.space/gradio_api/call/predict/<EVENT_ID>
```
