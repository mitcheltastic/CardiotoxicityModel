# TdP CiPA Screening — API Contract (v2)

Integration reference for the frontend. The model is deployed as a Gradio app on
Hugging Face Spaces, which auto-exposes a REST API. There are **two endpoints**:
`predict` (instant risk tier) and `explain` (per-prediction SHAP, slower).

**Live API docs (source of truth):** open the Space, click **"Use via API"** in the footer.
Confirm the endpoint names there — this doc assumes they are `predict` and `explain`.

---

## 1. Connection details

| Field | Value |
|---|---|
| Space | `mitcheltastic/tdp-cipa-screening` |
| Base URL | `https://mitcheltastic-tdp-cipa-screening.hf.space` |
| API prefix | `/gradio_api` (required — omitting it returns 405) |
| Auth | None (public Space). If ever set Private: header `Authorization: Bearer <HF_TOKEN>`. |
| OpenAPI spec | `https://mitcheltastic-tdp-cipa-screening.hf.space/gradio_api/openapi.json` |

**Gradio REST is two-step:** `POST /gradio_api/call/<name>` returns an `event_id`, then
`GET /gradio_api/call/<name>/<event_id>` streams the result (SSE). The `@gradio/client`
library (section 5) hides this — prefer it.

---

## 2. Input contract (same for both endpoints)  ⚠️

Send **11 RAW (UNSCALED) biomarker values** as a JSON array, in **exactly this order**.
Do NOT scale on the client — the server applies its own `StandardScaler`. Order matters,
not the names.

| # | Key | Type | | # | Key | Type |
|---|-----|------|-|---|-----|------|
| 1 | `qNet` | float | | 7 | `max_dv` | float |
| 2 | `dvdtmax` | float | | 8 | `camax` | float |
| 3 | `vmax` | float | | 9 | `carest` | float |
| 4 | `vrest` | float | | 10 | `CaTD50` | float |
| 5 | `APD50` | float | | 11 | `CaTD90` | float |
| 6 | `APD90` | float | | | | |

---

## 3. Endpoint `predict` — risk tier (instant)

**Route:** `POST /gradio_api/call/predict`

Returns a `data` array of 2 elements:
- `data[0]` — class probabilities (Gradio `Label` format):
  ```json
  { "label": "High",
    "confidences": [
      { "label": "high",         "confidence": 1.0 },
      { "label": "intermediate", "confidence": 0.0 },
      { "label": "low",          "confidence": 0.0 } ] }
  ```
- `data[1]` — risk tier string: `"High"` | `"Intermediate"` | `"Low"`.

Note: tree ensemble → confidences often come back as hard `1.0 / 0.0`. Display as a tier,
not as a calibrated clinical probability.

---

## 4. Endpoint `explain` — per-prediction SHAP (slower)

**Route:** `POST /gradio_api/call/explain`
Same 11-value input. Call this only on user demand (e.g. an "Explain" button) — it takes a
few seconds (model-agnostic KernelExplainer), and the FIRST call after the Space wakes is
slower (one-time JIT compile). `predict` stays instant regardless.

Returns a `data` array whose `data[0]` is the SHAP object:
```json
{
  "predicted_class": "high",
  "base_value": 0.3347,
  "contributions": [
    { "biomarker": "qNet",  "value": 43.64,  "shap": 0.3008 },
    { "biomarker": "APD90", "value": 441.25, "shap": 0.1090 }
    // ... all 11 biomarkers
  ]
}
```

**How to read it (for the chart):**
- SHAP values explain the **predicted class only** — i.e. "why this tier."
- `base_value` = model's average output for that class; `base_value + Σ shap ≈ P(predicted tier)`.
- `shap > 0` → that biomarker pushed the drug **toward** the predicted tier.
- `shap < 0` → pushed it **away** from the predicted tier.
- `value` is the raw (unscaled) biomarker value the user entered — show it as a label.

---

## 5. Recommended: JavaScript client (handles both endpoints + the two-step flow)

```bash
npm i @gradio/client
```
```js
import { Client } from "@gradio/client";
const app = await Client.connect("mitcheltastic/tdp-cipa-screening");

const inputs = [qNet, dvdtmax, vmax, vrest, APD50, APD90,
                max_dv, camax, carest, CaTD50, CaTD90]; // 11 RAW numbers, exact order

// Risk tier (instant)
const pred = await app.predict("/predict", inputs);
const [probs, tier] = pred.data;      // tier === "High" | "Intermediate" | "Low"

// SHAP explanation (on demand)
const exp = await app.predict("/explain", inputs);
const shap = exp.data[0];             // { predicted_class, base_value, contributions[] }
```

---

## 6. Rendering the SHAP explanation (suggested)

Horizontal bar chart (Chart.js / Recharts / D3 / plain divs):
- One bar per biomarker; **sort by `|shap|` descending**.
- Bar length = `shap`; positive one color (toward-risk), negative another (away).
- Row label = `biomarker` + its raw `value`.
- Optionally show `base_value` and the resulting `P(predicted tier)` as a header, and a note:
  "Explaining why this drug was classified **{predicted_class}**."

---

## 7. Raw HTTP (no client library)

**Step 1 — POST, get event id:**
```bash
curl -X POST https://mitcheltastic-tdp-cipa-screening.hf.space/gradio_api/call/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [43.64, 12.3, 40.1, -88.0, 210.0, 441.25, 8.2, 0.0004, 0.0001, 190.0, 260.0]}'
# -> {"event_id": "abc123"}
```
**Step 2 — GET, stream result (do this immediately; event_id expires fast):**
```bash
curl -N https://mitcheltastic-tdp-cipa-screening.hf.space/gradio_api/call/predict/abc123
# event: complete
# data: [{"label":"High","confidences":[...]}, "High"]
```
For SHAP, swap `predict` -> `explain` in both URLs.

---

## 8. Operational notes

- **Prefix required:** the `/gradio_api` prefix is mandatory. `/call/predict` (no prefix) -> 405.
- **Cold start:** free CPU Space sleeps after ~48h idle; first request wakes it (a few sec / brief 503). Add retry + loading state. `@gradio/client` waits automatically.
- **`explain` latency:** seconds, not instant, and slowest on the first call after wake. Keep it behind a button; keep `predict` on the main flow.
- **Errors:** wrong length / non-numeric input -> HTTP 422. Always send exactly 11 numbers.
- **CORS:** enabled — browser-side calls work.
