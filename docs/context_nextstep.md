# context.md — Handoff for Deployment/Testing Session

> Paste this into a new chat to continue the work. The previous chat completed all
> modeling; **the next session is temporary testing / deployment** of the finished model.

---

## 0. How to use this doc

Modeling is **done**. Do not re-train, re-tune, or re-run the sweep. The next job is to
load the already-persisted model and build/test a deployment path (a small inference tool
that takes 11 biomarker inputs and returns a risk tier + explanation). Treat the numbers
below as final and quotable.

---

## 1. Working style (how we collaborate)

- I (the user, **HCiM**) work in **Antigravity IDE** (Google's agentic IDE, Gemini-based),
  in a repo called **`TrainAI`**, main notebook **`plus_ultra.ipynb`**.
- The assistant's role is a **prompt partner**: give me clean, copy-paste-ready prompts to
  paste into the Antigravity agent, which executes them. Don't write code directly unless
  I hit a persistent bug that needs debugging.
- One clear step at a time. I prefer decisive recommendations over big option menus.
- I mix English + Bahasa Indonesia, casual tone.

### Antigravity agent quirks (important)
- The agent narrates fake "set 15-minute timer / timer expired" lines while background
  tasks run. **These are not real elapsed time** — judge progress by message timestamps
  and by files written to disk, not the timer narration.
- The agent builds the notebook via an external helper (`build_notebook.py` in its scratch
  dir) that runs all cells in a shared context and bakes outputs in. So after each phase it
  **re-runs the notebook top-to-bottom**; the editor tab can show a stale view until you
  reopen the file.
- Always instruct it to checkpoint to disk incrementally so a crash never loses long runs.

---

## 2. Project in one paragraph

In-silico cardiotoxicity (Torsade de Pointes / TdP) risk prediction from **11 CiPA cardiac
biomarkers**. Classifies a drug into a 3-tier risk category: **high / intermediate / low**.
Built via stacking + voting ensembles over ML (RF, XGB, LR, SVM) and DL (3 ANNs) base
learners. This is a bachelor's **capstone (Tugas Akhir)** — everything must be explainable
and thesis-defensible. End goal: one deployable model behind a web screening tool.

Full detail lives in **`PROJECT_SUMMARY.md`** in the repo root. This context.md is the
operational subset needed to continue.

---

## 3. The deployable model (THE thing you deploy)

**File:** `./artifacts/deployable_chantest_3class.joblib`

It's a dict bundle:
```
{
  'model':  StackingClassifier,          # base={RF, XGB}, meta=RandomForestClassifier
  'scaler': StandardScaler,              # fit on chantest train ONLY — must be applied to input
  'biomarkers': [...11 names, exact order...],
  'class_map':         {'high':0, 'intermediate':1, 'low':2},
  'inverse_class_map': {0:'high', 1:'intermediate', 2:'low'},
  'dataset':'chantest', 'scenario':'3class',
  'architecture':'stacking base={RF,XGB} meta=RF',
  ...metadata (macro-F1, CV scheme, seed=42, sklearn version)
}
```

### ⚠️ CRITICAL reload requirement
The bundle uses custom wrapper classes (`PreFittedEstimatorWrapper`, `FittedHolder`,
`FinalEstimatorWrapper`). **`deploy_utils.py` MUST be importable** when calling
`joblib.load()`, or it throws `ModuleNotFoundError`. So the deployment package must ship
**`deploy_utils.py` alongside the `.joblib`**. This is the #1 deployment gotcha.

### Input → output contract (for the web tool)
- **Input:** 11 raw (UNSCALED) biomarker values, in this EXACT order:
  `qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90`
- **Pipeline:** `X_raw → bundle['scaler'].transform(X_raw) → bundle['model'].predict(...) →
  bundle['inverse_class_map'][pred] → 'high'/'intermediate'/'low'`
- Verified in the prior session: reload produced **32,000/32,000 identical predictions** vs
  in-memory, and a raw-input demo classified high/intermediate/low correctly end to end.
- For explanation output, SHAP was computed via KernelExplainer on the full pipeline's
  `predict_proba` (see §5).

---

## 4. Final results (quotable, final)

**Deployable winner — chantest, 3-class, stacking base={RF,XGB} meta=RF:**
- Test macro-F1 (point): **0.8280** (accuracy 0.8242)
- Bootstrap macro-F1 median [95% CI]: **0.819 [0.694, 0.878]** (10,000 drug-level iters)
- Per-class (F1 / recall): high **0.795 / 0.992**, intermediate **0.768 / 0.660**,
  low **0.921 / 0.920**
- **Safety headline:** high-class recall ~0.99, and **zero high-risk drugs ever predicted
  as low** (confusion matrix bottom-left = 0). This is the key clinical selling point.

**Cross-dataset best (3-class), all won by stacking + meta=RF:**
| Dataset | Best combo | Macro-F1 | Recall(high) | F1(intermediate) |
|---|---|---|---|---|
| chantest | RF+XGB | 0.828 | 0.992 | 0.768 |
| nanion | RF+LR | 0.685 | 1.000 | 0.668 |
| manual | XGB+LR+SVM | 0.728 | 0.980 | 0.705 |

**Findings:** stacking > voting (0.828 vs 0.822); winning *architecture* (stacking+meta=RF)
invariant across all 3 datasets; intermediate is the hardest tier everywhere; high recall
strong everywhere.

---

## 5. Explainability (already done)
- SHAP via `shap.KernelExplainer` on full pipeline `predict_proba`, 100-row background,
  500-row eval, on scaled inputs.
- Global importance rank: **qNet (0.162) ≫ vrest (0.084) > APD50 (0.071) > max_dv (0.063)
  > APD90 (0.058)** > vmax > camax > CaTD50 > CaTD90 > dvdtmax > carest.
- qNet #1 aligns with FDA/CiPA literature (primary TdP biomarker) — strong defense point.

---

## 6. Honest caveats (keep, don't hide — examiners will probe)
- **Mixed slightly beat pure-ML** in the sweep (~0.842 vs 0.828). Pure-ML RF+XGB was chosen
  deliberately for paradigm purity, simplicity, explainability, lower latency. Defensible.
- chantest CI **[0.694, 0.878]** overlaps nanion's — highest *median*, not proven
  statistically significantly better.
- **vrest #2** in SHAP is mildly surprising vs classic repolarization markers; have an
  answer ready (excitability/Na availability).
- **manual** had suspiciously high grouped-CV training score (~0.98 XGB) but normal test
  (0.728) — leakage was checked and exonerated on the held-out test set.

---

## 7. The methodological crown jewel (lead with this in the thesis)
Caught **drug-level data leakage**: the 2,000 samples/drug meant row-level `StratifiedKFold`
gave fake ~0.99 CV. Diagnostic under whole-drug holdout collapsed to ~0.46, confirming
memorization. Fixed by re-tuning everything with **`StratifiedGroupKFold(n_splits=4)`
grouped by `drug_name`**, and the CiPA bootstrap resamples at the **drug level** too. All
reported numbers are leakage-free. This is the key contribution — don't bury it.

---

## 8. Repo / artifacts map
```
TrainAI/
  plus_ultra.ipynb                         # all phases 0–5c, executed
  deploy_utils.py                          # REQUIRED to reload the model bundle
  PROJECT_SUMMARY.md                       # full detailed record
  data/                                    # chantest, nanion, manual (train+test pairs)
  artifacts/
    deployable_chantest_3class.joblib      # <-- THE deployable model
    base_models_grouped.joblib             # leakage-free tuned base learners
    keras_grouped/                         # 18 ANN .keras files (grouped-CV)
    base_models.joblib, keras/             # old leaky baseline (backup only, do not use)
    phase3_protocolA_results.csv           # 612-config sweep results
    phase4_bootstrap.joblib / _summary.csv # bootstrap CIs
    shap_values.joblib                     # SHAP values + importance tables
    shap/                                  # 3 beeswarm plots
    figures/                               # 8 publication figures (300 dpi)
```

Dataset spec: 24k train (12 drugs) / 32k test (16 drugs), disjoint, 2000 samples/drug,
labels high/intermediate/low. manual had 25 train + 1 test NaN rows dropped.

Env: Python 3.12, scikit-learn 1.8.0, xgboost 3.2.0, tensorflow 2.21.0, scikeras 0.13.0,
shap + matplotlib installed. seed=42 everywhere.

---

## 9. NEXT SESSION GOAL — deployment / temporary testing
Build and test an inference path around `deployable_chantest_3class.joblib`:
1. Confirm the bundle reloads cleanly in a fresh env **with `deploy_utils.py` present**.
2. Wrap it in a minimal inference function/app: 11 raw biomarker inputs → scale → predict →
   return risk label (+ optionally class probabilities and a SHAP-based explanation).
3. "Temporary testing" first — a quick local sanity harness (e.g. a few known test drugs,
   confirm sensible labels), before any real web UI.
4. Remember: package must include `deploy_utils.py`, the scaler is inside the bundle (don't
   re-fit), and inputs must be in the exact 11-biomarker order above.

Open decision for next session: what deployment surface? (e.g. simple Python function,
Streamlit/Gradio quick tool, or a proper Flask/FastAPI endpoint.)
