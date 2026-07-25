==========================================
=== STEP 0 — HARD RULES AND SETUP ===
==========================================
models/chantest_stacking_model_rev.joblib hash matches: PASS
Canonical feature order loaded: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']

==========================================
=== STEP 1 — DECLARE THE EXPLANATION TARGET ===
==========================================
Approach: StackingClassifier is not natively supported by TreeExplainer since it's an ensemble of base models (RF, XGB) feeding a meta-learner. Explaining the meta-learner alone would attribute importance to base-model probabilities rather than the 11 biomarkers, missing the biological link.
Instead, we treat the FULL fitted stack's predict_proba as a black-box function and apply a model-agnostic explainer (KernelExplainer). This guarantees importance is correctly attributed to the input 11 biomarkers.
SHAP Version: 0.52.0
Explainer Class: shap.KernelExplainer

==========================================
=== STEP 2 — DATA FOR EXPLANATION ===
==========================================
Feature order verified against canonical order: PASS
Background: A seed=42 random subsample of chantest TRAINING rows. Size: 100
Explained on: A seed=42 stratified subsample of chantest TESTING rows. Size: 320
Per-drug row counts in the explained subsample:
  astemizole: 20
  azimilide: 20
  clarithromycin: 20
  clozapine: 20
  disopyramide: 20
  domperidone: 20
  droperidol: 20
  ibutilide: 20
  loratadine: 20
  metoprolol: 20
  nifedipine: 20
  nitrendipine: 20
  pimozide: 20
  risperidone: 20
  tamoxifen: 20
  vandetanib: 20
Confirmed: The explained subsample covers all 16 test drugs.

==========================================
=== STEP 3 — GLOBAL IMPORTANCE ===
==========================================
Computing SHAP values using KernelExplainer (this may take a few minutes)...
Aggregation rule: Mean of mean|SHAP| across the three classes.

Rank  | Biomarker       | High       | Int        | Low        | Aggregated (mean|SHAP|)
--------------------------------------------------------------------------------
1     | qNet            | 0.129443   | 0.132843   | 0.174315   | 0.145534
2     | vrest           | 0.028388   | 0.171309   | 0.146771   | 0.115489
3     | max_dv          | 0.115931   | 0.112952   | 0.010718   | 0.079867
4     | APD50           | 0.060440   | 0.082904   | 0.065969   | 0.069771
5     | APD90           | 0.070343   | 0.057913   | 0.018875   | 0.049044
6     | vmax            | 0.007236   | 0.053562   | 0.053743   | 0.038180
7     | dvdtmax         | 0.005932   | 0.019149   | 0.013368   | 0.012816
8     | camax           | 0.016483   | 0.011323   | 0.009453   | 0.012420
9     | CaTD50          | 0.011085   | 0.010114   | 0.006511   | 0.009237
10    | CaTD90          | 0.009562   | 0.008841   | 0.005707   | 0.008037
11    | carest          | 0.000511   | 0.000543   | 0.000912   | 0.000655

qNet exactly: Rank 1, mean|SHAP| = 0.145534

==========================================
=== STEP 4 — DEGENERATE FEATURE CHECK ===
==========================================
Biomarker       | Rank  | mean|SHAP| | Std        | CV
-----------------------------------------------------------------
qNet            | 1     | 0.145534   | 0.233621   | 0.277707
vrest           | 2     | 0.115489   | 0.000102   | -0.000102
max_dv          | 3     | 0.079867   | 0.337446   | -0.435700
APD50           | 4     | 0.069771   | 0.290217   | 0.238310
APD90           | 5     | 0.049044   | 0.433458   | 0.337659
vmax            | 6     | 0.038180   | 0.011765   | 0.011786
dvdtmax         | 7     | 0.012816   | 0.002599   | 0.002602
camax           | 8     | 0.012420   | 0.081676   | 0.081557
CaTD50          | 9     | 0.009237   | 0.082431   | 0.076962
CaTD90          | 10    | 0.008037   | 0.108409   | 0.103152
carest          | 11    | 0.000655   | 0.023430   | 0.023458

vrest (Rank 2) and dvdtmax (Rank 7) do not both rank at the bottom.

==========================================
=== STEP 5 — FIGURE ===
==========================================
Saved d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\fig_shap_global.pdf (11.4 KB)
Saved d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\fig_shap_global.png (32.9 KB)

==========================================
=== STEP 6 — PERSIST ===
==========================================
Saved CSV: d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\shap_global_importance_rev.csv
Rows: 11
SHA-256: 374DF6711DB892A24CE0855F1FD71139627FB8DED160EA4BE6A701C084ABF26D

==========================================
=== SANITY FLAGS ===
==========================================
  - FLAG: Surprising rank for degenerate feature: vrest Rank 2, dvdtmax Rank 7 despite tiny CV.