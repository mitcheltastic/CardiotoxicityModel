# -*- coding: utf-8 -*-
"""
build_full_report.py
Generates the definitive, comprehensive Capstone Design (CD-4) PDF Report:
"The Evolution of In-Silico Cardiotoxicity Screening: From Multi-Model Exploration
to the Deployed Conformal Random Forest on Hugging Face".

Author: Mitchel Mohamad Affandi (Telkom University)
"""

import os
import sys
import base64
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_b64_img(rel_path):
    full_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(full_path):
        print(f"Warning: image {full_path} not found.")
        return ""
    ext = os.path.splitext(full_path)[1].lstrip('.').lower()
    mime = 'image/png' if ext == 'png' else 'image/jpeg'
    with open(full_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime};base64,{data}"

print("Encoding figure assets to Base64...")
img_fig1 = get_b64_img("paper/paper_figures/fig1.png")
img_arch = get_b64_img("paper/paper_figures/fig_architecture.png")
img_rf_shap = get_b64_img("results/cd4_eval/rf/fig_shap_global_rf.png")
img_stack_shap = get_b64_img("results/cd4_eval/fig_shap_global_fixed.png")
img_macro_f1 = get_b64_img("results/report_assets/chart_macro_f1.png")
img_set_size = get_b64_img("results/report_assets/chart_set_size.png")
img_latency = get_b64_img("results/report_assets/chart_latency.png")

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Capstone Design CD-4 Technical Report: In-Silico Cardiotoxicity Model Evolution</title>
<style>
  @page {
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
    @bottom-left {
      content: "Capstone Design (CD-4) Technical Report | In-Silico Cardiotoxicity Screening";
      font-size: 7.5pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      color: #94a3b8;
    }
    @bottom-right {
      content: "Page " counter(page);
      font-size: 8pt;
      font-weight: 600;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      color: #475569;
    }
  }

  *, *:before, *:after {
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    line-height: 1.5;
    font-size: 9pt;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
  }

  h1, h2, h3, h4 {
    color: #0f172a;
    font-weight: 700;
    margin-top: 1.1em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
  }

  h1 {
    font-size: 19pt;
    line-height: 1.2;
    margin-top: 0;
    border-bottom: 2.5px solid #0284c7;
    padding-bottom: 8px;
  }

  h2 {
    font-size: 12.5pt;
    border-left: 4px solid #0284c7;
    padding-left: 8px;
    margin-top: 1.4em;
    background: #f8fafc;
    padding-top: 4px;
    padding-bottom: 4px;
  }

  h3 {
    font-size: 10.5pt;
    color: #1e3a8a;
    margin-top: 1.1em;
  }

  p {
    margin: 0 0 0.7em 0;
    text-align: justify;
  }

  ul, ol {
    margin: 0 0 0.7em 1.5em;
    padding: 0;
  }

  li {
    margin-bottom: 0.3em;
  }

  .header-block {
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #cbd5e1;
  }

  .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 8pt;
    margin-top: 10px;
  }

  .meta-item strong {
    color: #0f172a;
  }

  .badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7.5pt;
    font-weight: 700;
    text-transform: uppercase;
  }
  .badge-deployed { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
  .badge-rejected { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
  .badge-baseline { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 7.8pt;
    page-break-inside: avoid;
  }

  th, td {
    padding: 4.5px 7px;
    border: 1px solid #cbd5e1;
    text-align: center;
  }

  th {
    background-color: #1e293b;
    color: #ffffff;
    font-weight: 600;
  }

  tr:nth-child(even) {
    background-color: #f8fafc;
  }

  .text-left { text-align: left; }
  .highlight-best { background-color: #dcfce7 !important; font-weight: 700; color: #166534; }
  .highlight-warn { background-color: #fee2e2 !important; font-weight: 700; color: #991b1b; }
  .highlight-meta { background-color: #fef3c7 !important; color: #92400e; }

  .callout {
    padding: 9px 12px;
    margin: 10px 0;
    border-radius: 6px;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }

  .callout-info {
    background-color: #eff6ff;
    border-left: 4px solid #3b82f6;
    color: #1e40af;
  }

  .callout-warning {
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    color: #92400e;
  }

  .callout-danger {
    background-color: #fef2f2;
    border-left: 4px solid #ef4444;
    color: #991b1b;
  }

  .callout-success {
    background-color: #f0fdf4;
    border-left: 4px solid #22c55e;
    color: #166534;
  }

  .figure-container {
    margin: 12px 0;
    text-align: center;
    page-break-inside: avoid;
  }

  .figure-container img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }

  .figure-caption {
    font-size: 7.8pt;
    color: #475569;
    margin-top: 5px;
    font-style: italic;
  }

  .page-break {
    page-break-before: always;
  }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 8px 0;
  }

  .formula-box {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 7px 10px;
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    margin: 6px 0;
  }
</style>
</head>
<body>

<div class="header-block">
  <div style="font-size: 8.5pt; font-weight: 700; color: #0284c7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
    Capstone Design (CD-4) Engineering & Model Evolution Report
  </div>
  <h1>In-Silico Cardiotoxicity Risk Prediction Pipeline</h1>
  <div style="font-size: 10.5pt; color: #475569; font-weight: 500; margin-top: -4px;">
    From Multi-Model Exploration & Stacking Pathologies to the Deployed Conformal Random Forest on Hugging Face
  </div>

  <div class="meta-grid">
    <div class="meta-item"><strong>Author:</strong> Mitchel Mohamad Affandi</div>
    <div class="meta-item"><strong>Institution:</strong> School of Electrical Engineering, Telkom University</div>
    <div class="meta-item"><strong>Target Pathology:</strong> Torsade de Pointes (TdP) Arrhythmia / Sudden Cardiac Death</div>
    <div class="meta-item"><strong>Regulatory Framework:</strong> FDA Comprehensive in vitro Proarrhythmia Assay (CiPA)</div>
    <div class="meta-item"><strong>Production Serving Layer:</strong> Hugging Face Spaces (<code>mitcheltastic/tdp-cipa-screening</code>)</div>
    <div class="meta-item"><strong>Production Architecture:</strong> Standalone Bagged RF (400 trees) + Conformal Safety (Rank 11) + TreeSHAP</div>
  </div>
</div>

<h2>1. Executive Summary & Problem Formulation</h2>

<p>
Assessing drug-induced cardiac toxicity—specifically the risk of triggering <strong>Torsade de Pointes (TdP)</strong>, a lethal polymorphic ventricular arrhythmia—is a mandatory regulatory milestone in safety pharmacology. The international <strong>Comprehensive in vitro Proarrhythmia Assay (CiPA)</strong> initiative established an in-silico screening paradigm that pairs multi-channel ionic inhibition patch-clamp data with biophysical simulations of human ventricular cardiomyocytes (O'Hara-Rudy ventricular model). 
From these simulated cellular action potentials and calcium transients, <strong>11 summary electrophysiological biomarkers</strong> are extracted to classify candidate compounds into three ordinal risk categories: <strong>High</strong>, <strong>Intermediate</strong>, and <strong>Low</strong>.
</p>

<div class="callout callout-danger">
  <strong>The Fundamental Asymmetry of Screening Errors:</strong><br>
  In clinical screening, prediction errors are strictly non-equivalent. <em>Over-calling</em> a benign drug (e.g., labeling Low as Intermediate) incurs modest financial and pipeline delays through secondary confirmatory assays. Conversely, <em>under-triaging</em> a hazardous drug (e.g., labeling High as Intermediate/Low, or Intermediate as Low) is catastrophic: it allows a cardiotoxic compound to advance undetected into human clinical trials, creating fatal patient risk. Standard machine learning classifiers optimize symmetric objectives (accuracy, cross-entropy) and produce uncalibrated point predictions with <strong>zero safety guarantees</strong> against under-triage.
</div>

<p>
This engineering report documents the comprehensive research trajectory that began with extensive multi-model benchmarking (Stacking ensembles, Consensus voting, XGBoost, and Logistic Regression baselines) and culminated in the definitive architecture deployed to production on Hugging Face: a <strong>standalone 400-tree Bagging Random Forest equipped with a Leave-One-Drug-Out (LODO) Cross-Conformal Safety Layer (Rank 11, q_hat=0.533987)</strong>, an automated <strong>Out-of-Distribution (OOD) Abstention safeguard</strong>, and <strong>sub-2 millisecond TreeSHAP interpretability</strong>.
</p>

<div class="figure-container">
  <img src="__IMG_FIG1__" alt="CiPA Pipeline Overview" style="max-height: 220px;">
  <div class="figure-caption"><strong>Figure 1:</strong> End-to-end in-silico cardiotoxicity screening pipeline across 28 reference compounds, 3 patch-clamp platforms, Stratified Grouped CV, classifier candidates, conformal calibration, and safe clinical action mapping.</div>
</div>

<h2>2. Dataset Architecture & Grouped Cross-Validation Protocol</h2>

<p>
The evaluation benchmark utilizes in-silico simulation data derived from the official <strong>CiPA panel of 28 reference drugs</strong>, split strictly at the drug boundary into <strong>12 training/calibration compounds</strong> (balanced: 4 High, 4 Intermediate, 4 Low) and <strong>16 held-out test compounds</strong> (unbalanced: 4 High, 7 Intermediate, 5 Low).
Each compound is simulated across <strong>2,000 virtual cardiomyocytes</strong> to incorporate physiological conductances variability, generating 24,000 training and 32,000 test observations per instrument platform.
</p>

<table>
  <thead>
    <tr>
      <th>Platform Name</th>
      <th>Measurement Technology</th>
      <th>Training Split (Rows / Drugs)</th>
      <th>Testing Split (Rows / Drugs)</th>
      <th>Role in Pipeline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="text-left"><strong>ChanTest</strong></td>
      <td>Automated High-Throughput Patch Clamp</td>
      <td>24,000 rows (12 drugs)</td>
      <td>32,000 rows (16 drugs)</td>
      <td><span class="badge badge-deployed">Primary Training Split</span></td>
    </tr>
    <tr>
      <td class="text-left"><strong>Nanion</strong></td>
      <td>Automated Patch Clamp (SyncroPatch)</td>
      <td>24,000 rows (12 drugs)</td>
      <td>32,000 rows (16 drugs)</td>
      <td><span class="badge badge-baseline">Cross-Platform Test</span></td>
    </tr>
    <tr>
      <td class="text-left"><strong>Manual</strong></td>
      <td>Manual Pipette Patch Clamp (Gold Standard)</td>
      <td>23,975 rows (12 drugs; 25 dropped)</td>
      <td>31,999 rows (16 drugs; 1 dropped)</td>
      <td><span class="badge badge-baseline">Cross-Platform Test</span></td>
    </tr>
  </tbody>
</table>

<div class="callout callout-info">
  <strong>Audit of Upstream Sentinel Values:</strong> Across the entire multi-instrument dataset, exactly 26 corrupted rows containing upstream calculation infinities/NaNs were excluded at load time (recorded in <code>logs/DROPPED_ROWS_REV.csv</code>: 25 in manual training mexiletine, 1 in manual testing ibutilide). ChanTest and Nanion data contain zero exclusions. All 11 biomarkers are pre-normalized by the data provider against drug-free control baselines; therefore, <strong>no artificial scaling (e.g. MinMax or StandardScaler) is applied</strong>, preserving biophysical interpretability.
</div>

<p>
<strong>Compound-Grouped Validation Mandate:</strong> Because 2,000 observations represent replicates of identical compounds under cellular variability, an unconstrained split would place identical drug identities into both train and test folds. To prevent identity memorization, all cross-validation and hyperparameter tuning strictly enforce <strong>StratifiedGroupKFold (k=4, grouped by drug name)</strong>.
</p>

<div class="page-break"></div>

<h2>3. Phase 1: Exploration of Model Families & Architectures</h2>

<p>
To identify the optimal predictive engine, four foundational model families spanning different structural inductive biases were developed and hyperparameter-tuned under the identical 4-fold grouped cross-validation scheme:
</p>

<ol>
  <li><strong>Logistic Regression (Single-Biomarker qNet Baseline):</strong> The historical benchmark proposed by Li et al. (2019) under the CiPA initiative, utilizing only the net repolarization charge metric (C=0.01).</li>
  <li><strong>Logistic Regression (Multivariate 11-Biomarker Baseline):</strong> Multinomial logistic regression incorporating the complete panel of action potential and calcium biomarkers (C=0.01, max_iter=5000).</li>
  <li><strong>Gradient-Boosted Tree Ensemble (XGBoost):</strong> Non-linear gradient boosting using 300 estimators, max depth 10, learning rate 0.1, and histogram-based tree partitioning.</li>
  <li><strong>Bagging Random Forest (RandomForestClassifier):</strong> An ensemble of 400 fully grown de-correlated decision trees with entropy splitting criterion and unconstrained depth (<code>max_depth=None</code>).</li>
  <li><strong>Stacking Ensemble (Level-1 RF + XGBoost, Level-2 Meta-Learner):</strong> Intended as the flagship ensemble, passing out-of-fold probability vectors from RF and XGBoost to a secondary meta-classifier (initially Random Forest, later Logistic Regression).</li>
  <li><strong>Consensus Voting (Soft and Hard Voting):</strong> Combining base learner predictions via unweighted arithmetic probability averaging (Soft Voting) and majority discrete voting with probability tie-breaking (Hard Voting).</li>
</ol>

<div class="figure-container">
  <img src="__IMG_ARCH__" alt="Architecture Diagram" style="max-height: 260px;">
  <div class="figure-caption"><strong>Figure 2:</strong> High-level architectural schematic showing feature ingestion, virtual cell aggregation, model families, meta-learning layers, and downstream decision serving.</div>
</div>

<h2>4. Phase 2: The Stacking Pathology & The Failure of Ensembles</h2>

<p>
In classical machine learning literature, Stacking Ensembles are widely heralded as superior to individual base learners. Consequently, early research in this project heavily prioritized Stacking. However, rigorous auditing under <strong>STEP 12</strong> revealed severe systemic pathologies that led to the complete disqualification of Stacking as a viable production candidate.
</p>

<h3>4.1 The Meta-Learner Override Pathology</h3>
<p>
The core objective of stacking is to let the meta-learner synthesize complementary strengths. However, audit diagnostics revealed that the meta-learner frequently <strong>overrode unanimous, correct consensus between both base learners</strong>, introducing fatal errors.
</p>

<table>
  <thead>
    <tr>
      <th>Compound</th>
      <th>Ground Truth</th>
      <th>Random Forest (Base 1)</th>
      <th>XGBoost (Base 2)</th>
      <th>Stacking Meta-Learner</th>
      <th>Clinical Consequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>disopyramide</code> (ChanTest)</td>
      <td><strong>High</strong></td>
      <td class="highlight-best">High (p=0.685)</td>
      <td class="highlight-best">High (p=0.863)</td>
      <td class="highlight-warn">Intermediate (p=0.496)</td>
      <td class="highlight-warn">FATAL UNDER-TRIAGE</td>
    </tr>
    <tr>
      <td><code>disopyramide</code> (Nanion)</td>
      <td><strong>High</strong></td>
      <td class="highlight-best">High (p=0.754)</td>
      <td class="highlight-best">High (p=0.814)</td>
      <td class="highlight-warn">Intermediate (p=0.711)</td>
      <td class="highlight-warn">FATAL UNDER-TRIAGE</td>
    </tr>
    <tr>
      <td><code>loratadine</code> (ChanTest)</td>
      <td><strong>Low</strong></td>
      <td class="highlight-best">Low (p=0.922)</td>
      <td class="highlight-best">Low (p=0.977)</td>
      <td class="highlight-warn">Intermediate (p=0.629)</td>
      <td>Unnecessary Over-call</td>
    </tr>
    <tr>
      <td><code>loratadine</code> (Nanion)</td>
      <td><strong>Low</strong></td>
      <td class="highlight-best">Low (p=0.930)</td>
      <td class="highlight-best">Low (p=0.977)</td>
      <td class="highlight-warn">Intermediate (p=0.609)</td>
      <td>Unnecessary Over-call</td>
    </tr>
  </tbody>
</table>

<div class="callout callout-danger">
  <strong>Key Finding from Step 12:</strong> On the held-out ChanTest test set, Random Forest and XGBoost agreed unanimously on <strong>15 out of 16 drugs</strong>. Yet on <code>disopyramide</code> (a known hazardous antiarrhythmic drug), where RF predicted High (68.5%) and XGBoost predicted High (86.3%), the Stacking meta-learner overrode both models and assigned Intermediate (49.6%). <strong>The Stacking model directly manufactured an under-triage that neither base learner committed!</strong>
</div>

<h3>4.2 Cross-Platform Degradation</h3>
<p>
When transferred across experimental instruments without retraining, Stacking collapsed. On Nanion, Stacking Macro-F1 fell to an unacceptable <strong>0.6210</strong> (10/16 correct, 1 under-triage), whereas standalone Random Forest maintained <strong>0.8194</strong> (13/16 correct, 0 under-triage).
</p>

<h3>4.3 Consensus & Voting Analysis (Task 4)</h3>
<p>
We subsequently investigated whether simple Voting could resolve this issue. In an ensemble of two base learners with probability tie-breaking, hard voting is mathematically equivalent to soft voting. While soft voting prevented the meta-learner override, it diluted Random Forest's clean probability calibration by averaging in XGBoost's erratic probability peaks, resulting in wider prediction sets when wrapped with conformal prediction.
</p>

<div class="page-break"></div>

<h2>5. Phase 3: The Conformal Safety Layer</h2>

<p>
To permanently resolve the danger of under-triage across all candidate models, we engineered a distribution-free <strong>Cross-Conformal Prediction Layer</strong> calibrated directly at the drug level.
</p>

<h3>5.1 Mathematical Formulation</h3>
<p>
For each candidate compound d, cell-level probabilities p_i from the 2,000 simulated cells are averaged to yield a compound-level consensus probability vector p_bar_d = (1/2000) sum(p_i). For a calibration drug d with true class y_d, the nonconformity score s_d is:
</p>
<div class="formula-box">
  s_d = 1 - p_bar_d[y_d]  in [0, 1]
</div>
<p>
Calibration follows a strict <strong>Leave-One-Drug-Out (LODO)</strong> scheme over the 12 training drugs: the classifier is iteratively refitted on 11 drugs and evaluated on the 12th held-out drug. The resulting 12 nonconformity scores are sorted in ascending order s_(1) &le; s_(2) &le; ... &le; s_(12). The conformal quantile q_hat is selected at rank k in {1, ..., 12}.
At inference time, a risk category c in {High, Intermediate, Low} enters the prediction set C(d) if its compound-level probability satisfies:
</p>
<div class="formula-box">
  C(d) = { c in Y : p_bar_d[c] &ge; 1 - q_hat }
</div>
<p>
The final <strong>Clinical Recommended Action</strong> is mapped to the <strong>most severe class present in C(d)</strong> (High > Intermediate > Low).
</p>

<h3>5.2 Uniform Operating Point: Rank 11 Selection</h3>
<p>
A systematic quantile sweep (Ranks 12 down to 8) established that <strong>Rank 11</strong> (k=11, alpha = 2/13 &approx; 0.1538, nominal coverage 1 - alpha = 11/13 &approx; 84.62%) is the <strong>tightest universal operating point</strong> that guarantees <strong>ZERO under-triage and ZERO abstention</strong> across all models and all three patch-clamp platforms simultaneously.
</p>

<div class="figure-container">
  <img src="__IMG_SET_SIZE__" alt="Conformal Prediction Set Size" style="max-height: 210px;">
  <div class="figure-caption"><strong>Figure 3:</strong> Mean conformal prediction set size across models at Rank 11. Random Forest achieves near-perfect singleton sets (~1.00), incurring zero cost for conformal safety, while Logistic Regression expands to >1.5 categories.</div>
</div>

<h2>6. Phase 4: Verification of the Deployed Hugging Face Model</h2>

<div class="callout callout-success">
  <strong>EXPLICIT AUDIT VERIFICATION OF THE HUGGING FACE REPOSITORY:</strong><br>
  We performed a line-by-line inspection of <code>hf_space/app.py</code> and deserialized the production bundle <code>hf_space/deployable_chantest_rf.joblib</code>. The findings confirm:
  <ul>
    <li><strong>Model Engine:</strong> <code>sklearn.ensemble.RandomForestClassifier</code> (400 trees, <code>criterion='entropy'</code>, <code>max_depth=None</code>, <code>random_state=42</code>).</li>
    <li><strong>Conformal Layer Active:</strong> <strong>YES, FULLY INTEGRATED.</strong> Calibrated at Rank 11 with q_hat = 0.533987, alpha = 2/13 &approx; 0.1538, and inclusion threshold 1 - q_hat = 0.466013.</li>
    <li><strong>Severity Logic:</strong> Clinical action maps to the most severe category in the prediction set (<code>High > Intermediate > Low</code>).</li>
    <li><strong>CD-4 Rev2 Abstention Safeguard:</strong> If no class meets the 0.466013 threshold (an empty set indicating an out-of-distribution or corrupted biomarker profile), the app automatically flags <code>out_of_distribution: True</code>, abstains from a confident classification, and triggers an alert recommending <strong>High Risk Clinical Monitoring</strong>.</li>
  </ul>
</div>

<div class="page-break"></div>

<h2>7. Phase 5: The Showdown — Why Random Forest Was Chosen</h2>

<p>
The complete unified benchmark comparing all candidate architectures on the 16 held-out test drugs across all three physical assay platforms is summarized below:
</p>

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Architecture</th>
      <th rowspan="2">Platform</th>
      <th colspan="3">Hard Point Predictions</th>
      <th colspan="4">Conformal Safety Layer (Rank 11)</th>
      <th rowspan="2">Inference Latency</th>
    </tr>
    <tr>
      <th>Macro-F1</th>
      <th>Correct</th>
      <th>Under</th>
      <th>Correct</th>
      <th>Under</th>
      <th>Set Size</th>
      <th>Emp. Cov.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>Random Forest<br>(DEPLOYED MODEL)</strong></td>
      <td>ChanTest</td>
      <td class="highlight-best">0.8778</td>
      <td class="highlight-best">14 / 16</td>
      <td class="highlight-best">0</td>
      <td class="highlight-best">14 / 16</td>
      <td class="highlight-best">0</td>
      <td class="highlight-best">1.00</td>
      <td>87.5%</td>
      <td rowspan="3" class="highlight-best"><strong>1.37 ms</strong><br>(TreeSHAP)</td>
    </tr>
    <tr>
      <td>Nanion</td>
      <td class="highlight-best">0.8194</td>
      <td class="highlight-best">13 / 16</td>
      <td class="highlight-best">0</td>
      <td class="highlight-best">13 / 16</td>
      <td class="highlight-best">0</td>
      <td class="highlight-best">1.00</td>
      <td>81.2%</td>
    </tr>
    <tr>
      <td>Manual</td>
      <td>0.7556</td>
      <td>12 / 16</td>
      <td class="highlight-warn">1</td>
      <td class="highlight-best">13 / 16</td>
      <td class="highlight-best">0</td>
      <td class="highlight-best">1.06</td>
      <td>81.2%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Stacking Ensemble<br>(REJECTED)</strong></td>
      <td>ChanTest</td>
      <td>0.6944</td>
      <td>11 / 16</td>
      <td class="highlight-warn">1</td>
      <td>11 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.12</td>
      <td>75.0%</td>
      <td rowspan="3" class="highlight-warn"><strong>136.97 ms</strong><br>(Kernel 15BG)</td>
    </tr>
    <tr>
      <td>Nanion</td>
      <td>0.6210</td>
      <td>10 / 16</td>
      <td class="highlight-warn">1</td>
      <td>11 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.12</td>
      <td>75.0%</td>
    </tr>
    <tr>
      <td>Manual</td>
      <td>0.7548</td>
      <td>12 / 16</td>
      <td class="highlight-best">0</td>
      <td>11 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.38</td>
      <td>87.5%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>XGBoost<br>(Gradient Boosted)</strong></td>
      <td>ChanTest</td>
      <td>0.8194</td>
      <td>13 / 16</td>
      <td class="highlight-best">0</td>
      <td>12 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.25</td>
      <td>87.5%</td>
      <td rowspan="3"><strong>2.85 ms</strong><br>(TreeSHAP)</td>
    </tr>
    <tr>
      <td>Nanion</td>
      <td>0.7548</td>
      <td>12 / 16</td>
      <td class="highlight-best">0</td>
      <td>12 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.31</td>
      <td>87.5%</td>
    </tr>
    <tr>
      <td>Manual</td>
      <td class="highlight-best">0.8194</td>
      <td>13 / 16</td>
      <td class="highlight-best">0</td>
      <td>11 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.31</td>
      <td>81.2%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>LR Multivariate<br>(11 Biomarkers)</strong></td>
      <td>ChanTest</td>
      <td>0.8194</td>
      <td>13 / 16</td>
      <td class="highlight-best">0</td>
      <td>10 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.38</td>
      <td>87.5%</td>
      <td rowspan="3"><strong>0.45 ms</strong><br>(Linear)</td>
    </tr>
    <tr>
      <td>Nanion</td>
      <td>0.7548</td>
      <td>12 / 16</td>
      <td class="highlight-best">0</td>
      <td>11 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.38</td>
      <td>87.5%</td>
    </tr>
    <tr>
      <td>Manual</td>
      <td class="highlight-best">0.8194</td>
      <td>13 / 16</td>
      <td class="highlight-best">0</td>
      <td>11 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.31</td>
      <td>81.2%</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>LR qNet-only<br>(CiPA Reference)</strong></td>
      <td>ChanTest</td>
      <td>0.8120</td>
      <td>13 / 16</td>
      <td class="highlight-warn">1</td>
      <td>8 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.56</td>
      <td>87.5%</td>
      <td rowspan="3"><strong>0.30 ms</strong><br>(Linear)</td>
    </tr>
    <tr>
      <td>Nanion</td>
      <td>0.7566</td>
      <td>12 / 16</td>
      <td class="highlight-warn">1</td>
      <td>9 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.56</td>
      <td>87.5%</td>
    </tr>
    <tr>
      <td>Manual</td>
      <td>0.7515</td>
      <td>12 / 16</td>
      <td class="highlight-warn">1</td>
      <td>9 / 16</td>
      <td class="highlight-best">0</td>
      <td>1.62</td>
      <td>87.5%</td>
    </tr>
  </tbody>
</table>

<div class="two-col">
  <div class="figure-container">
    <img src="__IMG_MACRO_F1__" alt="Macro F1 Comparison">
    <div class="figure-caption"><strong>Figure 4:</strong> Drug Macro-F1 across platforms. Random Forest dominates on automated patch clamp (ChanTest: 0.878, Nanion: 0.819).</div>
  </div>
  <div class="figure-container">
    <img src="__IMG_LATENCY__" alt="Latency Comparison">
    <div class="figure-caption"><strong>Figure 5:</strong> SHAP Explanation Latency. TreeSHAP on RF (1.37 ms) is 100x faster than KernelExplainer on Stacking (136.97 ms).</div>
  </div>
</div>

<h3>The Five Pillars of the Random Forest Selection</h3>

<ol>
  <li><strong>Unmatched Predictive Accuracy:</strong> Random Forest achieves the highest Macro-F1 on ChanTest (0.8778) and Nanion (0.8194), outperforming Stacking by +0.183 and +0.198 points respectively.</li>
  <li><strong>Zero Cost for Conformal Safety:</strong> While every model achieves 0 under-triage at Rank 11, Random Forest pays the lowest penalty. Its prediction set size is <strong>1.00 on ChanTest and Nanion</strong>, and <strong>1.06 on Manual</strong>. For RF, 14 of 16 drugs remain unambiguous singletons, whereas LR qNet dilates to 1.62 categories (calling almost 2 categories per drug).</li>
  <li><strong>Defeating the Meta-Learner Pathology:</strong> Operating as a single bagging ensemble eliminates the meta-learner override pathology. RF correctly classifies <code>disopyramide</code> as High risk across all three platforms without artificial downgrade.</li>
  <li><strong>Sub-2 Millisecond TreeSHAP Explanations:</strong> Because Stacking contains heterogeneous estimators and a meta-classifier, it cannot use tree algorithms and requires <code>KernelExplainer</code>. Even when compressed to 15 background centroids, KernelExplainer takes 136.97 ms (and up to 4.2 seconds with larger sample backgrounds). In contrast, Random Forest leverages native <code>TreeExplainer</code>, computing exact Shapley attributions in <strong>1.37 ms</strong> (>100x speedup), enabling instant real-time UI feedback.</li>
  <li><strong>Architectural Simplicity & Reliability:</strong> Deploying a single self-contained serialized forest (2.28 MB) eliminates multi-model dependency desynchronization, reduces cold-start memory overhead, and guarantees deterministic execution in containerized cloud environments.</li>
</ol>

<div class="page-break"></div>

<h2>8. Phase 6: Mechanistic Interpretability & Global Feature Importance</h2>

<p>
To ensure regulatory transparency, global biomarker attribution was audited using TreeSHAP on the deployed Random Forest model and compared against the baseline KernelExplainer on Stacking.
</p>

<div class="two-col">
  <div class="figure-container">
    <img src="__IMG_RF_SHAP__" alt="RF Global SHAP">
    <div class="figure-caption"><strong>Figure 6:</strong> Global TreeSHAP feature importance for the Deployed Random Forest model on the 16 held-out test compounds.</div>
  </div>
  <div class="figure-container">
    <img src="__IMG_STACK_SHAP__" alt="Stacking Global SHAP">
    <div class="figure-caption"><strong>Figure 7:</strong> Global KernelExplainer SHAP attribution for the legacy Stacking Classifier.</div>
  </div>
</div>

<div class="callout callout-info">
  <strong>Biological Consistency:</strong> In both models, <strong>qNet (net repolarization charge)</strong> emerges as the primary discriminative biomarker by a wide margin (mean |SHAP| = 1.09), perfectly aligning with cardiac electrophysiology literature (Lancaster & Sobie, 2016; Li et al., 2019). Action potential repolarization biomarkers (APD90, APD50) and calcium transient dynamics (CaTD50) provide secondary refinement, capturing multi-channel block effects that single-channel assays fail to register.
</div>

<h2>9. Phase 7: Production Hugging Face Space Implementation</h2>

<p>
The final production architecture is deployed on Hugging Face Spaces (<code>mitcheltastic/tdp-cipa-screening</code>) as a standalone containerized Gradio application with dual capabilities: rapid risk triage and full mechanistic biophysical simulation.
</p>

<div class="figure-container" style="text-align: left; background: #0f172a; color: #f8fafc; padding: 12px 16px; border-radius: 6px; font-family: monospace; font-size: 7.5pt; line-height: 1.45; overflow-x: auto;">
  <span style="color: #38bdf8;"># Production Ingestion & Serving Architecture (hf_space/app.py)</span><br>
  <strong>Input:</strong> 11 Pre-normalized Biomarkers (qNet, dvdtmax, vmax, vrest, APD50, APD90, max_dv, camax, carest, CaTD50, CaTD90)<br>
  &nbsp;&nbsp;│<br>
  &nbsp;&nbsp;├──► <strong>RandomForestClassifier</strong> (400 trees, entropy) ──► Soft Consensus Probabilities: [p_High, p_Int, p_Low]<br>
  &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
  &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──► <strong>Conformal Safety Layer</strong> (Rank 11, Threshold: 1 - 0.533987 = 0.466013)<br>
  &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──► Prediction Set: C(d) = {{ c : p[c] >= 0.466013 }}<br>
  &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──► <em>If C(d) is Empty:</em> <strong>OOD Abstain Triggered</strong> ──► Recommended Action: <strong>HIGH (Conservative Monitoring)</strong><br>
  &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──► <em>If C(d) is Non-Empty:</em> Recommended Action = max_severity(C(d))<br>
  &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
  &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──► <strong>TreeExplainer</strong> (Exact Shapley Traversal) ──► Sub-2ms Local Attribution Array<br>
  &nbsp;&nbsp;│<br>
  &nbsp;&nbsp;└──► <strong>FDA/ORd Biophysical Mechanistic Simulation Engine</strong> (Optional Add-on via C++ ode15s / sim_service.py)<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Converts raw ion-channel IC50/Hill parameters & Cmax into the 11 cellular biomarkers (averaged over 1x-4x Cmax)
</div>

<h3>API Endpoints for Client Integration</h3>
<ul>
  <li><code>POST /gradio_api/call/predict</code>: Returns class probabilities, conformal prediction set, recommended clinical action, ambiguity flag, and out-of-distribution alert.</li>
  <li><code>POST /gradio_api/call/explain</code>: Returns predicted class, expected base value, and per-biomarker Shapley contribution values in 1.37 ms.</li>
  <li><code>POST /gradio_api/call/simulate</code>: Isolated CiPA ODE simulation endpoint running FDA ORd biophysical model on demand.</li>
</ul>

<h2>10. Conclusion & Architectural Decision Matrix</h2>

<table>
  <thead>
    <tr>
      <th>Evaluation Dimension</th>
      <th>Stacking Ensemble</th>
      <th>XGBoost</th>
      <th>Multivariate LR</th>
      <th>Random Forest (DEPLOYED)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="text-left"><strong>ChanTest Macro-F1</strong></td>
      <td>0.6944 (Poor)</td>
      <td>0.8194 (Strong)</td>
      <td>0.8194 (Strong)</td>
      <td class="highlight-best">0.8778 (Highest)</td>
    </tr>
    <tr>
      <td class="text-left"><strong>Nanion Generalization</strong></td>
      <td class="highlight-warn">0.6210 (Severely degraded)</td>
      <td>0.7548 (Moderate)</td>
      <td>0.7548 (Moderate)</td>
      <td class="highlight-best">0.8194 (Superior)</td>
    </tr>
    <tr>
      <td class="text-left"><strong>Meta-Learner Override</strong></td>
      <td class="highlight-warn">Present (Under-triaged disopyramide)</td>
      <td>None (Single model)</td>
      <td>None (Single model)</td>
      <td class="highlight-best">None (Single model)</td>
    </tr>
    <tr>
      <td class="text-left"><strong>Conformal Set Size Cost</strong></td>
      <td>1.12 - 1.38 (Moderate expansion)</td>
      <td>1.25 - 1.31 (Noticeable expansion)</td>
      <td>1.31 - 1.38 (High expansion)</td>
      <td class="highlight-best">1.00 - 1.06 (Nearly Free)</td>
    </tr>
    <tr>
      <td class="text-left"><strong>Under-Triage Rate (Rank 11)</strong></td>
      <td class="highlight-best">0.0% (Zero)</td>
      <td class="highlight-best">0.0% (Zero)</td>
      <td class="highlight-best">0.0% (Zero)</td>
      <td class="highlight-best">0.0% (Zero)</td>
    </tr>
    <tr>
      <td class="text-left"><strong>SHAP Explanation Latency</strong></td>
      <td class="highlight-warn">136.97 ms (KernelExplainer)</td>
      <td>2.85 ms (TreeSHAP)</td>
      <td>0.45 ms (Analytical)</td>
      <td class="highlight-best">1.37 ms (TreeSHAP)</td>
    </tr>
    <tr>
      <td class="text-left"><strong>Deployment Complexity</strong></td>
      <td class="highlight-warn">High (3 models, custom wrappers)</td>
      <td>Medium (C++ runtime)</td>
      <td>Low (Linear weights)</td>
      <td class="highlight-best">Low (Single pickle, robust)</td>
    </tr>
  </tbody>
</table>

<p>
<strong>Final Synthesis:</strong> The standalone <strong>400-tree Bagging Random Forest</strong> equipped with a <strong>Rank-11 Conformal Safety Layer</strong> was selected for production because it decisively resolves the clinical under-triage hazard without sacrificing predictive precision, avoids the catastrophic override failures of stacking meta-learners, preserves near-perfect singleton prediction sets (1.00 - 1.06), and provides sub-2 millisecond exact TreeSHAP interpretability for real-time regulatory and clinical safety workflows.
</p>

</body>
</html>
"""

html_content = html_template.replace("__IMG_FIG1__", img_fig1)
html_content = html_content.replace("__IMG_ARCH__", img_arch)
html_content = html_content.replace("__IMG_RF_SHAP__", img_rf_shap)
html_content = html_content.replace("__IMG_STACK_SHAP__", img_stack_shap)
html_content = html_content.replace("__IMG_MACRO_F1__", img_macro_f1)
html_content = html_content.replace("__IMG_SET_SIZE__", img_set_size)
html_content = html_content.replace("__IMG_LATENCY__", img_latency)


html_path = os.path.join(BASE_DIR, "report_temp.html")
pdf_path = os.path.join(BASE_DIR, "CD4_Comprehensive_Model_Selection_Report.pdf")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Wrote report HTML to {html_path} ({len(html_content)} bytes)")

# Convert to PDF via headless Chrome
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
temp_profile = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "chrome_pdf_report_profile")
temp_pdf = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "report_output.pdf")

os.makedirs(temp_profile, exist_ok=True)
if os.path.exists(temp_pdf):
    os.remove(temp_pdf)

cmd = [
    chrome_path,
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--user-data-dir={temp_profile}",
    f"--print-to-pdf={temp_pdf}",
    html_path
]

print("Executing Chrome headless PDF conversion...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Chrome returncode:", res.returncode)
if res.stderr:
    print("Chrome stderr:", res.stderr)

if os.path.exists(temp_pdf):
    shutil.copyfile(temp_pdf, pdf_path)
    print(f"SUCCESS: Generated PDF at {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    # Cleanup temp
    try:
        os.remove(temp_pdf)
        shutil.rmtree(temp_profile, ignore_errors=True)
        os.remove(html_path)
    except Exception as e:
        print("Cleanup note:", e)
else:
    print("ERROR: temp_pdf was not created!")
    sys.exit(1)
