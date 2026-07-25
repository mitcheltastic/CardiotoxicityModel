> [WARNING] STALE - OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

> ⚠️ STALE — OLD DATA. All numbers in this file were computed on the pre-`_rev` datasets and are
> SUPERSEDED. Do not cite, quote, or reuse any figure from this document in the ICCEREC 2026
> paper. Authoritative numbers come from the `_rev` re-run only.

# SCIENCE_CONTEXT.md — Scientific Context, Literature, & Methodological Narrative

This document synthesizes the scientific context, research framing, prior literature, dataset specifications, methodological narrative, discussion claims, and examiner-risk considerations directly from the repository's documentation files (`PROJECT_SUMMARY.md`, `README.md`, `context.md`, `context_model.md`, and `plus_ultra.ipynb` markdown cells). All claims, citations, and numbers are strictly sourced from these files.

---

## 1. Problem Framing & Novelty

### Stated Research Problem & Gap
* **Problem Framing:** "In-silico cardiotoxicity (Torsade de Pointes / TdP) risk prediction. Given 11 CiPA cardiac biomarkers for a drug, classify it into **high / intermediate / low** TdP risk." (`context_model.md` §1)
* **Research Gap:** Developing a robust, explainable, in-silico cardiotoxicity prediction pipeline ready for integration into a clinical screening web tool (`PROJECT_SUMMARY.md` §1), while identifying and correcting severe data leakage inherent in simulated virtual-population replicate datasets (`PROJECT_SUMMARY.md` §2.5, `README.md`).

### Explicit Novelty & Contribution Claims
1. **Leakage-Free CV Methodology (The Crown Jewel):** "The leakage confirmed as drug-level memorization (since the 2,000 samples per drug share identical drug-level parameters). To fix this, all base models were re-tuned... using a leakage-free `StratifiedGroupKFold(n_splits=4)` cross-validation scheme, grouping by `drug_name` so that whole drugs are held out in each fold. This represents the key methodological contribution of the project." (`PROJECT_SUMMARY.md` §3, §2.5; `context_model.md` §2)
2. **Drug-Level Split-Conformal Prediction Layer:** "drug-level split-conformal prediction... eliminates the sole under-triaged drug (clarithromycin on manual) at a cost of +0.0625 average set size; zero under-triaged drugs across all three datasets." (`context_model.md` §5)

### Clinical & Regulatory Motivation
* **CiPA & FDA Context:** Aligned with the Comprehensive In Vitro Proarrhythmia Assay (**CiPA**) initiative led by the FDA and steering committee to modernize cardiac safety paradigms beyond simple hERG blockade (`PROJECT_SUMMARY.md` §1, §6).
* **TdP Risk Stratification:** Classifies drugs into three clinical risk tiers (**high**, **intermediate**, **low**) based on cellular action potential simulations (`PROJECT_SUMMARY.md` §1).
* **Zero Critical Misses Safety Goal:** Achieve zero critical misses, defined as predicting a high-risk drug as low-risk (verified: 0 of 4 high-risk test drugs predicted low across all aggregation methods) (`context_model.md` §3, `PROJECT_SUMMARY.md` §4).

---

## 2. Prior Work / References

Every cited reference appearing in the project documentation:

| Citation / Reference | Scientific Scope & Context in Project | Project Relation & Differences | Quoted Baseline / Metric |
| :--- | :--- | :--- | :--- |
| **Fuadah et al. 2024**<br>(*CPT Pharmacometrics Syst. Pharmacol.*, 13(12):2159–2170) | Primary benchmark paper defining the 12-training / 16-testing drug partition and the 3-ANN stacking architecture (`context.md` §2, §7, §15). | **THE paper to debug against.** This project replicates Fuadah's 3-ANN stacking baseline ("Paper-faithful Stacking"), but discovers and fixes drug-level CV leakage, extends evaluation to 612 configs, and adds conformal prediction. | **Claimed 0.94 intermediate AUC** (`context.md` §12, §15).<br>*(Paper-faithful LR stack benchmark: 0.820 macro-F1, 0.646 intermediate recall; `context.md` §7).* |
| **Fuadah et al. 2023**<br>(*Frontiers Physiol.*, 14:1266084) | Earlier Random Forest cardiotoxicity prediction paper from the same research group (`context.md` §15). | Cited as inspiration for incorporating inter-individual / population cell model variability into future work (`context_model.md` §6). | NOT FOUND (no specific baseline numbers quoted from the 2023 paper). |
| **Li et al. 2019**<br>(*Clin. Pharmacol. Ther.*, 105(2):466–475) | Established the standard CiPA 10,000-iteration bootstrap resampling validation protocol (`context.md` §8, §15; `PROJECT_SUMMARY.md` §3). | Adopted for Phase 4 bootstrap validation (drawing 1 sample per test drug per iteration over 10,000 iterations to yield 95% CIs) (`context.md` §8). | Defines CiPA tier criteria: AUC (Excel $\ge0.90$, Good $\ge0.80$, Acc $\ge0.70$), LR+ ($\ge10.0, \ge5.0, \ge2.0$), LR- ($\le0.10, \le0.20, \le0.50$) (`context.md` §8). |
| **Pramudito et al. 2024**<br>(*Sci. Reports*, 14:24045) | Reference for SHAP biomarker feature importance ranking in cardiac safety (`context.md` §15). | Validates SHAP explainability methodology for ranking CiPA biomarkers (`context.md` §15). | NOT FOUND (no specific numeric benchmark quoted). |
| **Dutta et al. 2017**<br>(*Frontiers Physiol.*, 8:616) | Published the ORd-CiPAv1.0 (O'Hara-Rudy) human ventricular cardiomyocyte model (`context.md` §15). | Source of the underlying cell model simulation lineage producing the biomarker datasets (`context_model.md` §4, `context.md` §15). | NOT FOUND. |
| **Jeong et al.** | Reference for ordinal modeling in cardiac risk classification (`context_model.md` §6). | Cited as an untested future idea for ordinal classifier loss functions (used ordinal logistic regression) (`context_model.md` §6). | NOT FOUND. |

---

## 3. The Three Datasets — Scientific Description

### Measurement Platform & Dataset Sources
1. **`chantest`:** Primary reference dataset and deployment target; derived from automated patch-clamp electrophysiology recordings/simulations (`PROJECT_SUMMARY.md` §2).
2. **`nanion`:** High-throughput automated patch-clamp platform dataset (`PROJECT_SUMMARY.md` §2).
3. **`manual`:** Dataset compiled from manual patch-clamp electrophysiology measurements (`PROJECT_SUMMARY.md` §2).

### Biomarker Derivation & Simulation Model
* Derived from cellular action potential simulations using the **ORd** (O'Hara-Rudy) and **ToR-ORd** ventricular cardiomyocyte cell model lineages (`context_model.md` §4, `context.md` §2, §15).
* **11 Canonical CiPA Biomarkers:** `qNet` (net charge), `dvdtmax` ($dV_m/dt_{\max}$), `vmax` ($V_{m,\text{peak}}$), `vrest` ($V_{m,\text{rest}}$), `APD50` (APD$_{50}$), `APD90` (APD$_{90}$), `max_dv` ($dV_m/dt_{\text{repol}}$), `camax` (Ca$_{\text{peak}}$), `carest` (Ca$_{\text{rest}}$), `CaTD50` (CaD$_{50}$), `CaTD90` (CaD$_{90}$) (`PROJECT_SUMMARY.md` §2, `context.md` §3).

### Drug Lists per Dataset
* **Partition Size:** 12 unique training drugs, 16 unique testing drugs (completely disjoint, zero drug overlap) (`PROJECT_SUMMARY.md` §2).
* **Sample Count:** 2,000 simulated virtual-population replicate samples per drug (24,000 train / 32,000 test rows per dataset) (`PROJECT_SUMMARY.md` §2).
* **Specific Drug Names Explicitly Mentioned in Code/Docs:**
  * *Training drugs named:* `astemizole` (`context_model.md` §6).
  * *Testing drugs named:* `azimilide` (High), `domperidone` (Int), `pimozide` (Int), `clarithromycin` (Int), `loratadine` (Low), `diltiazem` (Low/Outlier), `tamoxifen` (Int) (`context_model.md` §3, §4, §5; `hf_space/app.py`).
  * *Complete 28-Drug Name List:* **NOT FOUND as a single unified text list in markdown documentation**, though exact counts are verified as 12 train / 16 test drugs across all 3 datasets.

### Scientific Purpose of Cross-Platform Evaluation
* Evaluates model robustness against **experimental domain shift** (automated patch-clamp vs. manual patch-clamp) (`context_model.md` §3, `context.md` §2).
* Demonstrates that applying the `chantest`-trained model with its `chantest`-fit scaler unchanged to `nanion` and `manual` maintains drug-level macro-F1 (0.878 $\to$ 0.819 $\to$ 0.756) without distribution breakdown, as qNet scaled means remain stable ($-0.4291 / -0.4505 / -0.4597$) (`context_model.md` §3).

---

## 4. Methodology Narrative

### The Leak-Free CV Story (in Project's Own Words)
* **The Discovery:** "In Phase 2, initial cross-validation using standard row-level `StratifiedKFold` yielded near-perfect validation Macro-F1 scores ($\approx 0.99$). However, Phase 2.5 diagnostic testing revealed that the models had memorized drug-level features... row-level CV (`StratifiedKFold`): **0.999542** Macro-F1 vs. drug-level CV (`GroupKFold`): **0.460310** Macro-F1." (`PROJECT_SUMMARY.md` §2.5)
* **The Mechanism:** "Since each drug contains 2,000 highly correlated simulated samples, a standard row-level split... scatters samples of the same drug across both training and validation folds. This allows classifiers to memorize drug-specific signatures and yields overly optimistic validation scores." (`plus_ultra.ipynb` Cell 21)
* **The Fix:** "Re-tuned everything with `StratifiedGroupKFold(n_splits=4)` grouped by `drug_name`... holding whole drugs out per fold." (`PROJECT_SUMMARY.md` §2.5; `context_model.md` §2)

### Reasoning for Stacking over Voting, and Pure-ML over Mixed
* **Stacking vs. Voting:** "Stacking ensembles consistently outperformed soft/hard voting ensembles (best stacking F1 was **0.828004** vs. best voting F1 of **0.821999**). Meta-learning architecture `meta=RF` won Rank 1 across all three datasets." (`PROJECT_SUMMARY.md` §4)
* **Pure-ML vs. Mixed ML+DL:** "The pure-ML `RF+XGB` stacking model was chosen as the deployable model. It offers superior explainability, lower latency, and structural simplicity (no deep learning wrappers required)" (`PROJECT_SUMMARY.md` §5). Furthermore, pure-ML `RF+XGB` (0.828) strictly beat mixed ML+DL (`mixed-all-7`: 0.817) under leak-free Protocol A (`context_model.md` §1).

### Conformal Prediction Motivation
* **Clinical Safety Problem:** Single point predictions (argmax) can produce dangerous under-triage errors on borderline compounds (e.g., `clarithromycin` on manual predicted Low instead of Intermediate) (`context_model.md` §3, §5).
* **Conformal Solution:** Conformal prediction constructs a prediction set of plausible classes. Under the clinical rule where action equals the most severe class in the set, conformal prediction guarantees zero under-triaged drugs across all datasets at a minor cost of $+0.0625$ set size (`context_model.md` §5).

---

## 5. Discussion-Level Claims & Limitations

### Stated Limitations & Threats to Validity
1. **Intermediate Tier Hardest:** Sample-level intermediate recall of 0.660 is low, though at the drug level errors are few (5/7 correct) and safe-direction (`context_model.md` §3, §4 #1).
2. **Wide Confidence Interval:** 95% CI of 0.819 [0.694, 0.878] due to small test set size (16 drugs). `chantest` CI overlaps `nanion` (`context_model.md` §4 #2).
3. **Trained on One Instrument:** Deployed model trained exclusively on `chantest` 12 training drugs (`context_model.md` §4 #3).
4. **Single Cell Model Lineage:** Derived from one ventricular cell model lineage (ORd/ToR-ORd) without inter-cell-model variability (`context_model.md` §4 #4).
5. **Deployment Dependency:** Unpickling requires `deploy_utils.py` custom wrappers (`context_model.md` §4 #6, `PROJECT_SUMMARY.md` §7).
6. **Selection Optimism:** 612-config sweep selected winning config via Protocol A test-set evaluation; reported test metrics carry selection optimism requiring nested CV (`context_model.md` §4 #7).
7. **Literature-Borderline Disagreements:** `domperidone` and `pimozide` are over-called High at $P(High) \approx 0.93 - 1.00$, representing label-boundary/literature disagreement rather than model uncertainty (`context_model.md` §4 #8).
8. **Conformal Small-Calibration-Set Limit & Domain Shift:** $\alpha$ quantized in steps of $1/13 \approx 0.077$; $\alpha=0.077$ is degenerate due to outlier `diltiazem`. Empirical coverage drops on domain shift (0.875 $\to$ 0.8125 vs nominal 0.846) (`context_model.md` §5).

### Interpretive Claims (Clinical Relevance)
* Model achieved **zero critical misses** on high-risk test drugs (0 of 4 high-risk drugs predicted low) (`context_model.md` §3, `PROJECT_SUMMARY.md` §4).
* Conformal prediction provides a distribution-free safety mechanism preventing unsafe under-triage (`context_model.md` §5).

### SHAP Interpretation Narrative
* **`qNet` #1 (Mean $|SHAP| = 0.1618$):** Primary FDA/CiPA biomarker integrating multi-channel blockade (IKr, ICaL, INaL currents), proving the model learned true cardiac electrophysiology (`PROJECT_SUMMARY.md` §6, `context_model.md` §3).
* **`vrest` #2 (Mean $|SHAP| = 0.0839$):** Resting membrane potential ranks #2. Affects sodium channel availability and cell excitability; noted as mildly unexpected vs classic repolarization markers (`APD90`), representing a dataset-specific signal or excitability driver for thesis discussion (`PROJECT_SUMMARY.md` §5, §6; `context_model.md` §4 #5).

---

## 6. Known Weaknesses & Examiner-Risk Items

Items explicitly flagged as weak spots, reviewer concerns, or defense topics:

1. **ML/DL Paradigm Mixing (Sempro Feedback):** "Examiners want either pure ML ensemble OR pure DL ensemble, not mixed" (`context.md` §12 #2). Addressed by selecting pure-ML `RF+XGB`.
2. **The 0.19 Intermediate AUC Gap vs. Fuadah 2024:** "Fuadah et al. 2024 reports intermediate AUC = 0.94; Mitchel's implementation gets intermediate AUC = 0.75 [or 0.825]. 0.19 gap, unresolved" (`context.md` §12 #1).
3. **Selection Optimism in Protocol A Sweep:** Winning configuration was chosen by evaluating all 612 configs on the test set; requires nested CV to remove optimism (`context_model.md` §4 #7).
4. **Static vs. Dynamic hERG Cell Models:** Static-hERG datasets collapse toward 3-class chance (macro-F1 0.30–0.36), whereas dynamic-hERG cluster at 0.74–0.77 (`context.md` §2).
5. **PCA Label-Feature Inconsistency Cases:** `domperidone`, `pimozide`, and `risperidone` exhibit clinical label vs. biomarker profile divergence in PCA space, representing label-boundary ambiguity rather than model capacity flaws (`context.md` §10).
