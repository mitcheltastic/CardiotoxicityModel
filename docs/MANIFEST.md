# SCRIPTS MANIFEST & DIRECTORY INDEX

> **Archival Note**: Legacy scripts located in `scripts/legacy/` carry an explicit two-level `sys.path` import bootstrap (`sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))`) to maintain module access to root-level utilities while preserved in their archival subfolder.

---

## Pipeline Scripts (`scripts/pipeline/`)
Execution scripts that performed model training, cross-platform evaluation, conformal quantile sweeps, SHAP analysis, and figure generation for the revised (`_rev`) benchmark re-run.

| Old Path | New Path | Description |
| :--- | :--- | :--- |
| `run_step3_stacking.py` | `scripts/pipeline/run_step3_stacking.py` | Trains Chantest StackingClassifier baseline using 4-fold Grouped CV. |
| `run_step4_cross_platform.py` | `scripts/pipeline/run_step4_cross_platform.py` | Evaluates frozen Chantest stacking model across Nanion and Manual test sets. |
| `run_step5_conformal.py` | `scripts/pipeline/run_step5_conformal.py` | Formulates split-conformal layer and computes 12 LODO calibration scores. |
| `run_step5_addendum.py` | `scripts/pipeline/run_step5_addendum.py` | Sensitivity analysis addendum for conformal calibration sizes. |
| `run_step5_addendum2.py` | `scripts/pipeline/run_step5_addendum2.py` | Second sensitivity addendum exploring non-conformity score quantile bounds. |
| `run_step6_conformal_op.py` | `scripts/pipeline/run_step6_conformal_op.py` | Scans ranks 12 to 8 to identify Rank 11 as tightest 0-under-triage operating point. |
| `run_step6_addendum.py` | `scripts/pipeline/run_step6_addendum.py` | Addendum calculating empirical coverage and set size distributions across platforms. |
| `run_step7_shap.py` | `scripts/pipeline/run_step7_shap.py` | Computes global SHAP feature importance using KernelExplainer. |
| `run_step8_diagnostic.py` | `scripts/pipeline/run_step8_diagnostic.py` | Diagnostic script auditing `vrest` split frequency, drug range overlaps, and permutation impact. |
| `run_step9_bootstrap.py` | `scripts/pipeline/run_step9_bootstrap.py` | Computes 10,000-iteration drug-level bootstrap confidence intervals for macro-F1. |
| `run_step10_fig1.py` | `scripts/pipeline/run_step10_fig1.py` | Generates two-row vector PDF block diagram (Figure 1). |

---

## Reporting & Compilation Scripts (`scripts/reporting/`)
Audit, compilation, coverage reconciliation, verification, and documentation helper scripts.

| Old Path | New Path | Description |
| :--- | :--- | :--- |
| `run_step10_vrest_recheck.py` | `scripts/reporting/run_step10_vrest_recheck.py` | Direct source-data recomputation of feature range overlaps across training and testing drugs. |
| `run_step11_compile1.py` | `scripts/reporting/run_step11_compile1.py` | First-pass compilation helper for canonical project metrics. |
| `run_step11_compile2.py` | `scripts/reporting/run_step11_compile2.py` | Second-pass compilation helper adding Wilson CIs and rule-of-three bounds. |
| `run_step12_compile_final.py` | `scripts/reporting/run_step12_compile_final.py` | Canonical compilation script generating `docs/PAPER_NUMBERS.md`. |
| `run_step13_fix_coverage.py` | `scripts/reporting/run_step13_fix_coverage.py` | Recomputes empirical coverage vs correct-outcome proportion per platform. |
| `reconcile_coverage_misses.py` | `scripts/reporting/reconcile_coverage_misses.py` | Reconciles coverage-miss drug lists between STEP6 and PAPER_NUMBERS. |
| `append_coverage_reconciliation.py` | `scripts/reporting/append_coverage_reconciliation.py` | Appends coverage reconciliation table to PAPER_NUMBERS.md. |
| `generate_storyline.py` | `scripts/reporting/generate_storyline.py` | Initial script generating `docs/STORYLINE.md`. |
| `rewrite_storyline.py` | `scripts/reporting/rewrite_storyline.py` | Rewrites STORYLINE.md stripping unsourced domain claims and applying neutral process history. |
| `update_vrest_conclusion.py` | `scripts/reporting/update_vrest_conclusion.py` | Updates STORYLINE.md Section 7, 8j, and 9 with non-separable vrest findings. |
| `fix_storyline_section7.py` | `scripts/reporting/fix_storyline_section7.py` | Helper script replacing fabricated thresholds with measured RF split ranges. |
| `verify_artifact_index.py` | `scripts/reporting/verify_artifact_index.py` | Verifies existence and paths of all project artifacts on disk. |

---

## Legacy Scripts (`scripts/legacy/`)
*Archival scripts, early unrevised diagnostics, plots, and deployment prototypes (July 16 and earlier).*

| Old Path | New Path | Description |
| :--- | :--- | :--- |
| `build_shap.py` | `scripts/legacy/build_shap.py` | Legacy SHAP plot builder script. |
| `conformal_ordinal.py` | `scripts/legacy/conformal_ordinal.py` | Legacy script comparing ordinal vs flat conformal formulations. |
| `debug_classifier.py` | `scripts/legacy/debug_classifier.py` | Legacy classifier debugging script. |
| `deploy.py` | `scripts/legacy/deploy.py` | Early deployment packaging prototype script. |
| `deploy_hf.py` | `scripts/legacy/deploy_hf.py` | HuggingFace Space deployment prototype script. |
| `diagnostic.py` | `scripts/legacy/diagnostic.py` | Early cross-instrument diagnostic script. |
| `diagnostic_cross_instrument.py` | `scripts/legacy/diagnostic_cross_instrument.py` | Legacy cross-instrument generalization script. |
| `diagnostic_guards.py` | `scripts/legacy/diagnostic_guards.py` | Early diagnostic guardrail check script. |
| `diagnostic_q3.py` | `scripts/legacy/diagnostic_q3.py` | Early confusion matrix diagnostic script. |
| `extract.py` | `scripts/legacy/extract.py` | Notebook code extraction helper script. |
| `find_images.py` | `scripts/legacy/find_images.py` | Asset image lookup script. |
| `plot_coverage.py` | `scripts/legacy/plot_coverage.py` | Legacy coverage plotting script. |
| `plot_cross_instrument.py` | `scripts/legacy/plot_cross_instrument.py` | Legacy cross-instrument performance plotting script. |
| `plot_lodo.py` | `scripts/legacy/plot_lodo.py` | Legacy LODO score distribution plotting script. |
| `plot_per_drug.py` | `scripts/legacy/plot_per_drug.py` | Legacy per-drug performance plotting script. |
| `plot_shap.py` | `scripts/legacy/plot_shap.py` | Legacy SHAP importance plotting script. |
| `relabel_figures.py` | `scripts/legacy/relabel_figures.py` | Legacy figure relabeling utility script. |

---

## Root Modules & Unmoved Items (`TrainAI/` Root)
- `deploy_utils.py`: **KEEP AT ROOT**. Contains custom wrapper classes (`PreFittedEstimatorWrapper`) required when unpickling serialized model artifacts.
- `conformal_diag.py`: **KEEP AT ROOT**. Provenance source cited in `docs/DATA_AUDIT_REV.md` and `docs/MODEL_BRIEFING.md`.
- `conformal_safety.py`: **KEEP AT ROOT**. Conformal utility module imported by `conformal_ordinal.py` and `test_conformal.py`.
- `test_conformal.py`: **KEEP AT ROOT**. Unit test script cited in `docs/MODEL_BRIEFING.md`.
- `plus_ultra.ipynb`: **KEEP AT ROOT**. Primary Jupyter Notebook cited across `docs/DATA_AUDIT_REV.md`, `docs/MODEL_BRIEFING.md`, and `docs/SCIENCE_CONTEXT.md`.
- `test_local.py`: **KEEP AT ROOT**. Local testing script using `__file__`-relative import resolution.