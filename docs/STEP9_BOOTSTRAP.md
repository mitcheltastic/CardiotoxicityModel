==========================================
=== STEP 9 — DRUG-LEVEL BOOTSTRAP ===
==========================================
Resampling unit: DRUG (16 held-out test drugs)
Iterations: 10000
Seed: 42
==========================================

DEGENERATE-RESAMPLE HANDLING:
Rule applied: We compute macro-average F1 solely over the classes PRESENT in the true labels of each resample, adjusting to avoid undefined division by zero.
Affected resamples missing a class for chantest: 131 out of 10000 adjusted.
Affected resamples missing a class for nanion: 131 out of 10000 adjusted.
Affected resamples missing a class for manual: 131 out of 10000 adjusted.

=== PLATFORM: CHANTEST ===
Drug-level macro-F1 : 0.679739 [95% CI: 0.391534, 0.896825] (Point Est: 0.694444)
Recall (High)       : 0.750000 [95% CI: 0.000000, 1.000000]
Recall (Int )       : 0.714286 [95% CI: 0.333333, 1.000000]
Recall (Low )       : 0.600000 [95% CI: 0.000000, 1.000000]
Rank-11 Correct     : 0.687500 [95% CI: 0.437500, 0.875000]
Rank-11 Over-called : 0.312500 [95% CI: 0.125000, 0.562500]
~~Rank-11 Under-triage: 0.000000 [95% CI: 0.000000, 0.000000]~~ (SUPERSEDED)
Rank-11 Under-triage: 0 of 16 under-triaged drugs observed; approximate 95% upper bound on the true rate = 0.187500 (18.75%).
Rank-11 Coverage    : 0.750000 [95% CI: 0.500000, 0.937500]
Rank-12 Coverage    : 0.937500 [95% CI: 0.812500, 1.000000]

=== PLATFORM: NANION ===
Drug-level macro-F1 : 0.595238 [95% CI: 0.311111, 0.851852] (Point Est: 0.621032)
Recall (High)       : 0.750000 [95% CI: 0.000000, 1.000000]
Recall (Int )       : 0.714286 [95% CI: 0.333333, 1.000000]
Recall (Low )       : 0.400000 [95% CI: 0.000000, 1.000000]
Rank-11 Correct     : 0.687500 [95% CI: 0.437500, 0.875000]
Rank-11 Over-called : 0.312500 [95% CI: 0.125000, 0.562500]
~~Rank-11 Under-triage: 0.000000 [95% CI: 0.000000, 0.000000]~~ (SUPERSEDED)
Rank-11 Under-triage: 0 of 16 under-triaged drugs observed; approximate 95% upper bound on the true rate = 0.187500 (18.75%).
Rank-11 Coverage    : 0.750000 [95% CI: 0.500000, 0.937500]
Rank-12 Coverage    : 0.937500 [95% CI: 0.812500, 1.000000]

=== PLATFORM: MANUAL ===
Drug-level macro-F1 : 0.738889 [95% CI: 0.479347, 0.944056] (Point Est: 0.754762)
Recall (High)       : 1.000000 [95% CI: 1.000000, 1.000000]
Recall (Int )       : 0.714286 [95% CI: 0.333333, 1.000000]
Recall (Low )       : 0.600000 [95% CI: 0.000000, 1.000000]
Rank-11 Correct     : 0.687500 [95% CI: 0.437500, 0.875000]
Rank-11 Over-called : 0.312500 [95% CI: 0.125000, 0.562500]
~~Rank-11 Under-triage: 0.000000 [95% CI: 0.000000, 0.000000]~~ (SUPERSEDED)
Rank-11 Under-triage: 0 of 16 under-triaged drugs observed; approximate 95% upper bound on the true rate = 0.187500 (18.75%).
Rank-11 Coverage    : 0.875000 [95% CI: 0.687500, 1.000000]
Rank-12 Coverage    : 1.000000 [95% CI: 1.000000, 1.000000]

=== PAIRWISE PLATFORM COMPARISON (Macro-F1 Differences) ===
Chantest - Nanion   :  0.069946 [95% CI:  0.000000,  0.263348] -> contains zero
Chantest - Manual   : -0.059096 [95% CI: -0.341773,  0.187302] -> contains zero
Nanion - Manual     : -0.128659 [95% CI: -0.376781,  0.000000] -> contains zero

==========================================
=== SANITY FLAGS ===
==========================================
FLAG: chantest macro-F1 CI width (0.505291) exceeds 0.30
FLAG: chantest Recall (High) CI width (1.000000) exceeds 0.30
FLAG: chantest Recall (Int) CI width (0.666667) exceeds 0.30
FLAG: chantest Recall (Low) CI width (1.000000) exceeds 0.30
FLAG: nanion macro-F1 CI width (0.540741) exceeds 0.30
FLAG: nanion Recall (High) CI width (1.000000) exceeds 0.30
FLAG: nanion Recall (Int) CI width (0.666667) exceeds 0.30
FLAG: nanion Recall (Low) CI width (1.000000) exceeds 0.30
FLAG: manual macro-F1 CI width (0.464709) exceeds 0.30
FLAG: manual Recall (Int) CI width (0.666667) exceeds 0.30
FLAG: manual Recall (Low) CI width (1.000000) exceeds 0.30