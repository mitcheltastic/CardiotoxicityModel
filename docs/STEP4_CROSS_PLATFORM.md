Loading model...
Loaded Model SHA-256: A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2
Expected SHA-256:     A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2
Hash Check: PASS

Loading metadata...

==========================================
=== STEP 0 — SEVERITY MAP ===
==========================================
SEVERITY = {0: 2, 1: 1, 2: 0}
severity(pred) > severity(true)  -> OVER-CALLED   (safe direction)
severity(pred) < severity(true)  -> UNDER-TRIAGED (unsafe direction)

==========================================
=== STEP 1 — REPRODUCTION GUARD ===
==========================================
Predict time features: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Canonical features:    ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Feature Order Check: PASS
Expected Drug-Level Macro-F1: 0.694444
Actual Drug-Level Macro-F1:   0.694444
Drug-Level F1 Check: PASS
Expected Row-Level Macro-F1: 0.688289
Actual Row-Level Macro-F1:   0.688289
Row-Level F1 Check: PASS

==========================================
=== STEP 2 — PRIMARY: 16 HELD-OUT DRUGS, ALL THREE PLATFORMS ===
==========================================

--- Platform: CHANTEST (TESTING) ---
ROW-LEVEL: Macro-F1=0.688289, Accuracy=0.682344
DRUG-LEVEL: Macro-F1=0.694444, Accuracy=0.687500
Per-Class Metrics:
  Class 0 (high        ): Precision=0.600000, Recall=0.750000, F1=0.666667
  Class 1 (intermediate): Precision=0.625000, Recall=0.714286, F1=0.666667
  Class 2 (low         ): Precision=1.000000, Recall=0.600000, F1=0.750000
Confusion Matrix (rows=true, cols=pred):
[[3 1 0]
 [2 5 0]
 [0 2 3]]

16-ROW DRUG-LEVEL PREDICTION TABLE:
Drug Name        | True Label   | Pred Label   | P(High)    | P(Int)     | P(Low)     | Error Direction
-----------------------------------------------------------------------------------------------
astemizole       | intermediate | intermediate | 0.039572   | 0.960353   | 0.000075   | MATCH
azimilide        | high         | high         | 0.989995   | 0.000000   | 0.010005   | MATCH
clarithromycin   | intermediate | intermediate | 0.015209   | 0.984211   | 0.000580   | MATCH
clozapine        | intermediate | intermediate | 0.078595   | 0.897620   | 0.023785   | MATCH
disopyramide     | high         | intermediate | 0.438730   | 0.495815   | 0.065455   | UNDER-TRIAGED
domperidone      | intermediate | high         | 0.986030   | 0.003185   | 0.010785   | OVER-CALLED
droperidol       | intermediate | intermediate | 0.057251   | 0.941604   | 0.001145   | MATCH
ibutilide        | high         | high         | 0.911960   | 0.048745   | 0.039295   | MATCH
loratadine       | low          | intermediate | 0.148550   | 0.629450   | 0.222000   | OVER-CALLED
metoprolol       | low          | low          | 0.003472   | 0.262423   | 0.734105   | MATCH
nifedipine       | low          | low          | 0.000000   | 0.000000   | 1.000000   | MATCH
nitrendipine     | low          | low          | 0.029895   | 0.089335   | 0.880770   | MATCH
pimozide         | intermediate | high         | 0.944341   | 0.046999   | 0.008660   | OVER-CALLED
risperidone      | intermediate | intermediate | 0.006530   | 0.978645   | 0.014825   | MATCH
tamoxifen        | low          | intermediate | 0.008780   | 0.930675   | 0.060545   | OVER-CALLED
vandetanib       | high         | high         | 0.927865   | 0.052095   | 0.020040   | MATCH

Errors: 5
  - disopyramide: True high -> Pred intermediate (UNDER-TRIAGED)
  - domperidone: True intermediate -> Pred high (OVER-CALLED)
  - loratadine: True low -> Pred intermediate (OVER-CALLED)
  - pimozide: True intermediate -> Pred high (OVER-CALLED)
  - tamoxifen: True low -> Pred intermediate (OVER-CALLED)
Explicit count of UNDER-TRIAGED drugs: 1

--- Platform: NANION (TESTING) ---
ROW-LEVEL: Macro-F1=0.632400, Accuracy=0.631969
DRUG-LEVEL: Macro-F1=0.621032, Accuracy=0.625000
Per-Class Metrics:
  Class 0 (high        ): Precision=0.600000, Recall=0.750000, F1=0.666667
  Class 1 (intermediate): Precision=0.555556, Recall=0.714286, F1=0.625000
  Class 2 (low         ): Precision=1.000000, Recall=0.400000, F1=0.571429
Confusion Matrix (rows=true, cols=pred):
[[3 1 0]
 [2 5 0]
 [0 3 2]]

16-ROW DRUG-LEVEL PREDICTION TABLE:
Drug Name        | True Label   | Pred Label   | P(High)    | P(Int)     | P(Low)     | Error Direction
-----------------------------------------------------------------------------------------------
astemizole       | intermediate | intermediate | 0.039650   | 0.960270   | 0.000080   | MATCH
azimilide        | high         | high         | 0.989990   | 0.000000   | 0.010010   | MATCH
clarithromycin   | intermediate | intermediate | 0.046096   | 0.830514   | 0.123390   | MATCH
clozapine        | intermediate | intermediate | 0.010763   | 0.989152   | 0.000085   | MATCH
disopyramide     | high         | intermediate | 0.264455   | 0.710815   | 0.024730   | UNDER-TRIAGED
domperidone      | intermediate | high         | 0.990000   | 0.000000   | 0.010000   | OVER-CALLED
droperidol       | intermediate | intermediate | 0.079385   | 0.918105   | 0.002510   | MATCH
ibutilide        | high         | high         | 0.835310   | 0.116440   | 0.048250   | MATCH
loratadine       | low          | intermediate | 0.146145   | 0.608570   | 0.245285   | OVER-CALLED
metoprolol       | low          | intermediate | 0.024942   | 0.817223   | 0.157835   | OVER-CALLED
nifedipine       | low          | low          | 0.000000   | 0.000000   | 1.000000   | MATCH
nitrendipine     | low          | low          | 0.000000   | 0.000000   | 1.000000   | MATCH
pimozide         | intermediate | high         | 0.945112   | 0.044843   | 0.010045   | OVER-CALLED
risperidone      | intermediate | intermediate | 0.005565   | 0.980140   | 0.014295   | MATCH
tamoxifen        | low          | intermediate | 0.004140   | 0.936830   | 0.059030   | OVER-CALLED
vandetanib       | high         | high         | 0.971765   | 0.009145   | 0.019090   | MATCH

Errors: 6
  - disopyramide: True high -> Pred intermediate (UNDER-TRIAGED)
  - domperidone: True intermediate -> Pred high (OVER-CALLED)
  - loratadine: True low -> Pred intermediate (OVER-CALLED)
  - metoprolol: True low -> Pred intermediate (OVER-CALLED)
  - pimozide: True intermediate -> Pred high (OVER-CALLED)
  - tamoxifen: True low -> Pred intermediate (OVER-CALLED)
Explicit count of UNDER-TRIAGED drugs: 1

--- Platform: MANUAL (TESTING) ---
ROW-LEVEL: Macro-F1=0.692524, Accuracy=0.688522
DRUG-LEVEL: Macro-F1=0.754762, Accuracy=0.750000
Per-Class Metrics:
  Class 0 (high        ): Precision=0.666667, Recall=1.000000, F1=0.800000
  Class 1 (intermediate): Precision=0.714286, Recall=0.714286, F1=0.714286
  Class 2 (low         ): Precision=1.000000, Recall=0.600000, F1=0.750000
Confusion Matrix (rows=true, cols=pred):
[[4 0 0]
 [2 5 0]
 [0 2 3]]

16-ROW DRUG-LEVEL PREDICTION TABLE:
Drug Name        | True Label   | Pred Label   | P(High)    | P(Int)     | P(Low)     | Error Direction
-----------------------------------------------------------------------------------------------
astemizole       | intermediate | intermediate | 0.034195   | 0.965555   | 0.000250   | MATCH
azimilide        | high         | high         | 0.929785   | 0.057645   | 0.012570   | MATCH
clarithromycin   | intermediate | intermediate | 0.067979   | 0.558811   | 0.373210   | MATCH
clozapine        | intermediate | intermediate | 0.069431   | 0.915694   | 0.014875   | MATCH
disopyramide     | high         | high         | 0.823025   | 0.149885   | 0.027090   | MATCH
domperidone      | intermediate | high         | 0.661065   | 0.318370   | 0.020565   | OVER-CALLED
droperidol       | intermediate | intermediate | 0.138538   | 0.855402   | 0.006060   | MATCH
ibutilide        | high         | high         | 0.794762   | 0.155993   | 0.049245   | MATCH
loratadine       | low          | low          | 0.026725   | 0.258380   | 0.714895   | MATCH
metoprolol       | low          | intermediate | 0.252260   | 0.686375   | 0.061365   | OVER-CALLED
nifedipine       | low          | low          | 0.000000   | 0.000000   | 1.000000   | MATCH
nitrendipine     | low          | low          | 0.000000   | 0.000000   | 1.000000   | MATCH
pimozide         | intermediate | high         | 0.631220   | 0.293695   | 0.075085   | OVER-CALLED
risperidone      | intermediate | intermediate | 0.042058   | 0.886972   | 0.070970   | MATCH
tamoxifen        | low          | intermediate | 0.013279   | 0.839711   | 0.147010   | OVER-CALLED
vandetanib       | high         | high         | 0.529700   | 0.423670   | 0.046630   | MATCH

Errors: 4
  - domperidone: True intermediate -> Pred high (OVER-CALLED)
  - metoprolol: True low -> Pred intermediate (OVER-CALLED)
  - pimozide: True intermediate -> Pred high (OVER-CALLED)
  - tamoxifen: True low -> Pred intermediate (OVER-CALLED)
Explicit count of UNDER-TRIAGED drugs: 0

CROSS-PLATFORM COMPARISON (Drug-Level Macro-F1):
CHANTEST: 0.694444
NANION:   0.621032
MANUAL:   0.754762
This is the paper's cross-platform result, evaluated on drugs the model never saw during training.

==========================================
=== STEP 3 — SECONDARY ===
==========================================
SEEN DRUGS — platform transfer only, not held-out generalization.
NANION (Seen Drugs) Drug-Level Macro-F1: 1.000000
MANUAL (Seen Drugs) Drug-Level Macro-F1: 1.000000

==========================================
=== STEP 4 — DISOPYRAMIDE FOCUS (AND OTHER UNDER-TRIAGES) ===
==========================================

--- DRUG: DISOPYRAMIDE ---
Platform: CHANTEST
  P(High): 0.438730 | P(Int): 0.495815 | P(Low): 0.065455
  Pred: intermediate | Status: UNDER-TRIAGED
  Difference (P(Int) - P(High)): 0.057085
Platform: NANION
  P(High): 0.264455 | P(Int): 0.710815 | P(Low): 0.024730
  Pred: intermediate | Status: UNDER-TRIAGED
  Difference (P(Int) - P(High)): 0.446360
Platform: MANUAL
  P(High): 0.823025 | P(Int): 0.149885 | P(Low): 0.027090
  Pred: high | Status: MATCH
  Difference (P(Int) - P(High)): -0.673140

==========================================
=== STEP 5 — PERSIST FOR THE CONFORMAL STEP ===
==========================================
Saved: d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\cross_platform_probs_chantest_rev.csv | Rows: 28 | SHA-256: 37E5906FE76A7F17CA98685DF3A72B094E109D7A031D15EB241ECD3C550B3397
Saved: d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\cross_platform_probs_nanion_rev.csv | Rows: 28 | SHA-256: 12D7BE04CF8235DB42A8ABE084998AF68F57B80A095176A699A7C9111531C5E5
Saved: d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\cross_platform_probs_manual_rev.csv | Rows: 28 | SHA-256: 67A431CABBF29D02DA19E5972B16734367B7D52E7ADD6495147211C5726A2D1D

==========================================
=== SANITY FLAGS ===
==========================================
  - FLAG: NANION macro-F1 (0.621032) differs sharply from chantest (0.694444)
  - FLAG: MANUAL macro-F1 (0.754762) differs sharply from chantest (0.694444)
  - FLAG: disopyramide is under-triaged on chantest
  - FLAG: disopyramide is under-triaged on nanion