==========================================
=== STEP 0 — SEVERITY MAP ===
==========================================
SEVERITY = {0: 2, 1: 1, 2: 0} (high most severe, low least)

==========================================
=== STEP 1 — FULL RANK ENUMERATION, ALL 12 DRUGS RETAINED ===
==========================================
Rank k   | q_hat(k)   | Alpha Interval                      | Smallest Alpha  | Inclusion Threshold (1 - q_hat)
---------------------------------------------------------------------------------------------------------
12       | 0.986695   | [0.076923, 0.153846)                | 0.076923        | 0.013305
11       | 0.762830   | [0.153846, 0.230769)                | 0.153846        | 0.237170
10       | 0.703345   | [0.230769, 0.307692)                | 0.230769        | 0.296655
9        | 0.630580   | [0.307692, 0.384615)                | 0.307692        | 0.369420
8        | 0.453940   | [0.384615, 0.461538)                | 0.384615        | 0.546060

Confirmed: rank 12 -> 0.986695 and rank 11 -> 0.762830.

==========================================
=== STEP 2 — OUTCOMES AT EVERY RANK, ALL THREE PLATFORMS ===
==========================================
Rank | Platform   | Cor  | Ovr  | Und  | Abs  | AvgSz  | Dist(1,2,3)     | EmpCov  | NomCov  | Gap
----------------------------------------------------------------------------------------------------
12   | chantest   | 6    | 10   | 0    | 0    | 2.19   | (3,7,6)         | 0.9375  | 0.9231  | +0.0144
12   | nanion     | 8    | 8    | 0    | 0    | 2.00   | (5,6,5)         | 0.9375  | 0.9231  | +0.0144
12   | manual     | 6    | 10   | 0    | 0    | 2.50   | (2,4,10)        | 1.0000  | 0.9231  | +0.0769
11   | chantest   | 11   | 5    | 0    | 0    | 1.12   | (14,2,0)        | 0.7500  | 0.8462  | -0.0962
11   | nanion     | 11   | 5    | 0    | 0    | 1.12   | (14,2,0)        | 0.7500  | 0.8462  | -0.0962
11   | manual     | 11   | 5    | 0    | 0    | 1.38   | (10,6,0)        | 0.8750  | 0.8462  | +0.0288
10   | chantest   | 12   | 4    | 0    | 0    | 1.06   | (15,1,0)        | 0.7500  | 0.7692  | -0.0192
10   | nanion     | 10   | 5    | 1    | 0    | 1.00   | (16,0,0)        | 0.6250  | 0.7692  | -0.1442
10   | manual     | 12   | 4    | 0    | 0    | 1.19   | (13,3,0)        | 0.8125  | 0.7692  | +0.0433
9    | chantest   | 12   | 4    | 0    | 0    | 1.06   | (15,1,0)        | 0.7500  | 0.6923  | +0.0577
9    | nanion     | 10   | 5    | 1    | 0    | 1.00   | (16,0,0)        | 0.6250  | 0.6923  | -0.0673
9    | manual     | 12   | 4    | 0    | 0    | 1.12   | (14,2,0)        | 0.7500  | 0.6923  | +0.0577
8    | chantest   | 11   | 4    | 0    | 1    | 0.94   | (15,0,0)        | 0.6875  | 0.6154  | +0.0721
8    | nanion     | 10   | 5    | 1    | 0    | 1.00   | (16,0,0)        | 0.6250  | 0.6154  | +0.0096
8    | manual     | 11   | 4    | 0    | 1    | 0.94   | (15,0,0)        | 0.6875  | 0.6154  | +0.0721

rank 12: under-triaged = chantest 0, nanion 0, manual 0
rank 11: under-triaged = chantest 0, nanion 0, manual 0
rank 10: under-triaged = chantest 0, nanion 1, manual 0
rank 9: under-triaged = chantest 0, nanion 1, manual 0
rank 8: under-triaged = chantest 0, nanion 1, manual 0

SMALLEST q_hat (tightest rank) at which under-triage is zero on ALL three platforms is 0.986695 at rank 12.

> [!WARNING]
> *The sentence above identifying Rank 12 as the tightest rank is erroneous because the search stopped prematurely. It is superseded by ADDENDUM — CORRECTED BOUNDARY AND FINAL TABLE below.*

==========================================
=== STEP 3 — VERIFY THE BOUNDARY IN BOTH DIRECTIONS ===
==========================================
Safe rank: 12 (q_hat = 0.986695)
Tighter rank: 11 (q_hat = 0.762830)

Drugs whose outcome changes between these two ranks:

Platform: chantest
Drug Name: astemizole
True Label: intermediate
p_high: 0.039572, p_int: 0.960353, p_low: 0.000075
Rank 12 Set: ['high', 'intermediate'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: chantest
Drug Name: clarithromycin
True Label: intermediate
p_high: 0.015209, p_int: 0.984211, p_low: 0.000580
Rank 12 Set: ['high', 'intermediate'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: chantest
Drug Name: clozapine
True Label: intermediate
p_high: 0.078595, p_int: 0.897620, p_low: 0.023785
Rank 12 Set: ['high', 'intermediate', 'low'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: chantest
Drug Name: droperidol
True Label: intermediate
p_high: 0.057251, p_int: 0.941604, p_low: 0.001145
Rank 12 Set: ['high', 'intermediate'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: chantest
Drug Name: nitrendipine
True Label: low
p_high: 0.029895, p_int: 0.089335, p_low: 0.880770
Rank 12 Set: ['high', 'intermediate', 'low'] | Action: high | Outcome: over-called
Rank 11 Set: ['low'] | Action: low | Outcome: correct

Platform: nanion
Drug Name: astemizole
True Label: intermediate
p_high: 0.039650, p_int: 0.960270, p_low: 0.000080
Rank 12 Set: ['high', 'intermediate'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: nanion
Drug Name: clarithromycin
True Label: intermediate
p_high: 0.046096, p_int: 0.830514, p_low: 0.123390
Rank 12 Set: ['high', 'intermediate', 'low'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: nanion
Drug Name: droperidol
True Label: intermediate
p_high: 0.079385, p_int: 0.918105, p_low: 0.002510
Rank 12 Set: ['high', 'intermediate'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: manual
Drug Name: astemizole
True Label: intermediate
p_high: 0.034195, p_int: 0.965555, p_low: 0.000250
Rank 12 Set: ['high', 'intermediate'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: manual
Drug Name: clarithromycin
True Label: intermediate
p_high: 0.067979, p_int: 0.558811, p_low: 0.373210
Rank 12 Set: ['high', 'intermediate', 'low'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate', 'low'] | Action: intermediate | Outcome: correct

Platform: manual
Drug Name: clozapine
True Label: intermediate
p_high: 0.069431, p_int: 0.915694, p_low: 0.014875
Rank 12 Set: ['high', 'intermediate', 'low'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: manual
Drug Name: droperidol
True Label: intermediate
p_high: 0.138538, p_int: 0.855402, p_low: 0.006060
Rank 12 Set: ['high', 'intermediate'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Platform: manual
Drug Name: risperidone
True Label: intermediate
p_high: 0.042058, p_int: 0.886972, p_low: 0.070970
Rank 12 Set: ['high', 'intermediate', 'low'] | Action: high | Outcome: over-called
Rank 11 Set: ['intermediate'] | Action: intermediate | Outcome: correct

Specifically, the following drugs become under-triaged when tightening by one rank:

==========================================
=== STEP 4 — HEAD-TO-HEAD OF THE TWO CANDIDATE OPERATING POINTS ===
==========================================
Platform   | Method (Rank/q_hat)       | Min Alpha  | NomCov   | EmpCov   | Gap      | Cor  | Ovr  | Und  | AvgSz  | % Size-3
------------------------------------------------------------------------------------------------------------------------
chantest   | ALWAYS-HIGH               | -          | -        | -        | -        | 4    | 12   | 0    | 1.00   | 0.0%
           | HARD-LABEL                | -          | -        | -        | -        | 11   | 4    | 1    | 1.00   | 0.0%
           | Rank 12 (q=0.986695)      | 0.076923   | 0.9231   | 0.9375   | +0.0144  | 6    | 10   | 0    | 2.19   | 37.5%
           | Rank 11 (q=0.762830)      | 0.153846   | 0.8462   | 0.7500   | -0.0962  | 11   | 5    | 0    | 1.12   | 0.0%
nanion     | ALWAYS-HIGH               | -          | -        | -        | -        | 4    | 12   | 0    | 1.00   | 0.0%
           | HARD-LABEL                | -          | -        | -        | -        | 10   | 5    | 1    | 1.00   | 0.0%
           | Rank 12 (q=0.986695)      | 0.076923   | 0.9231   | 0.9375   | +0.0144  | 8    | 8    | 0    | 2.00   | 31.2%
           | Rank 11 (q=0.762830)      | 0.153846   | 0.8462   | 0.7500   | -0.0962  | 11   | 5    | 0    | 1.12   | 0.0%
manual     | ALWAYS-HIGH               | -          | -        | -        | -        | 4    | 12   | 0    | 1.00   | 0.0%
           | HARD-LABEL                | -          | -        | -        | -        | 12   | 4    | 0    | 1.00   | 0.0%
           | Rank 12 (q=0.986695)      | 0.076923   | 0.9231   | 1.0000   | +0.0769  | 6    | 10   | 0    | 2.50   | 62.5%
           | Rank 11 (q=0.762830)      | 0.153846   | 0.8462   | 0.8750   | +0.0288  | 11   | 5    | 0    | 1.38   | 0.0%

==========================================
=== STEP 5 — DILTIAZEM AS A CALIBRATION OUTLIER ===
==========================================
12 LODO scores sorted descending:
  diltiazem       (True: low         ) : 0.986695
  sotalol         (True: high        ) : 0.762830
  bepridil        (True: high        ) : 0.703345
  dofetilide      (True: high        ) : 0.630580
  verapamil       (True: low         ) : 0.453940
  chlorpromazine  (True: intermediate) : 0.408085
  ranolazine      (True: low         ) : 0.255295
  mexiletine      (True: low         ) : 0.236885
  ondansetron     (True: intermediate) : 0.163680
  cisapride       (True: intermediate) : 0.152855
  terfenadine     (True: intermediate) : 0.094280
  quinidine       (True: high        ) : 0.045365

Gap between largest (diltiazem) and second largest (sotalol): 0.223865
Number of scores exceeding 0.5: 4 out of 12.
Note: This calibration set is small enough that a single outlier (diltiazem) stretches the inclusion threshold enormously for the entire conformal layer.

==========================================
=== SANITY FLAGS ===
==========================================
  - FLAG: Rank 12 on chantest average set size exceeds 2.0 (2.19).
  - FLAG: Rank 12 on manual average set size exceeds 2.0 (2.50).
  - FLAG: Rank 11 on chantest empirical coverage (0.7500) falls below nominal (0.8462).
  - FLAG: Rank 11 on nanion empirical coverage (0.7500) falls below nominal (0.8462).
  - FLAG: Rank 10 on chantest empirical coverage (0.7500) falls below nominal (0.7692).
  - FLAG: Rank 10 on nanion shows 1 under-triaged drug(s).
  - FLAG: Rank 10 on nanion empirical coverage (0.6250) falls below nominal (0.7692).
  - FLAG: Rank 9 on nanion shows 1 under-triaged drug(s).
  - FLAG: Rank 9 on nanion empirical coverage (0.6250) falls below nominal (0.6923).
  - FLAG: Rank 8 on nanion shows 1 under-triaged drug(s).
==========================================
=== ADDENDUM — CORRECTED BOUNDARY AND FINAL TABLE ===
==========================================

=== STEP 1 — FIX THE BOUNDARY LINE ===
rank 12 (q_hat=0.986695): under-triage count for chantest=0 / nanion=0 / manual=0
rank 11 (q_hat=0.762830): under-triage count for chantest=0 / nanion=0 / manual=0
rank 10 (q_hat=0.703345): under-triage count for chantest=0 / nanion=1 / manual=0
rank 9 (q_hat=0.630580): under-triage count for chantest=0 / nanion=1 / manual=0
rank 8 (q_hat=0.453940): under-triage count for chantest=0 / nanion=1 / manual=0

TIGHTEST (smallest q_hat) rank at which under-triage is zero on ALL three platforms: Rank 11 (q_hat = 0.762830)
The next tighter rank is Rank 10 (q_hat = 0.703345) with under-triages: chantest=0, nanion=1, manual=0.

ERRONEOUS SENTENCE (BEFORE):
"SMALLEST q_hat (tightest rank) at which under-triage is zero on ALL three platforms is 0.986695 at rank 12."
CORRECTED REPLACEMENT (AFTER):
"SMALLEST q_hat (tightest rank) at which under-triage is zero on ALL three platforms is 0.762830 at rank 11."

=== STEP 2 — PER-PLATFORM DETAIL AT RANK 12 AND RANK 11 ===
Platform   | Rank    | Cor | Ovr | Und | Abs | AvgSz   | Sz1 | Sz2 | Sz3 | %Sz3   | EmpCov      | NomCov  | Gap    
-----------------------------------------------------------------------------------------------------------------------------
chantest   | 12      | 6   | 10  | 0   | 0   | 2.1875  | 3   | 7   | 6   | 37.5 % | 15/16 (93.8%) | 0.9231  | +0.0144
chantest   | 11      | 11  | 5   | 0   | 0   | 1.1250  | 14  | 2   | 0   | 0.0  % | 12/16 (75.0%) | 0.8462  | -0.0962
nanion     | 12      | 8   | 8   | 0   | 0   | 2.0000  | 5   | 6   | 5   | 31.2 % | 15/16 (93.8%) | 0.9231  | +0.0144
nanion     | 11      | 11  | 5   | 0   | 0   | 1.1250  | 14  | 2   | 0   | 0.0  % | 12/16 (75.0%) | 0.8462  | -0.0962
manual     | 12      | 6   | 10  | 0   | 0   | 2.5000  | 2   | 4   | 10  | 62.5 % | 16/16 (100.0%) | 0.9231  | +0.0769
manual     | 11      | 11  | 5   | 0   | 0   | 1.3750  | 10  | 6   | 0   | 0.0  % | 14/16 (87.5%) | 0.8462  | +0.0288

=== STEP 3 — COVERAGE MISSES AT RANK 11 ===

--- Misses on CHANTEST at Rank 11 ---
domperidone
  True Label: intermediate
  Probs: p_high=0.986030 / p_int=0.003185 / p_low=0.010785
  Threshold (1-q_hat): 0.237170
  Set: ['high']
  Action: high
  Severity: Action(2) vs True(1)
  Outcome: over-called
loratadine
  True Label: low
  Probs: p_high=0.148550 / p_int=0.629450 / p_low=0.222000
  Threshold (1-q_hat): 0.237170
  Set: ['intermediate']
  Action: intermediate
  Severity: Action(1) vs True(0)
  Outcome: over-called
pimozide
  True Label: intermediate
  Probs: p_high=0.944341 / p_int=0.046999 / p_low=0.008660
  Threshold (1-q_hat): 0.237170
  Set: ['high']
  Action: high
  Severity: Action(2) vs True(1)
  Outcome: over-called
tamoxifen
  True Label: low
  Probs: p_high=0.008780 / p_int=0.930675 / p_low=0.060545
  Threshold (1-q_hat): 0.237170
  Set: ['intermediate']
  Action: intermediate
  Severity: Action(1) vs True(0)
  Outcome: over-called
Summary for chantest: ALL coverage misses are over-calls? YES. Over-calls: 4, Under-triages: 0

--- Misses on NANION at Rank 11 ---
domperidone
  True Label: intermediate
  Probs: p_high=0.990000 / p_int=0.000000 / p_low=0.010000
  Threshold (1-q_hat): 0.237170
  Set: ['high']
  Action: high
  Severity: Action(2) vs True(1)
  Outcome: over-called
metoprolol
  True Label: low
  Probs: p_high=0.024942 / p_int=0.817223 / p_low=0.157835
  Threshold (1-q_hat): 0.237170
  Set: ['intermediate']
  Action: intermediate
  Severity: Action(1) vs True(0)
  Outcome: over-called
pimozide
  True Label: intermediate
  Probs: p_high=0.945112 / p_int=0.044843 / p_low=0.010045
  Threshold (1-q_hat): 0.237170
  Set: ['high']
  Action: high
  Severity: Action(2) vs True(1)
  Outcome: over-called
tamoxifen
  True Label: low
  Probs: p_high=0.004140 / p_int=0.936830 / p_low=0.059030
  Threshold (1-q_hat): 0.237170
  Set: ['intermediate']
  Action: intermediate
  Severity: Action(1) vs True(0)
  Outcome: over-called
Summary for nanion: ALL coverage misses are over-calls? YES. Over-calls: 4, Under-triages: 0

--- Misses on MANUAL at Rank 11 ---
metoprolol
  True Label: low
  Probs: p_high=0.252260 / p_int=0.686375 / p_low=0.061365
  Threshold (1-q_hat): 0.237170
  Set: ['high', 'intermediate']
  Action: high
  Severity: Action(2) vs True(0)
  Outcome: over-called
tamoxifen
  True Label: low
  Probs: p_high=0.013279 / p_int=0.839711 / p_low=0.147010
  Threshold (1-q_hat): 0.237170
  Set: ['intermediate']
  Action: intermediate
  Severity: Action(1) vs True(0)
  Outcome: over-called
Summary for manual: ALL coverage misses are over-calls? YES. Over-calls: 2, Under-triages: 0

=== STEP 4 — COVERAGE UNCERTAINTY ===

--- Rank 12 ---
chantest   | Emp: 0.9375 | 95% Wilson CI: [0.7167, 0.9889] | Nominal 0.9231 inside? YES
nanion     | Emp: 0.9375 | 95% Wilson CI: [0.7167, 0.9889] | Nominal 0.9231 inside? YES
manual     | Emp: 1.0000 | 95% Wilson CI: [0.8064, 1.0000] | Nominal 0.9231 inside? YES

--- Rank 11 ---
chantest   | Emp: 0.7500 | 95% Wilson CI: [0.5050, 0.8982] | Nominal 0.8462 inside? YES
nanion     | Emp: 0.7500 | 95% Wilson CI: [0.5050, 0.8982] | Nominal 0.8462 inside? YES
manual     | Emp: 0.8750 | 95% Wilson CI: [0.6398, 0.9650] | Nominal 0.8462 inside? YES

=== STEP 5 — THE RANK-10 BOUNDARY EVIDENCE ===
At rank 10 (q_hat = 0.703345), identifying drugs that become under-triaged:
Drug: disopyramide on nanion
  True label: high
  p_mean: p_high=0.264455, p_int=0.710815, p_low=0.024730
  Threshold: 0.296655
  Set: ['intermediate']
  Action: intermediate
  Margin by which severe class (high) fell below threshold: 0.032200

=== STEP 6 — FINAL SUMMARY TABLE FOR THE PAPER ===
Platform   | Method (Rank/Base)   | q_hat      | Min Alpha  | NomCov   | EmpCov   | Cor  | Ovr  | Und  | AvgSz  | % Sz-3
-----------------------------------------------------------------------------------------------------------------------------
chantest   | ALWAYS-HIGH          | -          | -          | -        | -        | 4    | 12   | 0    | 1.0000 | 0.0%
           | HARD-LABEL           | -          | -          | -        | -        | 11   | 4    | 1    | 1.0000 | 0.0%
           | Rank 12              | 0.986695   | 0.076923   | 0.9231   | 0.9375   | 6    | 10   | 0    | 2.1875 | 37.5%
           | Rank 11              | 0.762830   | 0.153846   | 0.8462   | 0.7500   | 11   | 5    | 0    | 1.1250 | 0.0%
           | Rank 10              | 0.703345   | 0.230769   | 0.7692   | 0.7500   | 12   | 4    | 0    | 1.0625 | 0.0%
nanion     | ALWAYS-HIGH          | -          | -          | -        | -        | 4    | 12   | 0    | 1.0000 | 0.0%
           | HARD-LABEL           | -          | -          | -        | -        | 10   | 5    | 1    | 1.0000 | 0.0%
           | Rank 12              | 0.986695   | 0.076923   | 0.9231   | 0.9375   | 8    | 8    | 0    | 2.0000 | 31.2%
           | Rank 11              | 0.762830   | 0.153846   | 0.8462   | 0.7500   | 11   | 5    | 0    | 1.1250 | 0.0%
           | Rank 10              | 0.703345   | 0.230769   | 0.7692   | 0.6250   | 10   | 5    | 1    | 1.0000 | 0.0%
manual     | ALWAYS-HIGH          | -          | -          | -        | -        | 4    | 12   | 0    | 1.0000 | 0.0%
           | HARD-LABEL           | -          | -          | -        | -        | 12   | 4    | 0    | 1.0000 | 0.0%
           | Rank 12              | 0.986695   | 0.076923   | 0.9231   | 1.0000   | 6    | 10   | 0    | 2.5000 | 62.5%
           | Rank 11              | 0.762830   | 0.153846   | 0.8462   | 0.8750   | 11   | 5    | 0    | 1.3750 | 0.0%
           | Rank 10              | 0.703345   | 0.230769   | 0.7692   | 0.8125   | 12   | 4    | 0    | 1.1875 | 0.0%

==========================================
=== SANITY FLAGS ===
==========================================
  - CLEAR: No sanity flags raised for this section.