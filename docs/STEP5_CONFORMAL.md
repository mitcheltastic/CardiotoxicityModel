Loading metadata to extract hyperparameters...
Hyperparameters Loaded:
{
  "rf": {
    "criterion": "entropy",
    "max_depth": null,
    "n_estimators": 400
  },
  "xgb": {
    "learning_rate": 0.1,
    "max_depth": 10,
    "n_estimators": 300
  },
  "meta_rf": {
    "random_state": 42
  }
}

==========================================
=== STEP 0 — SEVERITY MAP ===
==========================================
SEVERITY = {0: 2, 1: 1, 2: 0}

==========================================
=== STEP 1 — LODO CALIBRATION (12 genuine refits) ===
==========================================

--- Refitting for hold-out drug: bepridil ---
ASSERT drug bepridil appears zero times in fit set: PASS

--- Refitting for hold-out drug: chlorpromazine ---
ASSERT drug chlorpromazine appears zero times in fit set: PASS

--- Refitting for hold-out drug: cisapride ---
ASSERT drug cisapride appears zero times in fit set: PASS

--- Refitting for hold-out drug: diltiazem ---
ASSERT drug diltiazem appears zero times in fit set: PASS

--- Refitting for hold-out drug: dofetilide ---
ASSERT drug dofetilide appears zero times in fit set: PASS

--- Refitting for hold-out drug: mexiletine ---
ASSERT drug mexiletine appears zero times in fit set: PASS

--- Refitting for hold-out drug: ondansetron ---
ASSERT drug ondansetron appears zero times in fit set: PASS

--- Refitting for hold-out drug: quinidine ---
ASSERT drug quinidine appears zero times in fit set: PASS

--- Refitting for hold-out drug: ranolazine ---
ASSERT drug ranolazine appears zero times in fit set: PASS

--- Refitting for hold-out drug: sotalol ---
ASSERT drug sotalol appears zero times in fit set: PASS

--- Refitting for hold-out drug: terfenadine ---
ASSERT drug terfenadine appears zero times in fit set: PASS

--- Refitting for hold-out drug: verapamil ---
ASSERT drug verapamil appears zero times in fit set: PASS

12-ROW LODO CALIBRATION TABLE:
Drug Name       | True Label   | P(High)    | P(Int)     | P(Low)     | s_d
---------------------------------------------------------------------------
bepridil        | high         | 0.296655   | 0.606690   | 0.096655   | 0.703345  
chlorpromazine  | intermediate | 0.031070   | 0.591915   | 0.377015   | 0.408085  
cisapride       | intermediate | 0.025955   | 0.847145   | 0.126900   | 0.152855  
diltiazem       | low          | 0.000335   | 0.986360   | 0.013305   | 0.986695  
dofetilide      | high         | 0.369420   | 0.599270   | 0.031310   | 0.630580  
mexiletine      | low          | 0.053010   | 0.183875   | 0.763115   | 0.236885  
ondansetron     | intermediate | 0.131340   | 0.836320   | 0.032340   | 0.163680  
quinidine       | high         | 0.954635   | 0.026775   | 0.018590   | 0.045365  
ranolazine      | low          | 0.008030   | 0.247265   | 0.744705   | 0.255295  
sotalol         | high         | 0.237170   | 0.560180   | 0.202650   | 0.762830  
terfenadine     | intermediate | 0.082700   | 0.905720   | 0.011580   | 0.094280  
verapamil       | low          | 0.005210   | 0.448730   | 0.546060   | 0.453940  

12 LODO scores SORTED ASCENDING: [np.float64(0.045365), np.float64(0.09428), np.float64(0.152855), np.float64(0.16368), np.float64(0.236885), np.float64(0.255295), np.float64(0.408085), np.float64(0.45394), np.float64(0.63058), np.float64(0.703345), np.float64(0.76283), np.float64(0.986695)]
Min:    0.045365
Median: 0.331690
Max:    0.986695

INTEGRITY CHECK:
Max frozen model score on these 12 drugs: 0.079499
Min LODO refit score on these 12 drugs:   0.045365
LODO scores are visibly larger than frozen-model scores. Refit integrity PASS.

==========================================
=== STEP 2 — ALPHA SWEEP AND QUANTIZATION ===
==========================================
n = 12
Minimum feasible alpha: 1/13 = 0.076923
Maximum attainable q_hat: 0.986695

Alpha      | Rank k   | Feasible   | q_hat
---------------------------------------------
0.05       | 13       | False      | N/A
0.077      | 12       | True       | 0.986695
0.1        | 12       | True       | 0.986695
0.15       | 12       | True       | 0.986695
0.2        | 11       | True       | 0.762830
0.25       | 10       | True       | 0.703345
0.3        | 10       | True       | 0.703345
0.4        | 8        | True       | 0.453940

Distinct alphas collapsing to same q_hat:
  q_hat 0.986695 is shared by alphas: [0.077, 0.1, 0.15]
  q_hat 0.703345 is shared by alphas: [0.25, 0.3]

==========================================
=== STEP 3 — THE DECISIVE CHECK ===
==========================================
For disopyramide on each platform, s_high = 1 - p_high
Chantest s_high: 0.561270 | Min q_hat needed: 0.561270
Nanion   s_high: 0.735545 | Min q_hat needed: 0.735545
Manual   s_high: 0.176975 | Min q_hat needed: 0.176975

Given max attainable q_hat (0.986695), is disopyramide under-triage recoverable on chantest? YES (needs 0.561270).
Given max attainable q_hat (0.986695), is disopyramide under-triage recoverable on nanion?   YES (needs 0.735545).

==========================================
=== STEP 4 — PREDICTION SETS ON 16 TEST DRUGS ===
==========================================
Alpha  | Platform   | Correct  | Over   | Under  | Abstain  | Avg Size  | Uninformative
--------------------------------------------------------------------------------
0.077  | chantest   | 6        | 10     | 0      | 0        | 2.19      | 6
0.077  | nanion     | 8        | 8      | 0      | 0        | 2.00      | 5
0.077  | manual     | 6        | 10     | 0      | 0        | 2.50      | 10
0.1    | chantest   | 6        | 10     | 0      | 0        | 2.19      | 6
0.1    | nanion     | 8        | 8      | 0      | 0        | 2.00      | 5
0.1    | manual     | 6        | 10     | 0      | 0        | 2.50      | 10
0.15   | chantest   | 6        | 10     | 0      | 0        | 2.19      | 6
0.15   | nanion     | 8        | 8      | 0      | 0        | 2.00      | 5
0.15   | manual     | 6        | 10     | 0      | 0        | 2.50      | 10
0.2    | chantest   | 11       | 5      | 0      | 0        | 1.12      | 0
0.2    | nanion     | 11       | 5      | 0      | 0        | 1.12      | 0
0.2    | manual     | 11       | 5      | 0      | 0        | 1.38      | 0
0.25   | chantest   | 12       | 4      | 0      | 0        | 1.06      | 0
0.25   | nanion     | 10       | 5      | 1      | 0        | 1.00      | 0
0.25   | manual     | 12       | 4      | 0      | 0        | 1.19      | 0
0.3    | chantest   | 12       | 4      | 0      | 0        | 1.06      | 0
0.3    | nanion     | 10       | 5      | 1      | 0        | 1.00      | 0
0.3    | manual     | 12       | 4      | 0      | 0        | 1.19      | 0
0.4    | chantest   | 11       | 4      | 0      | 1        | 0.94      | 0
0.4    | nanion     | 10       | 5      | 1      | 0        | 1.00      | 0
0.4    | manual     | 11       | 4      | 0      | 1        | 0.94      | 0

FULL 16-ROW DETAIL FOR HUMAN-PICKED ALPHA = 0.15
Drug Name       | True         | Hard Pred    | Prediction Set                      | Action       | Outcome
--------------------------------------------------------------------------------------------------------------
astemizole      | intermediate | intermediate | ['high', 'intermediate']            | high         | over-called
azimilide       | high         | high         | ['high']                            | high         | correct
clarithromycin  | intermediate | intermediate | ['high', 'intermediate']            | high         | over-called
clozapine       | intermediate | intermediate | ['high', 'intermediate', 'low']     | high         | over-called
disopyramide    | high         | intermediate | ['high', 'intermediate', 'low']     | high         | correct
domperidone     | intermediate | high         | ['high']                            | high         | over-called
droperidol      | intermediate | intermediate | ['high', 'intermediate']            | high         | over-called
ibutilide       | high         | high         | ['high', 'intermediate', 'low']     | high         | correct
loratadine      | low          | intermediate | ['high', 'intermediate', 'low']     | high         | over-called
metoprolol      | low          | low          | ['intermediate', 'low']             | intermediate | over-called
nifedipine      | low          | low          | ['low']                             | low          | correct
nitrendipine    | low          | low          | ['high', 'intermediate', 'low']     | high         | over-called
pimozide        | intermediate | high         | ['high', 'intermediate']            | high         | over-called
risperidone     | intermediate | intermediate | ['intermediate', 'low']             | intermediate | correct
tamoxifen       | low          | intermediate | ['intermediate', 'low']             | intermediate | over-called
vandetanib      | high         | high         | ['high', 'intermediate', 'low']     | high         | correct

==========================================
=== STEP 5 — COVERAGE ===
==========================================
Alpha  | Nominal  | Platform   | Empirical  | Gap
--------------------------------------------------
0.077  | 0.9230   | chantest   | 0.9375     | +0.0145
0.077  | 0.9230   | nanion     | 0.9375     | +0.0145
0.077  | 0.9230   | manual     | 1.0000     | +0.0770
0.1    | 0.9000   | chantest   | 0.9375     | +0.0375
0.1    | 0.9000   | nanion     | 0.9375     | +0.0375
0.1    | 0.9000   | manual     | 1.0000     | +0.1000
0.15   | 0.8500   | chantest   | 0.9375     | +0.0875
0.15   | 0.8500   | nanion     | 0.9375     | +0.0875
0.15   | 0.8500   | manual     | 1.0000     | +0.1500
0.2    | 0.8000   | chantest   | 0.7500     | -0.0500
0.2    | 0.8000   | nanion     | 0.7500     | -0.0500
0.2    | 0.8000   | manual     | 0.8750     | +0.0750
0.25   | 0.7500   | chantest   | 0.7500     | +0.0000
0.25   | 0.7500   | nanion     | 0.6250     | -0.1250
0.25   | 0.7500   | manual     | 0.8125     | +0.0625
0.3    | 0.7000   | chantest   | 0.7500     | +0.0500
0.3    | 0.7000   | nanion     | 0.6250     | -0.0750
0.3    | 0.7000   | manual     | 0.8125     | +0.1125
0.4    | 0.6000   | chantest   | 0.6875     | +0.0875
0.4    | 0.6000   | nanion     | 0.6250     | +0.0250
0.4    | 0.6000   | manual     | 0.6875     | +0.0875

==========================================
=== STEP 6 — HARD-LABEL BASELINE ===
==========================================
CHANTEST Hard-Label: Correct=11, Over-called=4, Under-triaged=1
NANION Hard-Label: Correct=10, Over-called=5, Under-triaged=1
MANUAL Hard-Label: Correct=12, Over-called=4, Under-triaged=0

==========================================
=== SANITY FLAGS ===
==========================================
  - FLAG: Alpha 0.077 on chantest has avg set size > 2.0 (2.19)
  - FLAG: Alpha 0.077 on manual has avg set size > 2.0 (2.50)
  - FLAG: Alpha 0.1 on chantest has avg set size > 2.0 (2.19)
  - FLAG: Alpha 0.1 on manual has avg set size > 2.0 (2.50)
  - FLAG: Alpha 0.15 on chantest has avg set size > 2.0 (2.19)
  - FLAG: Alpha 0.15 on manual has avg set size > 2.0 (2.50)
  - FLAG: Alpha 0.2 on chantest empirical coverage (0.7500) fell below nominal (0.8000)
  - FLAG: Alpha 0.2 on nanion empirical coverage (0.7500) fell below nominal (0.8000)
  - FLAG: Alpha 0.25 on nanion empirical coverage (0.6250) fell below nominal (0.7500)
  - FLAG: Alpha 0.25 on nanion still has 1 under-triaged drugs
  - FLAG: Alpha 0.3 on nanion empirical coverage (0.6250) fell below nominal (0.7000)
  - FLAG: Alpha 0.3 on nanion still has 1 under-triaged drugs
  - FLAG: Alpha 0.4 on nanion still has 1 under-triaged drugs
==========================================
=== ADDENDUM — BASELINES AND FRAGILITY ===
==========================================

=== STEP 0 — VERIFY THE SEVERITY LOGIC SURVIVED THE KeyError FIX ===
Code block from scripts/pipeline/run_step5_conformal.py that computes correct / over-called / under-triaged:

SEVERITY = {0: 2, 1: 1, 2: 0}

def get_action_severity(pred_set_classes):
    if len(pred_set_classes) == 0:
        return -1 # abstain
    max_sev = -1
    for c in pred_set_classes:
        if SEVERITY[c] > max_sev:
            max_sev = SEVERITY[c]
    return max_sev

def get_outcome(true_class, pred_set_classes):
    action_sev = get_action_severity(pred_set_classes)
    if action_sev == -1:
        return "abstain"
    true_sev = SEVERITY[true_class]
    if action_sev == true_sev:
        return "correct"
    elif action_sev > true_sev:
        return "over-called"
    else:
        return "under-triaged"

Confirmed it uses SEVERITY = {0:2, 1:1, 2:0} and not raw label integers.
--- Spot Check: disopyramide ---
True label: high (Severity: 2)
Prediction set: ['high', 'intermediate', 'low']
Most-severe class in set: high (Severity: 2)
Computed Outcome: correct
Recorded Outcome: correct
Match: PASS
--- Spot Check: loratadine ---
True label: low (Severity: 0)
Prediction set: ['high', 'intermediate', 'low']
Most-severe class in set: high (Severity: 2)
Computed Outcome: over-called
Recorded Outcome: over-called
Match: PASS
--- Spot Check: astemizole ---
True label: intermediate (Severity: 1)
Prediction set: ['high', 'intermediate']
Most-severe class in set: high (Severity: 2)
Computed Outcome: over-called
Recorded Outcome: over-called
Match: PASS

=== STEP 1 — OPERATING POINT AT alpha = 1/13 ===
alpha 0.076923 -> k = ceil(13 * (1 - 0.076923)) = 12
alpha 0.100000 -> k = ceil(13 * (1 - 0.100000)) = 12
alpha 0.150000 -> k = ceil(13 * (1 - 0.150000)) = 12
Identical q_hat across {1/13, 0.10, 0.15}: PASS
Identical prediction sets across all platforms: PASS

Adopting alpha = 1/13 (~0.076923) as the reported operating point.

--- Platform: CHANTEST ---
Nominal coverage (1-alpha): 0.9231
Empirical coverage: 0.9375
Signed gap: +0.0144
Correct: 6, Over-called: 10, Under-triaged: 0
Average set size: 2.19
All-three-class sets (Uninformative): 6

--- Platform: NANION ---
Nominal coverage (1-alpha): 0.9231
Empirical coverage: 0.9375
Signed gap: +0.0144
Correct: 8, Over-called: 8, Under-triaged: 0
Average set size: 2.00
All-three-class sets (Uninformative): 5

--- Platform: MANUAL ---
Nominal coverage (1-alpha): 0.9231
Empirical coverage: 1.0000
Signed gap: +0.0769
Correct: 6, Over-called: 10, Under-triaged: 0
Average set size: 2.50
All-three-class sets (Uninformative): 10

=== STEP 2 — TRIVIAL BASELINES ===
Platform   | Method          | Correct  | Over   | Under  | Avg Set Size
----------------------------------------------------------------------
chantest   | ALWAYS-HIGH     | 4        | 12     | 0      | 1.00
           | HARD LABEL      | 11       | 4      | 1      | 1.00
           | CONFORMAL       | 6        | 10     | 0      | 2.19
nanion     | ALWAYS-HIGH     | 4        | 12     | 0      | 1.00
           | HARD LABEL      | 10       | 5      | 1      | 1.00
           | CONFORMAL       | 8        | 8      | 0      | 2.00
manual     | ALWAYS-HIGH     | 4        | 12     | 0      | 1.00
           | HARD LABEL      | 12       | 4      | 0      | 1.00
           | CONFORMAL       | 6        | 10     | 0      | 2.50

On chantest, conformal gets 2 more drugs CORRECT than ALWAYS-HIGH.
On chantest, conformal converts 5 drugs from correct to over-called relative to HARD LABEL, and removes 1 under-triages in exchange.
On nanion, conformal gets 4 more drugs CORRECT than ALWAYS-HIGH.
On nanion, conformal converts 2 drugs from correct to over-called relative to HARD LABEL, and removes 1 under-triages in exchange.
On manual, conformal gets 2 more drugs CORRECT than ALWAYS-HIGH.
On manual, conformal converts 6 drugs from correct to over-called relative to HARD LABEL, and removes 0 under-triages in exchange.

=== STEP 3 — Q_HAT FRAGILITY ===

> [!WARNING]
> *The conclusion of this subsection was degenerate and is superseded by ADDENDUM 2 below.*

12 LODO scores sorted descending:
  diltiazem      : 0.986695
  sotalol        : 0.762830
  bepridil       : 0.703345
  dofetilide     : 0.630580
  verapamil      : 0.453940
  chlorpromazine : 0.408085
  ranolazine     : 0.255295
  mexiletine     : 0.236885
  ondansetron    : 0.163680
  cisapride      : 0.152855
  terfenadine    : 0.094280
  quinidine      : 0.045365

Recomputing q_hat (alpha=1/13, n=11) -> target rank = 12
Remove diltiazem       -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove sotalol         -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove bepridil        -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove dofetilide      -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove verapamil       -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove chlorpromazine  -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove ranolazine      -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove mexiletine      -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove ondansetron     -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove cisapride       -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove terfenadine     -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES
Remove quinidine       -> q_hat: 1.000000 | Chantest Recov: YES | Nanion Recov: YES

Min q_hat: 1.000000
Median q_hat: 1.000000
Max q_hat: 1.000000

Out of 12 leave-one-out variants, 12 preserve zero under-triage on ALL three platforms.

=== STEP 4 — COVERAGE MISSES ===

Platform: chantest
Drug Name: domperidone
True Label: intermediate
p_high: 0.986030
p_int:  0.003185
p_low:  0.010785
Threshold (1 - q_hat): 0.013305
Prediction Set: ['high']
Action Taken: high
Outcome: over-called
This miss coincided with an over-call.

Platform: nanion
Drug Name: domperidone
True Label: intermediate
p_high: 0.990000
p_int:  0.000000
p_low:  0.010000
Threshold (1 - q_hat): 0.013305
Prediction Set: ['high']
Action Taken: high
Outcome: over-called
This miss coincided with an over-call.

=== STEP 5 — SET-SIZE DISTRIBUTION ===

Platform: CHANTEST
Set size 1: 3 drugs
  Drugs: azimilide, domperidone, nifedipine
Set size 2: 7 drugs
  Drugs: astemizole, clarithromycin, droperidol, metoprolol, pimozide, risperidone, tamoxifen
Set size 3: 6 drugs
  Drugs: clozapine, disopyramide, ibutilide, loratadine, nitrendipine, vandetanib

Platform: NANION
Set size 1: 5 drugs
  Drugs: azimilide, clozapine, domperidone, nifedipine, nitrendipine
Set size 2: 6 drugs
  Drugs: astemizole, droperidol, pimozide, risperidone, tamoxifen, vandetanib
Set size 3: 5 drugs
  Drugs: clarithromycin, disopyramide, ibutilide, loratadine, metoprolol

Platform: MANUAL
Set size 1: 2 drugs
  Drugs: nifedipine, nitrendipine
Set size 2: 4 drugs
  Drugs: astemizole, azimilide, droperidol, tamoxifen
Set size 3: 10 drugs
  Drugs: clarithromycin, clozapine, disopyramide, domperidone, ibutilide, loratadine, metoprolol, pimozide, risperidone, vandetanib

==========================================
=== SANITY FLAGS ===
==========================================
  - No new sanity flags in addendum.
==========================================
=== ADDENDUM 2 — CORRECTED FRAGILITY ===
==========================================

=== STEP 1 — RETRACT THE DEGENERATE FRAGILITY RESULT ===
Example variant (removing diltiazem):
n: 11, alpha: 0.076923, rank: 12
Does rank exceed n? YES
Resulting q_hat: 1.000000 (+inf formally in conformal)
Inclusion threshold (1 - q_hat): 0.000000
Because the threshold is 0, any predicted probability p >= 0 satisfies 1 - p <= q_hat (1 - p <= 1.0).
Resulting set sizes across all 16 test drugs: 3 for every drug.
Resulting action for every drug: 'high' (most severe class in the set).
Is this equivalent to assigning every drug the most severe class regardless of its probabilities? YES, the evidence shows that a q_hat of 1.0 produces universally uninformative sets ['high', 'intermediate', 'low'] for every drug, forcing the fallback rule to blindly select 'high' across the board.

=== STEP 2 — CORRECTED FRAGILITY (smallest feasible alpha at each n) ===
Smallest feasible alpha at n=11 is 1/12.
Verification: ceil(12 * (1 - 1/12)) == 11
Removed: diltiazem       | Score: 0.986695 | q_hat_LOO: 0.762830 | Threshold: 0.237170
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: sotalol         | Score: 0.762830 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: bepridil        | Score: 0.703345 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: dofetilide      | Score: 0.630580 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: verapamil       | Score: 0.453940 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: chlorpromazine  | Score: 0.408085 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: ranolazine      | Score: 0.255295 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: mexiletine      | Score: 0.236885 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: ondansetron     | Score: 0.163680 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: cisapride       | Score: 0.152855 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: terfenadine     | Score: 0.094280 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)
Removed: quinidine       | Score: 0.045365 | q_hat_LOO: 0.986695 | Threshold: 0.013305
   -> Chantest Recovered: YES (requires >= 0.561270)
   -> Nanion Recovered:   YES (requires >= 0.735545)

Min q_hat_LOO: 0.762830
Median q_hat_LOO: 0.986695
Max q_hat_LOO: 0.986695

Out of 12 variants, 12 preserve zero under-triage on ALL three platforms.
The tightest surviving margin was produced by removing diltiazem.
This dropped q_hat_LOO to 0.762830.

=== STEP 3 — RECOMPUTE SETS UNDER THE TIGHTEST VARIANT ===
Using Tightest Variant q_hat = 0.762830 vs Original q_hat = 0.986695

Platform: CHANTEST
  ORIGINAL (q=0.986695): Correct=6, Over=10, Under=0, Abstain=0
      Avg Set Size: 2.19 | Dist: Size 1: 3, Size 2: 7, Size 3: 6
  TIGHTEST (q=0.762830): Correct=11, Over=5, Under=0, Abstain=0
      Avg Set Size: 1.12 | Dist: Size 1: 14, Size 2: 2, Size 3: 0

Platform: NANION
  ORIGINAL (q=0.986695): Correct=8, Over=8, Under=0, Abstain=0
      Avg Set Size: 2.00 | Dist: Size 1: 5, Size 2: 6, Size 3: 5
  TIGHTEST (q=0.762830): Correct=11, Over=5, Under=0, Abstain=0
      Avg Set Size: 1.12 | Dist: Size 1: 14, Size 2: 2, Size 3: 0

Platform: MANUAL
  ORIGINAL (q=0.986695): Correct=6, Over=10, Under=0, Abstain=0
      Avg Set Size: 2.50 | Dist: Size 1: 2, Size 2: 4, Size 3: 10
  TIGHTEST (q=0.762830): Correct=11, Over=5, Under=0, Abstain=0
      Avg Set Size: 1.38 | Dist: Size 1: 10, Size 2: 6, Size 3: 0

This shows how heavily the safety of the operating point depends on a single calibration drug (diltiazem) holding up the q_hat value.

=== STEP 4 — DOMPERIDONE DETAIL AT FULL PRECISION ===

--- domperidone on chantest ---
p_high: 0.986030
p_intermediate: 0.003185
p_low: 0.010785
Threshold (1 - q_hat): 0.013305
Classes that cleared threshold (1 - p <= 0.013305): ['high']
Resulting Set: ['high']
Action: high
Outcome: over-called
CONFIRMED: The true class (intermediate) probability (0.003185) means 1 - p = 0.996815, which falls below the threshold, excluding it from the set.

--- domperidone on nanion ---
p_high: 0.990000
p_intermediate: 0.000000
p_low: 0.010000
Threshold (1 - q_hat): 0.013305
Classes that cleared threshold (1 - p <= 0.013305): ['high']
Resulting Set: ['high']
Action: high
Outcome: over-called
CONFIRMED: The true class (intermediate) probability (0.000000) means 1 - p = 1.000000, which falls below the threshold, excluding it from the set.

=== STEP 5 — REPAIR THE SANITY FLAGS ===
Recomputing flags at alpha = 1/13 (q_hat = 0.986695):

Platform: CHANTEST
FLAG: Average set size is 2.19 (> 2.0).
CLEAR: Empirical coverage (0.937500) >= nominal (0.923077).
FLAG: 37.5% of drugs (6) receive size-3 sets (> 25%).
CLEAR: No remaining under-triages.
CLEAR: Hard-label comparison does not show purely wasteful sacrifices (Removed under-triages: 1).

Platform: NANION
CLEAR: Average set size is 2.00 (<= 2.0).
CLEAR: Empirical coverage (0.937500) >= nominal (0.923077).
FLAG: 31.2% of drugs (5) receive size-3 sets (> 25%).
CLEAR: No remaining under-triages.
CLEAR: Hard-label comparison does not show purely wasteful sacrifices (Removed under-triages: 1).

Platform: MANUAL
FLAG: Average set size is 2.50 (> 2.0).
CLEAR: Empirical coverage (1.000000) >= nominal (0.923077).
FLAG: 62.5% of drugs (10) receive size-3 sets (> 25%).
CLEAR: No remaining under-triages.
FLAG: Conformal removed ZERO under-triages relative to hard labels, but sacrificed 6 correct drugs by converting them to over-calls for zero safety gain.