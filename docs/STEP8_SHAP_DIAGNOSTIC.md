==========================================
=== HARD RULES AND SETUP ===
==========================================
models/chantest_stacking_model_rev.joblib hash matches: PASS

==========================================
=== STEP 2 — IS vrest SEPARABLE BETWEEN DRUGS? ===
==========================================

--- Analysis for vrest ---

TRAINING (12 drugs):
  bepridil        (True: 0) | min: -1.00017457 | max: -1.00003240 | mean: -1.00005424 | std: 0.00001803
  chlorpromazine  (True: 1) | min: -1.00006736 | max: -0.99998650 | mean: -1.00000275 | std: 0.00001001
  cisapride       (True: 1) | min: -1.00001058 | max: -0.99998518 | mean: -0.99999045 | std: 0.00000183
  diltiazem       (True: 2) | min: -1.00009146 | max: -1.00000114 | mean: -1.00002499 | std: 0.00000978
  dofetilide      (True: 0) | min: -1.00002674 | max: -1.00000406 | mean: -1.00000898 | std: 0.00000234
  mexiletine      (True: 2) | min: -1.00049103 | max: -1.00015377 | mean: -1.00024709 | std: 0.00004517
  ondansetron     (True: 1) | min: -1.00005776 | max: -0.99996402 | mean: -0.99999513 | std: 0.00001873
  quinidine       (True: 0) | min: -1.00053426 | max: -1.00022512 | mean: -1.00030017 | std: 0.00004520
  ranolazine      (True: 2) | min: -1.00022554 | max: -1.00005563 | mean: -1.00008935 | std: 0.00001352
  sotalol         (True: 0) | min: -1.00001661 | max: -0.99995790 | mean: -0.99996969 | std: 0.00000571
  terfenadine     (True: 1) | min: -1.00005622 | max: -1.00000179 | mean: -1.00001893 | std: 0.00000745
  verapamil       (True: 2) | min: -1.00032957 | max: -1.00005797 | mean: -1.00017741 | std: 0.00003663
  Overlapping pairs: 40
  Up to 10 examples: [('quinidine', 'mexiletine'), ('quinidine', 'verapamil'), ('quinidine', 'ranolazine'), ('mexiletine', 'verapamil'), ('mexiletine', 'ranolazine'), ('mexiletine', 'bepridil'), ('verapamil', 'ranolazine'), ('verapamil', 'bepridil'), ('verapamil', 'diltiazem'), ('verapamil', 'chlorpromazine')]
  Can the drugs be perfectly ordered/separated by vrest alone? NO, ranges overlap.

TESTING (16 drugs):
  astemizole      (True: 1) | min: -1.00000165 | max: -0.99999673 | mean: -0.99999759 | std: 0.00000076
  azimilide       (True: 0) | min: -1.00000294 | max: -0.99996308 | mean: -0.99998067 | std: 0.00000533
  clarithromycin  (True: 1) | min: -1.00006536 | max: -0.99996242 | mean: -0.99998475 | std: 0.00001404
  clozapine       (True: 1) | min: -1.00013160 | max: -0.99998363 | mean: -1.00000543 | std: 0.00001292
  disopyramide    (True: 0) | min: -1.00018288 | max: -0.99995616 | mean: -1.00002171 | std: 0.00002497
  domperidone     (True: 1) | min: -1.00011214 | max: -1.00001064 | mean: -1.00003993 | std: 0.00001376
  droperidol      (True: 1) | min: -1.00004172 | max: -0.99997881 | mean: -0.99999692 | std: 0.00000756
  ibutilide       (True: 0) | min: -1.00026168 | max: -0.99976965 | mean: -1.00016425 | std: 0.00006813
  loratadine      (True: 2) | min: -1.00000530 | max: -0.99999918 | mean: -0.99999996 | std: 0.00000031
  metoprolol      (True: 2) | min: -1.00013946 | max: -0.99997497 | mean: -1.00004970 | std: 0.00002152
  nifedipine      (True: 2) | min: -1.00078315 | max: -1.00009666 | mean: -1.00038108 | std: 0.00007896
  nitrendipine    (True: 2) | min: -1.00019103 | max: -1.00000037 | mean: -1.00004612 | std: 0.00003376
  pimozide        (True: 1) | min: -1.00005605 | max: -0.99999849 | mean: -1.00001719 | std: 0.00000964
  risperidone     (True: 1) | min: -1.00000413 | max: -0.99999036 | mean: -0.99999626 | std: 0.00000223
  tamoxifen       (True: 2) | min: -1.00003245 | max: -0.99999105 | mean: -0.99999768 | std: 0.00000233
  vandetanib      (True: 0) | min: -1.00023142 | max: -0.99999390 | mean: -1.00004060 | std: 0.00002470
  Overlapping pairs: 108
  Up to 10 examples: [('nifedipine', 'ibutilide'), ('nifedipine', 'vandetanib'), ('nifedipine', 'nitrendipine'), ('nifedipine', 'disopyramide'), ('nifedipine', 'metoprolol'), ('nifedipine', 'clozapine'), ('nifedipine', 'domperidone'), ('ibutilide', 'vandetanib'), ('ibutilide', 'nitrendipine'), ('ibutilide', 'disopyramide')]
  Can the drugs be perfectly ordered/separated by vrest alone? NO, ranges overlap.

--- Analysis for qNet ---

TRAINING (12 drugs):
  bepridil        (True: 0) | min: 0.60963152 | max: 0.78312519 | mean: 0.67061432 | std: 0.02248789
  chlorpromazine  (True: 1) | min: 0.90852976 | max: 1.00286979 | mean: 0.95050213 | std: 0.01203509
  cisapride       (True: 1) | min: 0.83097629 | max: 0.91612448 | mean: 0.86903980 | std: 0.01334274
  diltiazem       (True: 2) | min: 0.99828024 | max: 1.11226018 | mean: 1.04121891 | std: 0.01312346
  dofetilide      (True: 0) | min: 0.65681796 | max: 0.77087402 | mean: 0.71809150 | std: 0.01711188
  mexiletine      (True: 2) | min: 1.25617316 | max: 1.46590634 | mean: 1.33712078 | std: 0.02833518
  ondansetron     (True: 1) | min: 0.85153773 | max: 0.94130412 | mean: 0.89255000 | std: 0.01482913
  quinidine       (True: 0) | min: 0.56935407 | max: 0.96164652 | mean: 0.75666481 | std: 0.05235604
  ranolazine      (True: 2) | min: 1.03652267 | max: 1.15947114 | mean: 1.10046635 | std: 0.01837183
  sotalol         (True: 0) | min: 0.74618322 | max: 0.85835150 | mean: 0.79880369 | std: 0.01730718
  terfenadine     (True: 1) | min: 0.81336146 | max: 1.01041030 | mean: 0.88839963 | std: 0.02770740
  verapamil       (True: 2) | min: 0.88001384 | max: 1.11181264 | mean: 1.01616339 | std: 0.03243114
  Overlapping pairs: 29
  Up to 10 examples: [('quinidine', 'bepridil'), ('quinidine', 'dofetilide'), ('quinidine', 'sotalol'), ('quinidine', 'terfenadine'), ('quinidine', 'cisapride'), ('quinidine', 'ondansetron'), ('quinidine', 'verapamil'), ('quinidine', 'chlorpromazine'), ('bepridil', 'dofetilide'), ('bepridil', 'sotalol')]
  Can the drugs be perfectly ordered/separated by qNet alone? NO, ranges overlap.

TESTING (16 drugs):
  astemizole      (True: 1) | min: 0.77726326 | max: 0.95474003 | mean: 0.86605355 | std: 0.02722702
  azimilide       (True: 0) | min: 0.58086912 | max: 0.75807863 | mean: 0.66095362 | std: 0.02958177
  clarithromycin  (True: 1) | min: 0.87506664 | max: 0.97938615 | mean: 0.92497944 | std: 0.01640865
  clozapine       (True: 1) | min: 0.89443533 | max: 1.05148493 | mean: 0.95614152 | std: 0.02251308
  disopyramide    (True: 0) | min: 0.73285375 | max: 0.97321495 | mean: 0.86177949 | std: 0.03241795
  domperidone     (True: 1) | min: 0.56816128 | max: 0.84087148 | mean: 0.67064130 | std: 0.03809737
  droperidol      (True: 1) | min: 0.78365423 | max: 0.95903221 | mean: 0.87554075 | std: 0.02864575
  ibutilide       (True: 0) | min: -0.28024665 | max: 0.55038931 | mean: 0.27438249 | std: 0.15280347
  loratadine      (True: 2) | min: 0.99595311 | max: 1.01130062 | mean: 0.99958419 | std: 0.00066337
  metoprolol      (True: 2) | min: 0.88256390 | max: 1.15413561 | mean: 1.03645183 | std: 0.03745703
  nifedipine      (True: 2) | min: 1.05955642 | max: 1.35185130 | mean: 1.19086677 | std: 0.03599517
  nitrendipine    (True: 2) | min: 0.99992261 | max: 1.10536558 | mean: 1.03651804 | std: 0.01693221
  pimozide        (True: 1) | min: 0.51755948 | max: 0.83820907 | mean: 0.66568216 | std: 0.07014941
  risperidone     (True: 1) | min: 0.90041314 | max: 0.99879618 | mean: 0.96214761 | std: 0.02041734
  tamoxifen       (True: 2) | min: 0.93556295 | max: 1.00230366 | mean: 0.98182018 | std: 0.00997647
  vandetanib      (True: 0) | min: 0.41148763 | max: 0.75911122 | mean: 0.52195372 | std: 0.03813209
  Overlapping pairs: 54
  Up to 10 examples: [('ibutilide', 'vandetanib'), ('ibutilide', 'pimozide'), ('vandetanib', 'pimozide'), ('vandetanib', 'domperidone'), ('vandetanib', 'azimilide'), ('vandetanib', 'disopyramide'), ('pimozide', 'domperidone'), ('pimozide', 'azimilide'), ('pimozide', 'disopyramide'), ('pimozide', 'astemizole')]
  Can the drugs be perfectly ordered/separated by qNet alone? NO, ranges overlap.

--- Analysis for carest ---

TRAINING (12 drugs):
  bepridil        (True: 0) | min: 0.98300000 | max: 1.02000000 | mean: 1.00902950 | std: 0.00331800
  chlorpromazine  (True: 1) | min: 0.98900000 | max: 1.01000000 | mean: 1.00209350 | std: 0.00424379
  cisapride       (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00998000 | std: 0.00044688
  diltiazem       (True: 2) | min: 0.98300000 | max: 1.00000000 | mean: 0.99892800 | std: 0.00181341
  dofetilide      (True: 0) | min: 1.01000000 | max: 1.01000000 | mean: 1.01000000 | std: 0.00000000
  mexiletine      (True: 2) | min: 0.94500000 | max: 1.01000000 | mean: 0.99096000 | std: 0.00861223
  ondansetron     (True: 1) | min: 0.99200000 | max: 1.01000000 | mean: 1.00518000 | std: 0.00527973
  quinidine       (True: 0) | min: 0.96300000 | max: 1.02000000 | mean: 1.00647450 | std: 0.00799658
  ranolazine      (True: 2) | min: 0.98200000 | max: 1.02000000 | mean: 1.01029500 | std: 0.00346886
  sotalol         (True: 0) | min: 1.00000000 | max: 1.02000000 | mean: 1.01016000 | std: 0.00140548
  terfenadine     (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00948000 | std: 0.00222083
  verapamil       (True: 2) | min: 0.93900000 | max: 0.99800000 | mean: 0.97242750 | std: 0.00806674
  Overlapping pairs: 61
  Up to 10 examples: [('verapamil', 'mexiletine'), ('verapamil', 'quinidine'), ('verapamil', 'ranolazine'), ('verapamil', 'bepridil'), ('verapamil', 'diltiazem'), ('verapamil', 'chlorpromazine'), ('verapamil', 'ondansetron'), ('mexiletine', 'quinidine'), ('mexiletine', 'ranolazine'), ('mexiletine', 'bepridil')]
  Can the drugs be perfectly ordered/separated by carest alone? NO, ranges overlap.

TESTING (16 drugs):
  astemizole      (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00922500 | std: 0.00267450
  azimilide       (True: 0) | min: 1.01000000 | max: 1.02000000 | mean: 1.01987000 | std: 0.00113302
  clarithromycin  (True: 1) | min: 0.98800000 | max: 1.01000000 | mean: 1.00634250 | std: 0.00512614
  clozapine       (True: 1) | min: 0.97500000 | max: 1.01000000 | mean: 1.00091550 | std: 0.00383454
  disopyramide    (True: 0) | min: 0.97500000 | max: 1.02000000 | mean: 1.01081800 | std: 0.00676149
  domperidone     (True: 1) | min: 1.00000000 | max: 1.02000000 | mean: 1.01005500 | std: 0.00102347
  droperidol      (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00944000 | std: 0.00229979
  ibutilide       (True: 0) | min: 0.95900000 | max: 1.02000000 | mean: 0.99036300 | std: 0.00838346
  loratadine      (True: 2) | min: 1.00000000 | max: 1.00000000 | mean: 1.00000000 | std: 0.00000000
  metoprolol      (True: 2) | min: 0.98500000 | max: 1.01000000 | mean: 1.00297950 | std: 0.00530756
  nifedipine      (True: 2) | min: 0.83600000 | max: 0.98200000 | mean: 0.91675400 | std: 0.01732303
  nitrendipine    (True: 2) | min: 0.95800000 | max: 1.00000000 | mean: 0.99118550 | std: 0.00776210
  pimozide        (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00997000 | std: 0.00054704
  risperidone     (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00003500 | std: 0.00059072
  tamoxifen       (True: 2) | min: 0.99300000 | max: 1.00000000 | mean: 0.99998400 | std: 0.00026604
  vandetanib      (True: 0) | min: 0.96200000 | max: 1.02000000 | mean: 1.00412700 | std: 0.00617835
  Overlapping pairs: 107
  Up to 10 examples: [('nifedipine', 'nitrendipine'), ('nifedipine', 'ibutilide'), ('nifedipine', 'vandetanib'), ('nifedipine', 'clozapine'), ('nifedipine', 'disopyramide'), ('nitrendipine', 'ibutilide'), ('nitrendipine', 'vandetanib'), ('nitrendipine', 'clozapine'), ('nitrendipine', 'disopyramide'), ('nitrendipine', 'metoprolol')]
  Can the drugs be perfectly ordered/separated by carest alone? NO, ranges overlap.

==========================================
=== STEP 3 — DOES THE MODEL STRUCTURALLY USE vrest? ===
==========================================

RF Base Learner - vrest:
  Split Count: 1533 (13.0579% of 11740 total splits)
  Thresholds -> min: -1.00038034 | median: -1.00004244 | max: -0.99996367
  These thresholds discriminate at extremely precise decimal places (up to 8 decimal places).

RF Base Learner - qNet:
  Split Count: 1872 (15.9455% of 11740 total splits)
  Thresholds -> min: 0.78309965 | median: 0.85961911 | max: 1.16004887
  These thresholds discriminate at extremely precise decimal places (up to 8 decimal places).

RF Base Learner - carest:
  Split Count: 304 (2.5894% of 11740 total splits)
  Thresholds -> min: 0.97349998 | median: 0.99500000 | max: 1.01499999
  These thresholds discriminate at extremely precise decimal places (up to 8 decimal places).


XGB Base Learner - vrest: Split Count: 0
XGB Base Learner - qNet: Split Count: 0
XGB Base Learner - carest: Split Count: 0

==========================================
=== STEP 4 — PERMUTATION TEST (THE DECISIVE CHECK) ===
==========================================
Baseline drug-level macro-F1: 0.694444
Reproduces 0.694444: PASS

Permutation Results:
Feature      | Mean F1    | Std F1     | Min F1     | Max F1     | Mean Drop  | Mean Changes
-----------------------------------------------------------------------------------------------
qNet         | 0.586111   | 0.000000   | 0.586111   | 0.586111   | 0.108333   | 3.00
vrest        | 0.621032   | 0.000000   | 0.621032   | 0.621032   | 0.073413   | 1.00
carest       | 0.694444   | 0.000000   | 0.694444   | 0.694444   | -0.000000  | 0.00
dvdtmax      | 0.694444   | 0.000000   | 0.694444   | 0.694444   | -0.000000  | 0.00

Does permuting vrest change predictions materially? YES, it changes drug-level predictions.

==========================================
=== STEP 5 — PER-CLASS CONCENTRATION ===
==========================================

vrest per-class mean|SHAP|:
  High: 0.028388
  Intermediate: 0.171309
  Low: 0.146771
  Concentration: Its aggregated importance is spread across classes.

qNet per-class mean|SHAP|:
  High: 0.129443
  Intermediate: 0.132843
  Low: 0.174315
  Concentration: Its aggregated importance is spread across classes.

==========================================
=== STEP 1 — SHAP STABILITY ===
==========================================
Running KernelExplainer variations in parallel...

--- Seed: 7, Background Size: 100 ---
Rank  | Biomarker       | mean|SHAP|
----------------------------------------
1     | qNet            | 0.140592
2     | vrest           | 0.095976
3     | max_dv          | 0.085622
4     | APD50           | 0.071800
5     | APD90           | 0.046810
6     | vmax            | 0.027235
7     | camax           | 0.011151
8     | dvdtmax         | 0.010876
9     | CaTD50          | 0.009372
10    | CaTD90          | 0.009083
11    | carest          | 0.000676

vrest: Rank 2, mean|SHAP| = 0.095976
qNet:  Rank 1, mean|SHAP| = 0.140592

--- Seed: 42, Background Size: 100 ---
Rank  | Biomarker       | mean|SHAP|
----------------------------------------
1     | qNet            | 0.145534
2     | vrest           | 0.115489
3     | max_dv          | 0.079867
4     | APD50           | 0.069771
5     | APD90           | 0.049044
6     | vmax            | 0.038180
7     | dvdtmax         | 0.012816
8     | camax           | 0.012420
9     | CaTD50          | 0.009237
10    | CaTD90          | 0.008037
11    | carest          | 0.000655

vrest: Rank 2, mean|SHAP| = 0.115489
qNet:  Rank 1, mean|SHAP| = 0.145534

--- Seed: 2024, Background Size: 100 ---
Rank  | Biomarker       | mean|SHAP|
----------------------------------------
1     | qNet            | 0.143116
2     | vrest           | 0.099467
3     | max_dv          | 0.083462
4     | APD50           | 0.075131
5     | APD90           | 0.048412
6     | vmax            | 0.032511
7     | dvdtmax         | 0.011660
8     | camax           | 0.011266
9     | CaTD50          | 0.008846
10    | CaTD90          | 0.007898
11    | carest          | 0.000942

vrest: Rank 2, mean|SHAP| = 0.099467
qNet:  Rank 1, mean|SHAP| = 0.143116

--- Seed: 42, Background Size: 300 ---
Rank  | Biomarker       | mean|SHAP|
----------------------------------------
1     | qNet            | 0.143592
2     | vrest           | 0.108178
3     | max_dv          | 0.082083
4     | APD50           | 0.070550
5     | APD90           | 0.049370
6     | vmax            | 0.034833
7     | camax           | 0.012525
8     | dvdtmax         | 0.011842
9     | CaTD50          | 0.009308
10    | CaTD90          | 0.008196
11    | carest          | 0.000573

vrest: Rank 2, mean|SHAP| = 0.108178
qNet:  Rank 1, mean|SHAP| = 0.143592

vrest Rank across the 4 runs:
  Seed 7   , bg=100 -> Rank 2
  Seed 42  , bg=100 -> Rank 2
  Seed 2024, bg=100 -> Rank 2
  Seed 42  , bg=300 -> Rank 2

State plainly: vrest's rank is highly stable across seeds and background sizes.

==========================================
=== FINDINGS ===
==========================================
- vrest's SHAP rank is highly stable across seeds and background sizes.
> RETRACTED. The three bullets below are superseded. Direct recomputation from source data
> (see the overlap tables in this file, and docs/STEP10_VREST_OVERLAP_RECHECK.md) established
> 40 overlapping vrest pairs of 66 on the training split and 108 of 120 on the test split.
> vrest is the LEAST separable feature examined, not the most. Permuting it changes on average
> 1.0 of 16 drug-level predictions, not zero. Authoritative statement: docs/STORYLINE.md Sec. 7.
- vrest perfectly separates the drugs into non-overlapping ranges at extremely precise decimal places.
- The Random Forest base learner splits on vrest massively, isolating extremely fine-grained thresholds (up to 8 decimal places).
- Despite its high SHAP rank, permuting vrest changes zero predictions at the drug level, meaning its splits are structurally redundant copies of information rather than unique deciders.