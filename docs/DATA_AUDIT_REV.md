==========================================
=== STEP 1: LOCATE THE FILES ===
==========================================
File: TrainAI\data\chantest_training_scaled_rev.csv | Size: 3660597 bytes | Modified: 2026-07-25 09:11:54
File: TrainAI\data\chantest_testing_scaled_rev.csv | Size: 4863261 bytes | Modified: 2026-07-25 09:11:51
File: TrainAI\data\nanion_training_scaled_rev.csv | Size: 4282518 bytes | Modified: 2026-07-25 09:12:08
File: TrainAI\data\nanion_testing_scaled_rev.csv | Size: 5006907 bytes | Modified: 2026-07-25 09:12:05
File: TrainAI\data\manual_training_scaled_rev.csv | Size: 3675246 bytes | Modified: 2026-07-25 09:12:02
File: TrainAI\data\manual_testing_scaled_rev.csv | Size: 4944688 bytes | Modified: 2026-07-25 09:11:57

Chantest TESTING _rev file present: YES

MISSING PLATFORMS: None (All 3 platforms chantest, nanion, manual have _rev files)

==========================================
=== STEP 2: CANONICAL 11 BIOMARKER NAMES ===
==========================================
Derived 11 Canonical Biomarker Features (in exact model input order):
   1. qNet
   2. dvdtmax
   3. vmax
   4. vrest
   5. APD50
   6. APD90
   7. max_dv
   8. camax
   9. carest
  10. CaTD50
  11. CaTD90

Derivation Sources:
  1. plus_ultra.ipynb: Cell 24 (Line 29) & Cell 30 (Line 22): `BIOMARKERS = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']`
  2. hf_space/app.py: Line 57: `features = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']`
  3. conformal_diag.py: Line 14: `biomarkers = ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']`

==========================================
=== STEP 3: PER-FILE COLUMN VERIFICATION ===
==========================================

--- FILE: chantest_training_scaled_rev.csv ---
Shape: (24000, 14)
Header columns (14): ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90', 'drug_name', 'Sample_ID', 'label']
Biomarker Status:
  qNet: PRESENT
  dvdtmax: PRESENT
  vmax: PRESENT
  vrest: PRESENT
  APD50: PRESENT
  APD90: PRESENT
  max_dv: PRESENT
  camax: PRESENT
  carest: PRESENT
  CaTD50: PRESENT
  CaTD90: PRESENT
drug_name: PRESENT
Label column found: 'label' | Unique values & counts: {'high': 8000, 'intermediate': 8000, 'low': 8000}
Extra columns: ['Sample_ID']
Column dtypes:
  qNet: float64
  dvdtmax: float64
  vmax: float64
  vrest: float64
  APD50: float64
  APD90: float64
  max_dv: float64
  camax: float64
  carest: float64
  CaTD50: float64
  CaTD90: float64
  drug_name: str
  Sample_ID: int64
  label: str
Nonzero Invalid Cell Counts:
  drug_name: NaN=0, Inf=0, Empty=0, NonNumericStr=24000
  label: NaN=0, Inf=0, Empty=0, NonNumericStr=24000

--- FILE: chantest_testing_scaled_rev.csv ---
Shape: (32000, 14)
Header columns (14): ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90', 'drug_name', 'Sample_ID', 'label']
Biomarker Status:
  qNet: PRESENT
  dvdtmax: PRESENT
  vmax: PRESENT
  vrest: PRESENT
  APD50: PRESENT
  APD90: PRESENT
  max_dv: PRESENT
  camax: PRESENT
  carest: PRESENT
  CaTD50: PRESENT
  CaTD90: PRESENT
drug_name: PRESENT
Label column found: 'label' | Unique values & counts: {'intermediate': 14000, 'low': 10000, 'high': 8000}
Extra columns: ['Sample_ID']
Column dtypes:
  qNet: float64
  dvdtmax: float64
  vmax: float64
  vrest: float64
  APD50: float64
  APD90: float64
  max_dv: float64
  camax: float64
  carest: float64
  CaTD50: float64
  CaTD90: float64
  drug_name: str
  Sample_ID: int64
  label: str
Nonzero Invalid Cell Counts:
  drug_name: NaN=0, Inf=0, Empty=0, NonNumericStr=32000
  label: NaN=0, Inf=0, Empty=0, NonNumericStr=32000

--- FILE: nanion_training_scaled_rev.csv ---
Shape: (24000, 15)
Header columns (15): ['drug_name', 'Sample_ID', 'qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90', 'label', 'Training_or_testing']
Biomarker Status:
  qNet: PRESENT
  dvdtmax: PRESENT
  vmax: PRESENT
  vrest: PRESENT
  APD50: PRESENT
  APD90: PRESENT
  max_dv: PRESENT
  camax: PRESENT
  carest: PRESENT
  CaTD50: PRESENT
  CaTD90: PRESENT
drug_name: PRESENT
Label column found: 'label' | Unique values & counts: {'high': 8000, 'intermediate': 8000, 'low': 8000}
Extra columns: ['Sample_ID', 'Training_or_testing']
Column dtypes:
  drug_name: str
  Sample_ID: int64
  qNet: float64
  dvdtmax: float64
  vmax: float64
  vrest: float64
  APD50: float64
  APD90: float64
  max_dv: float64
  camax: float64
  carest: float64
  CaTD50: float64
  CaTD90: float64
  label: str
  Training_or_testing: str
Nonzero Invalid Cell Counts:
  drug_name: NaN=0, Inf=0, Empty=0, NonNumericStr=24000
  label: NaN=0, Inf=0, Empty=0, NonNumericStr=24000
  Training_or_testing: NaN=0, Inf=0, Empty=0, NonNumericStr=24000

--- FILE: nanion_testing_scaled_rev.csv ---
Shape: (32000, 15)
Header columns (15): ['drug_name', 'Sample_ID', 'qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90', 'label', 'Training_or_testing']
Biomarker Status:
  qNet: PRESENT
  dvdtmax: PRESENT
  vmax: PRESENT
  vrest: PRESENT
  APD50: PRESENT
  APD90: PRESENT
  max_dv: PRESENT
  camax: PRESENT
  carest: PRESENT
  CaTD50: PRESENT
  CaTD90: PRESENT
drug_name: PRESENT
Label column found: 'label' | Unique values & counts: {'intermediate': 14000, 'low': 10000, 'high': 8000}
Extra columns: ['Sample_ID', 'Training_or_testing']
Column dtypes:
  drug_name: str
  Sample_ID: int64
  qNet: float64
  dvdtmax: float64
  vmax: float64
  vrest: float64
  APD50: float64
  APD90: float64
  max_dv: float64
  camax: float64
  carest: float64
  CaTD50: float64
  CaTD90: float64
  label: str
  Training_or_testing: str
Nonzero Invalid Cell Counts:
  drug_name: NaN=0, Inf=0, Empty=0, NonNumericStr=32000
  label: NaN=0, Inf=0, Empty=0, NonNumericStr=32000
  Training_or_testing: NaN=0, Inf=0, Empty=0, NonNumericStr=32000

--- FILE: manual_training_scaled_rev.csv ---
Shape: (24000, 14)
Header columns (14): ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90', 'drug_name', 'Sample_ID', 'label']
Biomarker Status:
  qNet: PRESENT
  dvdtmax: PRESENT
  vmax: PRESENT
  vrest: PRESENT
  APD50: PRESENT
  APD90: PRESENT
  max_dv: PRESENT
  camax: PRESENT
  carest: PRESENT
  CaTD50: PRESENT
  CaTD90: PRESENT
drug_name: PRESENT
Label column found: 'label' | Unique values & counts: {'high': 8000, 'intermediate': 8000, 'low': 8000}
Extra columns: ['Sample_ID']
Column dtypes:
  qNet: float64
  dvdtmax: float64
  vmax: float64
  vrest: float64
  APD50: str
  APD90: str
  max_dv: str
  camax: float64
  carest: float64
  CaTD50: float64
  CaTD90: float64
  drug_name: str
  Sample_ID: int64
  label: str
FLAG / ERROR: Non-numeric biomarker columns: ['APD50', 'APD90', 'max_dv']
  -> Column 'APD50': 25 non-numeric cell(s) with values {'#VALUE!': 25} in drug(s) ['mexiletine']
  -> Column 'APD90': 25 non-numeric cell(s) with values {'#VALUE!': 25} in drug(s) ['mexiletine']
  -> Column 'max_dv': 25 non-numeric cell(s) with values {'#VALUE!': 25} in drug(s) ['mexiletine']
Nonzero Invalid Cell Counts:
  APD50: NaN=0, Inf=0, Empty=0, NonNumericStr=25
  APD90: NaN=0, Inf=0, Empty=0, NonNumericStr=25
  max_dv: NaN=0, Inf=0, Empty=0, NonNumericStr=25
  drug_name: NaN=0, Inf=0, Empty=0, NonNumericStr=24000
  label: NaN=0, Inf=0, Empty=0, NonNumericStr=24000

--- FILE: manual_testing_scaled_rev.csv ---
Shape: (32000, 14)
Header columns (14): ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90', 'drug_name', 'Sample_ID', 'label']
Biomarker Status:
  qNet: PRESENT
  dvdtmax: PRESENT
  vmax: PRESENT
  vrest: PRESENT
  APD50: PRESENT
  APD90: PRESENT
  max_dv: PRESENT
  camax: PRESENT
  carest: PRESENT
  CaTD50: PRESENT
  CaTD90: PRESENT
drug_name: PRESENT
Label column found: 'label' | Unique values & counts: {'intermediate': 14000, 'low': 10000, 'high': 8000}
Extra columns: ['Sample_ID']
Column dtypes:
  qNet: float64
  dvdtmax: float64
  vmax: float64
  vrest: float64
  APD50: float64
  APD90: float64
  max_dv: float64
  camax: float64
  carest: float64
  CaTD50: float64
  CaTD90: str
  drug_name: str
  Sample_ID: int64
  label: str
FLAG / ERROR: Non-numeric biomarker columns: ['CaTD90']
  -> Column 'CaTD90': 1 non-numeric cell(s) with values {'#VALUE!': 1} in drug(s) ['ibutilide']
Nonzero Invalid Cell Counts:
  CaTD90: NaN=0, Inf=0, Empty=0, NonNumericStr=1
  drug_name: NaN=0, Inf=0, Empty=0, NonNumericStr=32000
  label: NaN=0, Inf=0, Empty=0, NonNumericStr=32000

==========================================
=== STEP 4: DRUG PARTITION ===
==========================================

--- PLATFORM: CHANTEST ---
Train drugs count: 12 | Sorted list: ['bepridil', 'chlorpromazine', 'cisapride', 'diltiazem', 'dofetilide', 'mexiletine', 'ondansetron', 'quinidine', 'ranolazine', 'sotalol', 'terfenadine', 'verapamil']
Test drugs count:  16 | Sorted list: ['astemizole', 'azimilide', 'clarithromycin', 'clozapine', 'disopyramide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', 'metoprolol', 'nifedipine', 'nitrendipine', 'pimozide', 'risperidone', 'tamoxifen', 'vandetanib']
Assert train count == 12: PASS (Actual: 12)
Assert test count == 16:  PASS (Actual: 16)
Assert train INTERSECT test == empty: PASS (Overlap: None)
Assert train UNION test == 28 unique drugs: PASS (Actual unique count: 28)
Rows per drug in TRAIN:
  Min: 2000 | Median: 2000.0 | Max: 2000 | Mode: 2000
  All train drugs match mode exactly (2,000 rows/drug).
Rows per drug in TEST:
  Min: 2000 | Median: 2000.0 | Max: 2000 | Mode: 2000
  All test drugs match mode exactly (2,000 rows/drug).

--- PLATFORM: NANION ---
Train drugs count: 12 | Sorted list: ['bepridil', 'chlorpromazine', 'cisapride', 'diltiazem', 'dofetilide', 'mexiletine', 'ondansetron', 'quinidine', 'ranolazine', 'sotalol', 'terfenadine', 'verapamil']
Test drugs count:  16 | Sorted list: ['astemizole', 'azimilide', 'clarithromycin', 'clozapine', 'disopyramide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', 'metoprolol', 'nifedipine', 'nitrendipine', 'pimozide', 'risperidone', 'tamoxifen', 'vandetanib']
Assert train count == 12: PASS (Actual: 12)
Assert test count == 16:  PASS (Actual: 16)
Assert train INTERSECT test == empty: PASS (Overlap: None)
Assert train UNION test == 28 unique drugs: PASS (Actual unique count: 28)
Rows per drug in TRAIN:
  Min: 2000 | Median: 2000.0 | Max: 2000 | Mode: 2000
  All train drugs match mode exactly (2,000 rows/drug).
Rows per drug in TEST:
  Min: 2000 | Median: 2000.0 | Max: 2000 | Mode: 2000
  All test drugs match mode exactly (2,000 rows/drug).

--- PLATFORM: MANUAL ---
Train drugs count: 12 | Sorted list: ['bepridil', 'chlorpromazine', 'cisapride', 'diltiazem', 'dofetilide', 'mexiletine', 'ondansetron', 'quinidine', 'ranolazine', 'sotalol', 'terfenadine', 'verapamil']
Test drugs count:  16 | Sorted list: ['astemizole', 'azimilide', 'clarithromycin', 'clozapine', 'disopyramide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', 'metoprolol', 'nifedipine', 'nitrendipine', 'pimozide', 'risperidone', 'tamoxifen', 'vandetanib']
Assert train count == 12: PASS (Actual: 12)
Assert test count == 16:  PASS (Actual: 16)
Assert train INTERSECT test == empty: PASS (Overlap: None)
Assert train UNION test == 28 unique drugs: PASS (Actual unique count: 28)
Rows per drug in TRAIN:
  Min: 2000 | Median: 2000.0 | Max: 2000 | Mode: 2000
  All train drugs match mode exactly (2,000 rows/drug).
Rows per drug in TEST:
  Min: 2000 | Median: 2000.0 | Max: 2000 | Mode: 2000
  All test drugs match mode exactly (2,000 rows/drug).

--- CROSS-PLATFORM DRUG SET COMPARISON ---
28-drug name sets IDENTICAL across all 3 platforms: YES

==========================================
=== STEP 5: LABEL CONSISTENCY ===
==========================================

File: chantest_training_scaled_rev.csv
  Row-level class distribution: {'high': 8000, 'intermediate': 8000, 'low': 8000}
  Drug-level class distribution: {'high': 4, 'intermediate': 4, 'low': 4}
  VERIFIED: Each drug carries exactly ONE label within this file.

File: chantest_testing_scaled_rev.csv
  Row-level class distribution: {'intermediate': 14000, 'low': 10000, 'high': 8000}
  Drug-level class distribution: {'intermediate': 7, 'low': 5, 'high': 4}
  VERIFIED: Each drug carries exactly ONE label within this file.

File: nanion_training_scaled_rev.csv
  Row-level class distribution: {'high': 8000, 'intermediate': 8000, 'low': 8000}
  Drug-level class distribution: {'high': 4, 'intermediate': 4, 'low': 4}
  VERIFIED: Each drug carries exactly ONE label within this file.

File: nanion_testing_scaled_rev.csv
  Row-level class distribution: {'intermediate': 14000, 'low': 10000, 'high': 8000}
  Drug-level class distribution: {'intermediate': 7, 'low': 5, 'high': 4}
  VERIFIED: Each drug carries exactly ONE label within this file.

File: manual_training_scaled_rev.csv
  Row-level class distribution: {'high': 8000, 'intermediate': 8000, 'low': 8000}
  Drug-level class distribution: {'high': 4, 'intermediate': 4, 'low': 4}
  VERIFIED: Each drug carries exactly ONE label within this file.

File: manual_testing_scaled_rev.csv
  Row-level class distribution: {'intermediate': 14000, 'low': 10000, 'high': 8000}
  Drug-level class distribution: {'intermediate': 7, 'low': 5, 'high': 4}
  VERIFIED: Each drug carries exactly ONE label within this file.

--- CROSS-PLATFORM DRUG LABEL DISAGREEMENTS ---
VERIFIED: Each drug carries the exact SAME label across all platforms and splits.

==========================================
=== STEP 6: SCALE SANITY ===
==========================================

--- Scale Sanity Table: chantest_training_scaled_rev.csv ---
              min       max      mean       std    median
qNet     0.569354  1.465906  0.919970  0.179730  0.892058
dvdtmax  0.900509  1.001332  0.993161  0.017307  0.999727
vmax     0.909437  1.009985  0.989239  0.019490  0.998099
vrest   -1.000534 -0.999958 -1.000073  0.000107 -1.000022
APD50    0.932773  2.056723  1.205303  0.225924  1.135504
APD90    0.978448  2.488793  1.255128  0.286869  1.167241
max_dv  -1.005477 -0.174490 -0.791837  0.169766 -0.831757
camax    0.757602  1.341898  1.026941  0.074425  1.027993
carest   0.939000  1.020000  1.002917  0.011821  1.010000
CaTD50   0.877907  1.392857  1.044227  0.047672  1.035714
CaTD90   0.966942  1.284435  1.029042  0.040541  1.013774
No constant biomarkers.

--- Scale Sanity Table: chantest_testing_scaled_rev.csv ---
              min       max      mean       std    median
qNet    -0.280247  1.351851  0.842844  0.228442  0.910348
dvdtmax  0.953006  1.003099  0.998878  0.003465  0.999991
vmax     0.945743  1.033044  0.998322  0.011621  1.000019
vrest   -1.000783 -0.999770 -1.000045  0.000101 -1.000006
APD50    0.826681  3.555672  1.215552  0.282810  1.103992
APD90    0.856034  4.558621  1.275671  0.401863  1.125862
max_dv  -1.010216  3.215577 -0.785548  0.293001 -0.868980
camax    0.537640  1.442061  1.001308  0.080153  1.017943
carest   0.836000  1.020000  0.998879  0.023229  1.000000
CaTD50   0.779900  1.956811  1.070208  0.078033  1.037375
CaTD90   0.983127  2.107782  1.047992  0.095172  1.013430
No constant biomarkers.

--- Scale Sanity Table: nanion_training_scaled_rev.csv ---
              min       max      mean       std    median
qNet     0.260453  1.245039  0.900968  0.175361  0.885958
dvdtmax  0.946835  1.001042  0.994310  0.012004  0.999249
vmax     0.943824  1.013898  0.988748  0.017936  0.995163
vrest   -1.000545 -0.999958 -1.000080  0.000128 -1.000032
APD50    0.968487  2.155462  1.214459  0.244470  1.139706
APD90    0.976724  3.236207  1.268824  0.321089  1.168103
max_dv  -1.001400  1.016641 -0.781298  0.182879 -0.823813
camax    0.634346  1.221718  1.013239  0.111142  1.040881
carest   0.887500  1.017500  0.998689  0.028019  1.007500
CaTD50   0.940199  1.263289  1.052899  0.054973  1.037375
CaTD90   0.963499  1.591598  1.038438  0.062122  1.015152
No constant biomarkers.

--- Scale Sanity Table: nanion_testing_scaled_rev.csv ---
              min       max      mean       std    median
qNet    -0.348417  1.316547  0.839003  0.252518  0.907074
dvdtmax  0.982404  1.002498  0.999314  0.001902  0.999946
vmax     0.942332  1.041300  0.996483  0.017075  1.000000
vrest   -1.000761 -0.999801 -1.000070  0.000164 -1.000008
APD50    0.847689  3.569328  1.215373  0.294866  1.110294
APD90    0.873276  4.497414  1.281121  0.430657  1.131897
max_dv  -1.004820  2.984068 -0.775112  0.313811 -0.862881
camax    0.530447  1.245286  0.978775  0.122312  1.009634
carest   0.832500  1.020000  0.991552  0.038950  1.000000
CaTD50   0.950997  1.916113  1.075224  0.075790  1.053987
CaTD90   0.996556  2.112259  1.056795  0.106194  1.022383
No constant biomarkers.

--- Scale Sanity Table: manual_training_scaled_rev.csv ---
              min       max      mean       std    median
qNet     0.373427  1.432374  0.918260  0.227751  0.886085
dvdtmax  0.755486  1.003802  0.988017  0.033392  0.999742
vmax     0.469078  1.009050  0.979006  0.031944  0.996700
vrest   -1.001034 -0.999962 -1.000181  0.000237 -1.000058
APD50    0.808824  2.161765  1.206749  0.270244  1.138655
APD90    0.844828  2.449138  1.256089  0.320206  1.169828
max_dv  -1.024670 -0.194425 -0.775996  0.168848 -0.815452
camax    0.478540  1.226679  0.930868  0.157122  0.991396
carest   0.809000  1.020000  0.975741  0.051253  1.000000
CaTD50   0.955980  1.516611  1.087496  0.073669  1.062292
CaTD90   0.981061  1.346763  1.058777  0.063941  1.029959
No constant biomarkers.

--- Scale Sanity Table: manual_testing_scaled_rev.csv ---
              min       max      mean       std    median
qNet    -0.378505  1.315131  0.837321  0.267365  0.868970
dvdtmax  0.900253  1.002494  0.996491  0.012446  0.999850
vmax     0.934734  1.044740  0.989056  0.020823  0.993526
vrest   -1.000760 -0.999771 -1.000117  0.000175 -1.000037
APD50    0.846639  3.607143  1.221186  0.311037  1.126050
APD90    0.871552  4.256897  1.287537  0.450406  1.141379
max_dv  -1.022203  2.114495 -0.769996  0.306452 -0.864532
camax    0.531625  1.311252  0.928638  0.139493  0.969480
carest   0.833000  1.030000  0.979547  0.039616  0.994000
CaTD50   0.862126  2.305648  1.116051  0.130249  1.065615
CaTD90   0.927686  1.984848  1.078085  0.120311  1.046832
No constant biomarkers.

==========================================
=== BLOCKERS ===
==========================================
BLOCKER 1: manual_training_scaled_rev.csv: Column 'APD50' has non-numeric dtype 'str' due to 25 invalid string cell(s) {'#VALUE!': 25}
BLOCKER 2: manual_training_scaled_rev.csv: Column 'APD90' has non-numeric dtype 'str' due to 25 invalid string cell(s) {'#VALUE!': 25}
BLOCKER 3: manual_training_scaled_rev.csv: Column 'max_dv' has non-numeric dtype 'str' due to 25 invalid string cell(s) {'#VALUE!': 25}
BLOCKER 4: manual_testing_scaled_rev.csv: Column 'CaTD90' has non-numeric dtype 'str' due to 1 invalid string cell(s) {'#VALUE!': 1}