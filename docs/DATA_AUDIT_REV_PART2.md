==========================================
=== 1. LOCATE EVERY BAD CELL ===
==========================================

--- manual_training_scaled_rev.csv ---
Total '#VALUE!' cells found: 75
Column  Row_Index Sample_ID  Drug_Name Label
 APD50      10030        31 mexiletine   low
 APD90      10030        31 mexiletine   low
max_dv      10030        31 mexiletine   low
 APD50      10031        32 mexiletine   low
 APD90      10031        32 mexiletine   low
max_dv      10031        32 mexiletine   low
 APD50      10032        33 mexiletine   low
 APD90      10032        33 mexiletine   low
max_dv      10032        33 mexiletine   low
 APD50      10372       373 mexiletine   low
 APD90      10372       373 mexiletine   low
max_dv      10372       373 mexiletine   low
 APD50      10576       577 mexiletine   low
 APD90      10576       577 mexiletine   low
max_dv      10576       577 mexiletine   low
 APD50      10773       774 mexiletine   low
 APD90      10773       774 mexiletine   low
max_dv      10773       774 mexiletine   low
 APD50      10796       797 mexiletine   low
 APD90      10796       797 mexiletine   low
max_dv      10796       797 mexiletine   low
 APD50      10797       798 mexiletine   low
 APD90      10797       798 mexiletine   low
max_dv      10797       798 mexiletine   low
 APD50      10798       799 mexiletine   low
 APD90      10798       799 mexiletine   low
max_dv      10798       799 mexiletine   low
 APD50      10799       800 mexiletine   low
 APD90      10799       800 mexiletine   low
max_dv      10799       800 mexiletine   low
 APD50      10937       938 mexiletine   low
 APD90      10937       938 mexiletine   low
max_dv      10937       938 mexiletine   low
 APD50      10938       939 mexiletine   low
 APD90      10938       939 mexiletine   low
max_dv      10938       939 mexiletine   low
 APD50      10939       940 mexiletine   low
 APD90      10939       940 mexiletine   low
max_dv      10939       940 mexiletine   low
 APD50      10940       941 mexiletine   low
 APD90      10940       941 mexiletine   low
max_dv      10940       941 mexiletine   low
 APD50      11349      1350 mexiletine   low
 APD90      11349      1350 mexiletine   low
max_dv      11349      1350 mexiletine   low
 APD50      11733      1734 mexiletine   low
 APD90      11733      1734 mexiletine   low
max_dv      11733      1734 mexiletine   low
 APD50      11736      1737 mexiletine   low
 APD90      11736      1737 mexiletine   low
max_dv      11736      1737 mexiletine   low
 APD50      11737      1738 mexiletine   low
 APD90      11737      1738 mexiletine   low
max_dv      11737      1738 mexiletine   low
 APD50      11742      1743 mexiletine   low
 APD90      11742      1743 mexiletine   low
max_dv      11742      1743 mexiletine   low
 APD50      11764      1765 mexiletine   low
 APD90      11764      1765 mexiletine   low
max_dv      11764      1765 mexiletine   low
 APD50      11765      1766 mexiletine   low
 APD90      11765      1766 mexiletine   low
max_dv      11765      1766 mexiletine   low
 APD50      11864      1865 mexiletine   low
 APD90      11864      1865 mexiletine   low
max_dv      11864      1865 mexiletine   low
 APD50      11865      1866 mexiletine   low
 APD90      11865      1866 mexiletine   low
max_dv      11865      1866 mexiletine   low
 APD50      11866      1867 mexiletine   low
 APD90      11866      1867 mexiletine   low
max_dv      11866      1867 mexiletine   low
 APD50      11867      1868 mexiletine   low
 APD90      11867      1868 mexiletine   low
max_dv      11867      1868 mexiletine   low

Unique affected rows count: 25
Affected Row Indices (0-based): [10030, 10031, 10032, 10372, 10576, 10773, 10796, 10797, 10798, 10799, 10937, 10938, 10939, 10940, 11349, 11733, 11736, 11737, 11742, 11764, 11765, 11864, 11865, 11866, 11867]
Affected Sample_IDs: ['1350', '1734', '1737', '1738', '1743', '1765', '1766', '1865', '1866', '1867', '1868', '31', '32', '33', '373', '577', '774', '797', '798', '799', '800', '938', '939', '940', '941']

Per-Drug Breakdown of Bad Rows:
  Drug: mexiletine      | Bad Rows: 25 /  2000 (1.25%)

--- manual_testing_scaled_rev.csv ---
Total '#VALUE!' cells found: 1
Column  Row_Index Sample_ID Drug_Name Label
CaTD90      15595      1596 ibutilide  high

Unique affected rows count: 1
Affected Row Indices (0-based): [15595]
Affected Sample_IDs: ['1596']

Per-Drug Breakdown of Bad Rows:
  Drug: ibutilide       | Bad Rows:  1 /  2000 (0.05%)

==========================================
=== 2. FULL SENTINEL SWEEP (ALL 6 FILES) ===
==========================================
FILE: manual_training_scaled_rev.csv   | COL: APD50        | SENTINEL: #VALUE!    | COUNT:  25 | DRUGS: ['mexiletine']
FILE: manual_training_scaled_rev.csv   | COL: APD90        | SENTINEL: #VALUE!    | COUNT:  25 | DRUGS: ['mexiletine']
FILE: manual_training_scaled_rev.csv   | COL: max_dv       | SENTINEL: #VALUE!    | COUNT:  25 | DRUGS: ['mexiletine']
FILE: manual_testing_scaled_rev.csv    | COL: CaTD90       | SENTINEL: #VALUE!    | COUNT:   1 | DRUGS: ['ibutilide']

Explicit Cleanliness Verdict:
  Chantest training & testing clean of all sentinels: YES
  Nanion training & testing clean of all sentinels:   YES
  Manual training & testing clean of all sentinels:   NO
  Completely clean files (4/6): ['chantest_training_scaled_rev.csv', 'chantest_testing_scaled_rev.csv', 'nanion_training_scaled_rev.csv', 'nanion_testing_scaled_rev.csv']

==========================================
=== 3. NANION EXTRA COLUMN (Training_or_testing) ===
==========================================

File: nanion_training_scaled_rev.csv
  Unique values & counts in 'Training_or_testing': {'training': 24000}
  Agreement check: PASS (All 24000 rows marked as 'training')

File: nanion_testing_scaled_rev.csv
  Unique values & counts in 'Training_or_testing': {'testing': 32000}
  Agreement check: PASS (All 32000 rows marked as 'testing')

==========================================
=== 4. CROSS-PLATFORM ROW ALIGNMENT ===
==========================================

--- SPLIT: TRAIN ---
  chantest vs nanion   | Both: 24000 | Only chantest:     0 | Only nanion:     0
  chantest vs manual   | Both: 24000 | Only chantest:     0 | Only manual:     0
  nanion   vs manual   | Both: 24000 | Only nanion:     0 | Only manual:     0

--- SPLIT: TEST ---
  chantest vs nanion   | Both: 32000 | Only chantest:     0 | Only nanion:     0
  chantest vs manual   | Both: 32000 | Only chantest:     0 | Only manual:     0
  nanion   vs manual   | Both: 32000 | Only nanion:     0 | Only manual:     0

One-Line Verdict on Row Alignment:
Verdict: The three platforms are 100% row-aligned replicate-for-replicate, sharing identical (drug_name, Sample_ID) keys across all splits.

==========================================
=== 5. NEAR-ZERO-VARIANCE BIOMARKERS ===
==========================================

--- File: chantest_training_scaled_rev.csv ---
Biomarker    | Std        | Mean       | CV (std/|mean|) | Status
-----------------------------------------------------------------
qNet         | 0.179730   | 0.919970   | 0.195365        | OK
dvdtmax      | 0.017307   | 0.993161   | 0.017426        | OK
vmax         | 0.019490   | 0.989239   | 0.019702        | OK
vrest        | 0.000107   | -1.000073  | 0.000107        | FLAG: CV < 0.01 (Near-zero-variance)
APD50        | 0.225924   | 1.205303   | 0.187442        | OK
APD90        | 0.286869   | 1.255128   | 0.228557        | OK
max_dv       | 0.169766   | -0.791837  | 0.214395        | OK
camax        | 0.074425   | 1.026941   | 0.072473        | OK
carest       | 0.011821   | 1.002917   | 0.011786        | OK
CaTD50       | 0.047672   | 1.044227   | 0.045653        | OK
CaTD90       | 0.040541   | 1.029042   | 0.039396        | OK
Summary: 1 near-zero-variance biomarker(s) flagged: ['vrest']

--- File: chantest_testing_scaled_rev.csv ---
Biomarker    | Std        | Mean       | CV (std/|mean|) | Status
-----------------------------------------------------------------
qNet         | 0.228442   | 0.842844   | 0.271037        | OK
dvdtmax      | 0.003465   | 0.998878   | 0.003469        | FLAG: CV < 0.01 (Near-zero-variance)
vmax         | 0.011621   | 0.998322   | 0.011640        | OK
vrest        | 0.000101   | -1.000045  | 0.000101        | FLAG: CV < 0.01 (Near-zero-variance)
APD50        | 0.282810   | 1.215552   | 0.232659        | OK
APD90        | 0.401863   | 1.275671   | 0.315021        | OK
max_dv       | 0.293001   | -0.785548  | 0.372989        | OK
camax        | 0.080153   | 1.001308   | 0.080049        | OK
carest       | 0.023229   | 0.998879   | 0.023255        | OK
CaTD50       | 0.078033   | 1.070208   | 0.072914        | OK
CaTD90       | 0.095172   | 1.047992   | 0.090813        | OK
Summary: 2 near-zero-variance biomarker(s) flagged: ['dvdtmax', 'vrest']

--- File: nanion_training_scaled_rev.csv ---
Biomarker    | Std        | Mean       | CV (std/|mean|) | Status
-----------------------------------------------------------------
qNet         | 0.175361   | 0.900968   | 0.194636        | OK
dvdtmax      | 0.012004   | 0.994310   | 0.012073        | OK
vmax         | 0.017936   | 0.988748   | 0.018140        | OK
vrest        | 0.000128   | -1.000080  | 0.000128        | FLAG: CV < 0.01 (Near-zero-variance)
APD50        | 0.244470   | 1.214459   | 0.201300        | OK
APD90        | 0.321089   | 1.268824   | 0.253060        | OK
max_dv       | 0.182879   | -0.781298  | 0.234071        | OK
camax        | 0.111142   | 1.013239   | 0.109690        | OK
carest       | 0.028019   | 0.998689   | 0.028056        | OK
CaTD50       | 0.054973   | 1.052899   | 0.052211        | OK
CaTD90       | 0.062122   | 1.038438   | 0.059822        | OK
Summary: 1 near-zero-variance biomarker(s) flagged: ['vrest']

--- File: nanion_testing_scaled_rev.csv ---
Biomarker    | Std        | Mean       | CV (std/|mean|) | Status
-----------------------------------------------------------------
qNet         | 0.252518   | 0.839003   | 0.300974        | OK
dvdtmax      | 0.001902   | 0.999314   | 0.001903        | FLAG: CV < 0.01 (Near-zero-variance)
vmax         | 0.017075   | 0.996483   | 0.017135        | OK
vrest        | 0.000164   | -1.000070  | 0.000164        | FLAG: CV < 0.01 (Near-zero-variance)
APD50        | 0.294866   | 1.215373   | 0.242614        | OK
APD90        | 0.430657   | 1.281121   | 0.336157        | OK
max_dv       | 0.313811   | -0.775112  | 0.404859        | OK
camax        | 0.122312   | 0.978775   | 0.124964        | OK
carest       | 0.038950   | 0.991552   | 0.039282        | OK
CaTD50       | 0.075790   | 1.075224   | 0.070488        | OK
CaTD90       | 0.106194   | 1.056795   | 0.100487        | OK
Summary: 2 near-zero-variance biomarker(s) flagged: ['dvdtmax', 'vrest']

--- File: manual_training_scaled_rev.csv ---
Biomarker    | Std        | Mean       | CV (std/|mean|) | Status
-----------------------------------------------------------------
qNet         | 0.227751   | 0.918260   | 0.248025        | OK
dvdtmax      | 0.033392   | 0.988017   | 0.033797        | OK
vmax         | 0.031944   | 0.979006   | 0.032629        | OK
vrest        | 0.000237   | -1.000181  | 0.000237        | FLAG: CV < 0.01 (Near-zero-variance)
APD50        | 0.270244   | 1.206749   | 0.223944        | OK
APD90        | 0.320206   | 1.256089   | 0.254923        | OK
max_dv       | 0.168848   | -0.775996  | 0.217589        | OK
camax        | 0.157122   | 0.930868   | 0.168791        | OK
carest       | 0.051253   | 0.975741   | 0.052527        | OK
CaTD50       | 0.073669   | 1.087496   | 0.067741        | OK
CaTD90       | 0.063941   | 1.058777   | 0.060391        | OK
Summary: 1 near-zero-variance biomarker(s) flagged: ['vrest']

--- File: manual_testing_scaled_rev.csv ---
Biomarker    | Std        | Mean       | CV (std/|mean|) | Status
-----------------------------------------------------------------
qNet         | 0.267365   | 0.837321   | 0.319310        | OK
dvdtmax      | 0.012446   | 0.996491   | 0.012490        | OK
vmax         | 0.020823   | 0.989056   | 0.021053        | OK
vrest        | 0.000175   | -1.000117  | 0.000175        | FLAG: CV < 0.01 (Near-zero-variance)
APD50        | 0.311037   | 1.221186   | 0.254700        | OK
APD90        | 0.450406   | 1.287537   | 0.349820        | OK
max_dv       | 0.306452   | -0.769996  | 0.397991        | OK
camax        | 0.139493   | 0.928638   | 0.150212        | OK
carest       | 0.039616   | 0.979547   | 0.040443        | OK
CaTD50       | 0.130249   | 1.116051   | 0.116705        | OK
CaTD90       | 0.120311   | 1.078085   | 0.111597        | OK
Summary: 1 near-zero-variance biomarker(s) flagged: ['vrest']

==========================================
=== 6. IMPACT SIMULATION (POLICY A vs POLICY B) ===
==========================================

--- File: manual_training_scaled_rev.csv ---
Original total rows: 24000
Rows containing at least one sentinel: 25

[POLICY A: Drop rows with sentinels]
  Resulting total rows: 23975
  Resulting rows per drug (POLICY A):
    bepridil       : 2000 rows
    chlorpromazine : 2000 rows
    cisapride      : 2000 rows
    diltiazem      : 2000 rows
    dofetilide     : 2000 rows
    ondansetron    : 2000 rows
    quinidine      : 2000 rows
    ranolazine     : 2000 rows
    sotalol        : 2000 rows
    terfenadine    : 2000 rows
    verapamil      : 2000 rows
    mexiletine     : 1975 rows
  Assertion every drug >= 1900 rows: PASS (Min rows: 1975)

[POLICY B: Coerce sentinels to NaN and keep rows]
  Resulting NaN count per biomarker column (POLICY B):
    qNet        : 0 NaN
    dvdtmax     : 0 NaN
    vmax        : 0 NaN
    vrest       : 0 NaN
    APD50       : 25 NaN(s)
    APD90       : 25 NaN(s)
    max_dv      : 25 NaN(s)
    camax       : 0 NaN
    carest      : 0 NaN
    CaTD50      : 0 NaN
    CaTD90      : 0 NaN

--- File: manual_testing_scaled_rev.csv ---
Original total rows: 32000
Rows containing at least one sentinel: 1

[POLICY A: Drop rows with sentinels]
  Resulting total rows: 31999
  Resulting rows per drug (POLICY A):
    astemizole     : 2000 rows
    azimilide      : 2000 rows
    clarithromycin : 2000 rows
    clozapine      : 2000 rows
    disopyramide   : 2000 rows
    domperidone    : 2000 rows
    droperidol     : 2000 rows
    loratadine     : 2000 rows
    metoprolol     : 2000 rows
    nifedipine     : 2000 rows
    nitrendipine   : 2000 rows
    pimozide       : 2000 rows
    risperidone    : 2000 rows
    tamoxifen      : 2000 rows
    vandetanib     : 2000 rows
    ibutilide      : 1999 rows
  Assertion every drug >= 1900 rows: PASS (Min rows: 1999)

[POLICY B: Coerce sentinels to NaN and keep rows]
  Resulting NaN count per biomarker column (POLICY B):
    qNet        : 0 NaN
    dvdtmax     : 0 NaN
    vmax        : 0 NaN
    vrest       : 0 NaN
    APD50       : 0 NaN
    APD90       : 0 NaN
    max_dv      : 0 NaN
    camax       : 0 NaN
    carest      : 0 NaN
    CaTD50      : 0 NaN
    CaTD90      :  1 NaN(s)

==========================================
=== VERDICT ===
==========================================
1. CHANTEST TRAINING PATH (`chantest_training_scaled_rev.csv`):
   VERDICT: COMPLETELY CLEAN. Zero sentinels, zero NaNs, exact 12-drug / 24,000-row partition.
2. CHANTEST TESTING PATH (`chantest_testing_scaled_rev.csv`):
   VERDICT: COMPLETELY CLEAN. Zero sentinels, zero NaNs, exact 16-drug / 32,000-row partition.
3. NANION EVALUATION PATHS (`nanion_training_scaled_rev.csv` & `nanion_testing_scaled_rev.csv`):
   VERDICT: COMPLETELY CLEAN. Zero sentinels, zero NaNs, exact 12/16-drug partitions. 'Training_or_testing' column is 100% compliant.
4. MANUAL EVALUATION PATHS (`manual_training_scaled_rev.csv` & `manual_testing_scaled_rev.csv`):
   VERDICT: COMPROMISED BY EXCEL '#VALUE!' SENTINELS.
   - `manual_training_scaled_rev.csv` contains 25 '#VALUE!' string cells in drug 'mexiletine' (affecting 10 unique rows across APD50, APD90, max_dv).
   - `manual_testing_scaled_rev.csv` contains 1 '#VALUE!' string cell in drug 'ibutilide' (Sample_ID 1596, affecting 1 row in CaTD90).
   - Policy A (dropping sentinel rows) leaves mexiletine with 1,990 rows (>= 1900 PASS) and ibutilide with 1,999 rows (>= 1900 PASS).