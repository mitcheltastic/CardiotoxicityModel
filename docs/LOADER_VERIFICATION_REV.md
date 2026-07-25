==========================================
=== DATA LOADER VERIFICATION (src/data_loader_rev.py) ===
==========================================

--- LABEL ENCODING DERIVATION & WARNING ---
Derived Label Mapping:
  LABEL_MAP = {'high': 0, 'intermediate': 1, 'low': 2}
Derivation Source:
  1. plus_ultra.ipynb: Cell 30 (Line 72): `'class_map': {'high': 0, 'intermediate': 1, 'low': 2}`
  2. conformal_diag.py: Line 17: `label_map = {'high': 0, 'intermediate': 1, 'low': 2}`
  3. hf_space/app.py: Line 67 & deployable bundle metadata: `inverse_class_map = {0: 'high', 1: 'intermediate', 2: 'low'}`

--------------------------------------------------------------------------------
[WARNING] REPO LABEL MAPPING IS NOT IN SEVERITY ORDER!
The repo's class map encodes High=0, Intermediate=1, Low=2.
The clinical severity hierarchy is High (most severe) > Intermediate > Low (least severe).
Conformal prediction rules ('most severe class in prediction set') depend on severity ordering.
When implementing conformal prediction, functions MUST explicitly map integer labels to severity ranks
(e.g., severity(0)=2 for High, severity(1)=1 for Int, severity(2)=0 for Low).
--------------------------------------------------------------------------------

==========================================
Loading CHANTEST TRAINING...
==========================================
Rows before load:   24000
Rows dropped:       0
Rows after load:    24000
Rows per drug (min / median / max): 2000 / 2000.0 / 2000
Column order loaded: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Data types (all float64): True
NaN count in X:     0
Inf count in X:     0

ASSERTION CHECKLIST:
  1. Canonical 11 biomarkers present & in exact order: PASS
  2. X.shape[1] == 11:                                PASS
  3. Every X column is float64:                       PASS
  4. Zero NaN and zero Inf after cleaning:            PASS
  5. Drug count == 12 (train) or 16 (test):           PASS
  6. Labels encoded to {0,1,2} subset:               PASS
  7. Every drug retains >= 1900 rows:                 PASS
  8. Length match len(X)==len(y)==len(groups):        PASS
  9. Expected row drop pattern match:                 PASS

==========================================
Loading CHANTEST TESTING...
==========================================
Rows before load:   32000
Rows dropped:       0
Rows after load:    32000
Rows per drug (min / median / max): 2000 / 2000.0 / 2000
Column order loaded: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Data types (all float64): True
NaN count in X:     0
Inf count in X:     0

ASSERTION CHECKLIST:
  1. Canonical 11 biomarkers present & in exact order: PASS
  2. X.shape[1] == 11:                                PASS
  3. Every X column is float64:                       PASS
  4. Zero NaN and zero Inf after cleaning:            PASS
  5. Drug count == 12 (train) or 16 (test):           PASS
  6. Labels encoded to {0,1,2} subset:               PASS
  7. Every drug retains >= 1900 rows:                 PASS
  8. Length match len(X)==len(y)==len(groups):        PASS
  9. Expected row drop pattern match:                 PASS

==========================================
Loading NANION TRAINING...
==========================================
Rows before load:   24000
Rows dropped:       0
Rows after load:    24000
Rows per drug (min / median / max): 2000 / 2000.0 / 2000
Column order loaded: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Data types (all float64): True
NaN count in X:     0
Inf count in X:     0

ASSERTION CHECKLIST:
  1. Canonical 11 biomarkers present & in exact order: PASS
  2. X.shape[1] == 11:                                PASS
  3. Every X column is float64:                       PASS
  4. Zero NaN and zero Inf after cleaning:            PASS
  5. Drug count == 12 (train) or 16 (test):           PASS
  6. Labels encoded to {0,1,2} subset:               PASS
  7. Every drug retains >= 1900 rows:                 PASS
  8. Length match len(X)==len(y)==len(groups):        PASS
  9. Expected row drop pattern match:                 PASS

==========================================
Loading NANION TESTING...
==========================================
Rows before load:   32000
Rows dropped:       0
Rows after load:    32000
Rows per drug (min / median / max): 2000 / 2000.0 / 2000
Column order loaded: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Data types (all float64): True
NaN count in X:     0
Inf count in X:     0

ASSERTION CHECKLIST:
  1. Canonical 11 biomarkers present & in exact order: PASS
  2. X.shape[1] == 11:                                PASS
  3. Every X column is float64:                       PASS
  4. Zero NaN and zero Inf after cleaning:            PASS
  5. Drug count == 12 (train) or 16 (test):           PASS
  6. Labels encoded to {0,1,2} subset:               PASS
  7. Every drug retains >= 1900 rows:                 PASS
  8. Length match len(X)==len(y)==len(groups):        PASS
  9. Expected row drop pattern match:                 PASS

==========================================
Loading MANUAL TRAINING...
==========================================
Rows before load:   24000
Rows dropped:       25
Rows after load:    23975
Rows per drug (min / median / max): 1975 / 2000.0 / 2000
Column order loaded: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Data types (all float64): True
NaN count in X:     0
Inf count in X:     0

ASSERTION CHECKLIST:
  1. Canonical 11 biomarkers present & in exact order: PASS
  2. X.shape[1] == 11:                                PASS
  3. Every X column is float64:                       PASS
  4. Zero NaN and zero Inf after cleaning:            PASS
  5. Drug count == 12 (train) or 16 (test):           PASS
  6. Labels encoded to {0,1,2} subset:               PASS
  7. Every drug retains >= 1900 rows:                 PASS
  8. Length match len(X)==len(y)==len(groups):        PASS
  9. Expected row drop pattern match:                 PASS

==========================================
Loading MANUAL TESTING...
==========================================
Rows before load:   32000
Rows dropped:       1
Rows after load:    31999
Rows per drug (min / median / max): 1999 / 2000.0 / 2000
Column order loaded: ['qNet', 'dvdtmax', 'vmax', 'vrest', 'APD50', 'APD90', 'max_dv', 'camax', 'carest', 'CaTD50', 'CaTD90']
Data types (all float64): True
NaN count in X:     0
Inf count in X:     0

ASSERTION CHECKLIST:
  1. Canonical 11 biomarkers present & in exact order: PASS
  2. X.shape[1] == 11:                                PASS
  3. Every X column is float64:                       PASS
  4. Zero NaN and zero Inf after cleaning:            PASS
  5. Drug count == 12 (train) or 16 (test):           PASS
  6. Labels encoded to {0,1,2} subset:               PASS
  7. Every drug retains >= 1900 rows:                 PASS
  8. Length match len(X)==len(y)==len(groups):        PASS
  9. Expected row drop pattern match:                 PASS

==========================================
=== DROP PATTERN VERIFICATION SUMMARY ===
==========================================
Expected Drop Behavior:
  - Chantest Training: 0 rows dropped (24,000 after)
  - Chantest Testing:  0 rows dropped (32,000 after)
  - Nanion Training:   0 rows dropped (24,000 after)
  - Nanion Testing:    0 rows dropped (32,000 after)
  - Manual Training:   25 rows dropped (23,975 after)
  - Manual Testing:    1 row dropped  (31,999 after)

VERIFIED PASS: Drop patterns match expectations exactly across all 6 datasets!

Audit Trail Log File ('logs/DROPPED_ROWS_REV.csv'):
  Total logged dropped rows: 26
platform    split  row_index  Sample_ID  drug_name label  offending_columns
  manual training      10030         31 mexiletine   low APD50;APD90;max_dv
  manual training      10031         32 mexiletine   low APD50;APD90;max_dv
  manual training      10032         33 mexiletine   low APD50;APD90;max_dv
  manual training      10372        373 mexiletine   low APD50;APD90;max_dv
  manual training      10576        577 mexiletine   low APD50;APD90;max_dv
  manual training      10773        774 mexiletine   low APD50;APD90;max_dv
  manual training      10796        797 mexiletine   low APD50;APD90;max_dv
  manual training      10797        798 mexiletine   low APD50;APD90;max_dv
  manual training      10798        799 mexiletine   low APD50;APD90;max_dv
  manual training      10799        800 mexiletine   low APD50;APD90;max_dv
  manual training      10937        938 mexiletine   low APD50;APD90;max_dv
  manual training      10938        939 mexiletine   low APD50;APD90;max_dv
  manual training      10939        940 mexiletine   low APD50;APD90;max_dv
  manual training      10940        941 mexiletine   low APD50;APD90;max_dv
  manual training      11349       1350 mexiletine   low APD50;APD90;max_dv
  manual training      11733       1734 mexiletine   low APD50;APD90;max_dv
  manual training      11736       1737 mexiletine   low APD50;APD90;max_dv
  manual training      11737       1738 mexiletine   low APD50;APD90;max_dv
  manual training      11742       1743 mexiletine   low APD50;APD90;max_dv
  manual training      11764       1765 mexiletine   low APD50;APD90;max_dv
  manual training      11765       1766 mexiletine   low APD50;APD90;max_dv
  manual training      11864       1865 mexiletine   low APD50;APD90;max_dv
  manual training      11865       1866 mexiletine   low APD50;APD90;max_dv
  manual training      11866       1867 mexiletine   low APD50;APD90;max_dv
  manual training      11867       1868 mexiletine   low APD50;APD90;max_dv
  manual  testing      15595       1596  ibutilide  high             CaTD90