==========================================
=== STEP 18B — ORDINAL BINARY AUC (ROW LEVEL & FINAL DOC) ===
==========================================

==========================================
=== 1. MUAT HASIL LEVEL OBAT YANG SUDAH ADA ===
==========================================
Berhasil membaca 18 baris level obat dari d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\ordinal_binary_auc_rev.csv:
level model platform   ambang      auc  n_positif  n_negatif  konkordan  diskordan  ties
 drug STACK chantest ambang_1 0.981818         11          5         54          1     0
 drug STACK chantest ambang_2 0.875000          4         12         42          6     0
 drug STACK   nanion ambang_1 0.981818         11          5         54          1     0
 drug STACK   nanion ambang_2 0.875000          4         12         42          6     0
 drug STACK   manual ambang_1 0.927273         11          5         51          4     0
 drug STACK   manual ambang_2 0.958333          4         12         46          2     0
 drug    RF chantest ambang_1 1.000000         11          5         55          0     0
 drug    RF chantest ambang_2 0.916667          4         12         44          4     0
 drug    RF   nanion ambang_1 0.963636         11          5         53          2     0
 drug    RF   nanion ambang_2 0.895833          4         12         43          5     0
 drug    RF   manual ambang_1 0.945455         11          5         52          3     0
 drug    RF   manual ambang_2 0.958333          4         12         46          2     0
 drug   XGB chantest ambang_1 1.000000         11          5         55          0     0
 drug   XGB chantest ambang_2 0.916667          4         12         44          4     0
 drug   XGB   nanion ambang_1 0.981818         11          5         54          1     0
 drug   XGB   nanion ambang_2 0.937500          4         12         45          3     0
 drug   XGB   manual ambang_1 0.872727         11          5         48          7     0
 drug   XGB   manual ambang_2 0.979167          4         12         47          1     0

==========================================
=== 2. VERIFIKASI KEBETULAN ANGKA IDENTIK ===
==========================================

--- KASUS A: STACK (chantest vs nanion) ---
          drug  true_label  chan_s1   nan_s1  chan_s2   nan_s2  chan_rank_s1  nan_rank_s1  chan_rank_s2  nan_rank_s2
    astemizole           1 0.999925 0.999920 0.039572 0.039650           1.0          1.0          10.0         10.0
     azimilide           0 0.989995 0.989990 0.989995 0.989990           5.0          5.0           1.0          2.0
clarithromycin           1 0.999420 0.876610 0.015209 0.046096           2.0         12.0          12.0          9.0
     clozapine           1 0.976215 0.999915 0.078595 0.010763           9.0          2.0           8.0         12.0
  disopyramide           0 0.934545 0.975270 0.438730 0.264455          12.0          9.0           6.0          6.0
   domperidone           1 0.989215 0.990000 0.986030 0.990000           6.0          4.0           2.0          1.0
    droperidol           1 0.998855 0.997490 0.057251 0.079385           3.0          3.0           9.0          8.0
     ibutilide           0 0.960705 0.951750 0.911960 0.835310          10.0         10.0           5.0          5.0
    loratadine           2 0.778000 0.754715 0.148550 0.146145          13.0         14.0           7.0          7.0
    metoprolol           2 0.265895 0.842165 0.003472 0.024942          14.0         13.0          15.0         11.0
    nifedipine           2 0.000000 0.000000 0.000000 0.000000          16.0         15.0          16.0         15.0
  nitrendipine           2 0.119230 0.000000 0.029895 0.000000          15.0         15.0          11.0         15.0
      pimozide           1 0.991340 0.989955 0.944341 0.945112           4.0          6.0           3.0          4.0
   risperidone           1 0.985175 0.985705 0.006530 0.005565           7.0          7.0          14.0         13.0
     tamoxifen           2 0.939455 0.940970 0.008780 0.004140          11.0         11.0          13.0         14.0
    vandetanib           0 0.979960 0.980910 0.927865 0.971765           8.0          8.0           4.0          3.0

Apakah urutan peringkat STACK s1 chantest vs nanion identik? False
Apakah urutan peringkat STACK s2 chantest vs nanion identik? False

--- KASUS B: RF vs XGB (chantest) ---
          drug  true_label    rf_s1   xgb_s1    rf_s2   xgb_s2  rf_rank_s1  xgb_rank_s1  rf_rank_s2  xgb_rank_s2
    astemizole           1 0.999996 0.999989 0.006555 0.003612         2.0          3.0         8.0          8.0
     azimilide           0 0.999999 0.999998 0.999997 0.999986         1.0          1.0         1.0          1.0
clarithromycin           1 0.997738 0.999577 0.000216 0.000017         7.0          8.0        10.0         15.0
     clozapine           1 0.936961 0.981153 0.000003 0.000017         9.0          9.0        12.0         14.0
  disopyramide           0 0.840126 0.978808 0.685284 0.862995        10.0         10.0         6.0          6.0
   domperidone           1 0.999886 0.999997 0.999788 0.999971         4.0          2.0         2.0          2.0
    droperidol           1 0.999830 0.999984 0.067308 0.045931         5.0          5.0         7.0          7.0
     ibutilide           0 0.999105 0.999961 0.998726 0.999951         6.0          6.0         3.0          3.0
    loratadine           2 0.077575 0.022950 0.000000 0.000450        14.0         14.0        14.0         10.0
    metoprolol           2 0.344854 0.192741 0.000182 0.000024        13.0         13.0        11.0         12.0
    nifedipine           2 0.011283 0.000024 0.001344 0.000004        16.0         16.0         9.0         16.0
  nitrendipine           2 0.029106 0.001037 0.000000 0.000020        15.0         15.0        14.0         13.0
      pimozide           1 0.999981 0.999988 0.951236 0.949631         3.0          4.0         5.0          5.0
   risperidone           1 0.684213 0.906524 0.000000 0.000173        11.0         11.0        14.0         11.0
     tamoxifen           2 0.383598 0.625129 0.000003 0.000794        12.0         12.0        12.0          9.0
    vandetanib           0 0.995969 0.999957 0.993006 0.999925         8.0          7.0         4.0          4.0

Apakah urutan peringkat RF vs XGB s1 chantest identik? False
Apakah urutan peringkat RF vs XGB s2 chantest identik? False

==========================================
=== 3. LANGKAH 3 — AUC LEVEL BARIS (SEKUNDER) ===
==========================================
Menggunakan sklearn.metrics.roc_auc_score langsung tanpa loop bersarang Python.
Angka utama paper adalah LEVEL OBAT (drug-level mean probability).

TABEL AUC LEVEL BARIS (Sekunder, Pakai roc_auc_score):
model platform  n_rows     auc1     auc2
STACK chantest   32000 0.910365 0.857566
STACK   nanion   32000 0.808630 0.842805
STACK   manual   31999 0.835618 0.924437
   RF chantest   32000 0.987054 0.877379
   RF   nanion   32000 0.935552 0.882408
   RF   manual   31999 0.932486 0.951367
  XGB chantest   32000 0.979364 0.904890
  XGB   nanion   32000 0.943501 0.921747
  XGB   manual   31999 0.835919 0.949524

Updated d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\results\ordinal_binary_auc_rev.csv | Total baris: 36 | Ukuran: 1923 bytes

==========================================
=== 4. CATATAN PERBANDINGAN AUC VS DRUG-LEVEL MACRO-F1 ===
==========================================
Sumber Drug-Level Macro-F1: docs/PAPER_NUMBERS.md (H1)
model platform  auc_ambang_1  auc_ambang_2 drug_macro_f1
STACK chantest      0.981818      0.875000      0.694444
STACK   nanion      0.981818      0.875000      0.621032
STACK   manual      0.927273      0.958333      0.754762
   RF chantest      1.000000      0.916667      0.877778
   RF   nanion      0.963636      0.895833      0.819373
   RF   manual      0.945455      0.958333      0.755556
  XGB chantest      1.000000      0.916667      0.819373
  XGB   nanion      0.981818      0.937500      0.754762
  XGB   manual      0.872727      0.979167      0.819373

==========================================
=== 5. PENULISAN DOKUMEN & SINKRONISASI MIRROR ===
==========================================