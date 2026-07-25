# STEP 17 — CI RECONCILIATION & REPRODUCIBILITY AUDIT REPORT

**Metodologi Audit:** Audit riwayat pembuatan dokumen, verifikasi determinisme bootstrap paired STEP 11, pengujian konsistensi arah perbandingan model, dan rekonsiliasi formal Seksi H1 di `PAPER_NUMBERS.md`.  

---

## 1. Riwayat Pembuatan dan Modifikasi File

| Komponen File | Path File | Waktu Modifikasi Terakhir | SHA-256 (16 Karakter Pertama) |
|---|---|---|---|
| Tree1 STEP11.md | `d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI\docs\STEP11_BASE_LEARNER_REFERENCE.md` | `2026-07-25 20:00:12` | `3c71669f73a2ea3c...` |
| Tree2 STEP11.md | `d:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\docs\STEP11_BASE_LEARNER_REFERENCE.md` | `2026-07-25 20:00:12` | `3c71669f73a2ea3c...` |
| run_step11.py | `C:\Users\Mitch\.gemini\antigravity-ide\brain\de1c2213-c5cc-468e-aaba-d523367de2fa\scratch\run_step11.py` | `2026-07-25 19:55:24` | `abf75c2893b7b0e0...` |
| run_step11_sync.py | `C:\Users\Mitch\.gemini\antigravity-ide\brain\de1c2213-c5cc-468e-aaba-d523367de2fa\scratch\run_step11_sync.py` | `2026-07-25 19:56:55` | `2f599dc867832fd0...` |

**Analisis Riwayat File:** `run_step11.py` dibuat pada 19:55:24, diikuti sinkronisasi `run_step11_sync.py` pada 19:56:55, dan penulisan dokumen `STEP11_BASE_LEARNER_REFERENCE.md` pada 20:00:12. Dokumen **STEP 11 dibuat secara berurutan setelah script selesai dijalankan** dan kedua tree `docs/` berada dalam keadaan 100% identik.

---

## 2. Cetak Verbatim Hasil Bootstrap STEP11

Kutipan VERBATIM dari `docs/STEP11_BASE_LEARNER_REFERENCE.md` Seksi 4 (*Paired Bootstrap Analysis*, 10.000 Iterasi, Seed 42):

```markdown
| Platform | Perbandingan | Median Selisih | 95% Confidence Interval | Lebar CI | Memuat Nol? |
|---|---|---|---|---|---|
| chantest | STACK - RF  | -0.187302 | [-0.448155, +0.000000] | 0.448155 | Ya |
| chantest | STACK - XGB | -0.123151 | [-0.357156, +0.000000] | 0.357156 | Ya |
| nanion   | STACK - RF  | -0.197739 | [-0.454772, +0.000000] | 0.454772 | Ya |
| nanion   | STACK - XGB | -0.126623 | [-0.374185, +0.000000] | 0.374185 | Ya |
| manual   | STACK - RF  | +0.000000 | [-0.197902, +0.188889] | 0.386791 | Ya |
| manual   | STACK - XGB | -0.061601 | [-0.241282, +0.000000] | 0.241282 | Ya |
```

---

## 3. Uji Determinisme Paired Bootstrap (3 Run Single Process)

| Perbandingan | Run 1 (95% CI) | Run 2 (95% CI) | Run 3 (95% CI) | Max Diff | Status |
|---|---|---|---|---|---|
| `chantest_rf` | `[-0.448155, +0.000000]` | `[-0.448155, +0.000000]` | `[-0.448155, +0.000000]` | `0.00000000` | **PASS** |
| `chantest_xgb` | `[-0.357156, +0.000000]` | `[-0.357156, +0.000000]` | `[-0.357156, +0.000000]` | `0.00000000` | **PASS** |
| `manual_rf` | `[-0.197902, +0.188889]` | `[-0.197902, +0.188889]` | `[-0.197902, +0.188889]` | `0.00000000` | **PASS** |
| `manual_xgb` | `[-0.241282, +0.000000]` | `[-0.241282, +0.000000]` | `[-0.241282, +0.000000]` | `0.00000000` | **PASS** |
| `nanion_rf` | `[-0.454772, +0.000000]` | `[-0.454772, +0.000000]` | `[-0.454772, +0.000000]` | `0.00000000` | **PASS** |
| `nanion_xgb` | `[-0.374185, +0.000000]` | `[-0.374185, +0.000000]` | `[-0.374185, +0.000000]` | `0.00000000` | **PASS** |

**Hasil Uji Determinisme:** **PASS: Bootstrap STEP11 100% deterministik sampai 6 desimal.**

---

## 4. Uji Konsistensi Arah (STACK Benar vs RF Salah)

Pemeriksaan terhadap jumlah obat di mana STACK memberikan prediksi hard-label BENAR sedangkan RF SALAH:

| Platform | Jumlah Obat (STACK Benar & RF Salah) | Daftar Obat | Batas Atas Selisih Macro-F1 (STACK-RF) Harus $\le 0.000000$? | Batas Atas CI STEP11 | Status Guard Arah |
|---|---|---|---|---|---|
| chantest | 0 | `None (0 obat)` | YA | `+0.000000` | **PASS** |
| nanion | 0 | `None (0 obat)` | YA | `+0.000000` | **PASS** |
| manual | 1 | `clarithromycin` | TIDAK (Dapat > 0) | `+0.188889` | **PASS** |

**Kesimpulan Uji Konsistensi Arah:** **PASS: Batas atas CI STEP11 100% konsisten dengan distribusi kesalahan model.**

---

## 5. Sinkronisasi Seksi H1 PAPER_NUMBERS.MD

Untuk menghindari kerancuan antara **Point Estimate Selisih Rata-rata** dan **Bootstrap Median Selisih**, Seksi H1 pada `PAPER_NUMBERS.md` telah diselaraskan secara eksplisit memisahkan kedua metrik tersebut:

```markdown
### H1. Performa Base Learner (16 Obat Uji, Model Beku, Tanpa Refit)
* **Sumber Data:** `docs/STEP11_BASE_LEARNER_REFERENCE.md` & `results/base_learner_reference_rev.csv`
* **Populasi Evaluasi:** 16 obat test held-out per platform (`chantest`, `nanion`, `manual`), model beku dari `models/chantest_stacking_model_rev.joblib`.
* **Metrik Drug-Level Macro-F1 & Under-Triage:**
  - **Chantest Platform (16 obat):**
    - RF Base Learner: Macro-F1 = `0.877778` (Benar: 14/16, Under-triage: `0`)
    - XGB Base Learner: Macro-F1 = `0.819373` (Benar: 13/16, Under-triage: `0`)
    - STACK Model: Macro-F1 = `0.694444` (Benar: 11/16, Under-triage: `1` [senyawa `disopyramide`])
  - **Nanion Platform (16 obat):**
    - RF Base Learner: Macro-F1 = `0.819373` (Benar: 13/16, Under-triage: `0`)
    - XGB Base Learner: Macro-F1 = `0.754762` (Benar: 12/16, Under-triage: `0`)
    - STACK Model: Macro-F1 = `0.621032` (Benar: 10/16, Under-triage: `1` [senyawa `disopyramide`])
  - **Manual Platform (16 obat):**
    - RF Base Learner: Macro-F1 = `0.755556` (Benar: 12/16, Under-triage: `1` [senyawa `clarithromycin`])
    - XGB Base Learner: Macro-F1 = `0.819373` (Benar: 13/16, Under-triage: `0`)
    - STACK Model: Macro-F1 = `0.754762` (Benar: 12/16, Under-triage: `0`)
* **Selisih Berpasangan Paired Bootstrap STACK minus Base Learner (10.000 Iterasi, Seed 42, Verbatim STEP11):**
  - **Chantest Platform:**
    - STACK - RF: Point Estimate Diff = `-0.183333` | Bootstrap Median = `-0.187302` (95% CI `[-0.448155, +0.000000]`, Lebar CI: `0.448155`, Memuat 0: YA)
    - STACK - XGB: Point Estimate Diff = `-0.124929` | Bootstrap Median = `-0.123151` (95% CI `[-0.357156, +0.000000]`, Lebar CI: `0.357156`, Memuat 0: YA)
  - **Nanion Platform:**
    - STACK - RF: Point Estimate Diff = `-0.198341` | Bootstrap Median = `-0.197739` (95% CI `[-0.454772, +0.000000]`, Lebar CI: `0.454772`, Memuat 0: YA)
    - STACK - XGB: Point Estimate Diff = `-0.133730` | Bootstrap Median = `-0.126623` (95% CI `[-0.374185, +0.000000]`, Lebar CI: `0.374185`, Memuat 0: YA)
  - **Manual Platform:**
    - STACK - RF: Point Estimate Diff = `-0.000794` | Bootstrap Median = `+0.000000` (95% CI `[-0.197902, +0.188889]`, Lebar CI: `0.386791`, Memuat 0: YA)
    - STACK - XGB: Point Estimate Diff = `-0.064611` | Bootstrap Median = `-0.061601` (95% CI `[-0.241282, +0.000000]`, Lebar CI: `0.241282`, Memuat 0: YA)
```


---

## SANITY FLAGS

- `SANITY FLAG 1: Determinisme Paired Bootstrap STEP11 terverifikasi 100% (PASS)`
- `SANITY FLAG 2: Uji Konsistensi Arah Batas Atas CI STEP11 terverifikasi (PASS)`
- `SANITY FLAG 3: Point Estimate vs Bootstrap Median dilabeli terpisah di H1 (PASS)`
- `SANITY FLAG 4: SHA-256 di kedua docs/ tree 100% match dengan 0 mismatches (PASS)`