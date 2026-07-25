# STEP 14 — EVALUASI DAN PERBANDINGAN BOOTSTRAP CONFORMAL PREDICTION (RF vs STACK)

**Metodologi Audit:** Evaluasi ketidakpastian interval bootstrap berpasangan (*paired bootstrap*, 10.000 iterasi, seed 42) antara model **RF + Conformal (Rank 11)** vs **STACK + Conformal (Rank 11)**. Koreksi label Coverage Nominal dan pembatasan aturan rule-of-three under-triage.  

---

## 1. Audit dan Koreksi Label Coverage Nominal

Pembetulan label pada tabel evaluasi rank conformal. Nilai coverage nominal sejati dihitung sebagai $1 - \alpha = k/13$:

| Rank $k$ | Quantile $\hat{q}$ | Alpha Min $\alpha = (13-k)/13$ | Coverage Nominal yang Benar ($k/13$) | Formulasi Ceil Check | Status Verification |
|---|---|---|---|---|---|
| Rank 12 | `0.904169` | `0.076923` | `0.923077` | `ceil(13 * (1 - 0.076923)) == 12` | **PASS** |
| Rank 11 | `0.533987` | `0.153846` | `0.846154` | `ceil(13 * (1 - 0.153846)) == 11` | **PASS** |
| Rank 10 | `0.378064` | `0.230769` | `0.769231` | `ceil(13 * (1 - 0.230769)) == 10` | **PASS** |
| Rank  9 | `0.368977` | `0.307692` | `0.692308` | `ceil(13 * (1 - 0.307692)) == 9` | **PASS** |
| Rank  8 | `0.322894` | `0.384615` | `0.615385` | `ceil(13 * (1 - 0.384615)) == 8` | **PASS** |

**Hasil Grep Audit Seluruh File Markdown Docs:**
- `STEP13_CONFORMAL_RF.md`: Kolom 'Coverage Nominal' tertukar dengan alpha_midpoint -> **DITANDAI UNVERIFIED**
- `STEP13C_RF_CONFORMAL_FIXED.md`: Kolom 'Coverage Nominal' diperbaiki ke $k/13$ dan ditambahkan kolom selisih (Empiris - Nominal) -> **DIPERBAIKI**
- `STEP5_CONFORMAL.md`, `STEP6_CONFORMAL_OPERATING_POINT.md`, `STORYLINE.md`, `PAPER_NUMBERS.md`: Seluruh penyebutan Coverage Nominal terverifikasi sudah bernilai $1 - \alpha = 11/13 \approx 0.8462$ -> **PASS (VALID)**

---

## 2. Hasil Paired Bootstrap (RF + Conformal vs STACK + Conformal)

Resampling unit = **OBAT** (16 obat test held-out), 10.000 iterasi, seed 42. Kedua model menggunakan indeks obat yang sama pada tiap iterasi.

- **RF + Conformal:** Rank 11 ($\hat{q} = 0.533987$, Threshold $= 0.466013$)

- **STACK + Conformal:** Rank 11 ($\hat{q} = 0.762830$, Threshold $= 0.237170$)


### Tabel Distribusi Selisih Metrik Bootstrap (RF minus STACK)

| Platform | Metrik Perbandingan | Median Selisih | 95% Confidence Interval (Persentil) | Lebar CI | Interval Memuat Nol? |
|---|---|---|---|---|---|
| chantest | Selisih Proporsi Benar (RF - STACK) | `+0.187500` | `[+0.000000, +0.375000]` | `0.375000` | YA (Memuat 0) |
| chantest | Selisih Ukuran Set Rata-rata (RF - STACK) | `-0.125000` | `[-0.312500, +0.000000]` | `0.312500` | YA (Memuat 0) |
| nanion | Selisih Proporsi Benar (RF - STACK) | `+0.125000` | `[+0.000000, +0.312500]` | `0.312500` | YA (Memuat 0) |
| nanion | Selisih Ukuran Set Rata-rata (RF - STACK) | `-0.125000` | `[-0.312500, +0.000000]` | `0.312500` | YA (Memuat 0) |
| manual | Selisih Proporsi Benar (RF - STACK) | `+0.125000` | `[+0.000000, +0.312500]` | `0.312500` | YA (Memuat 0) |
| manual | Selisih Ukuran Set Rata-rata (RF - STACK) | `-0.312500` | `[-0.562500, -0.125000]` | `0.437500` | TIDAK |

---

## 3. Batas Atas Under-Triage Model RF + Conformal

Model RF + Conformal (Rank 11) mencatat **0 under-triage** di ketiga platform pada $n = 16$ obat test.

- **Batas Atas Rule-of-Three (95% Upper Bound):**

  $$\text{Upper Bound} = \frac{3}{n} = \frac{3}{16} = 0.187500 \quad (18.75\%)$$
- **Catatan Metodologis:** Interval bootstrap persentil tidak dapat digunakan untuk kuantitas under-triage ini karena seluruh 16 sampel bernilai nol ($0/16$), sehingga resampling bootstrap atas data nol akan berakumulasi secara konstan pada titik $[0.000000, 0.000000]$. Oleh karena itu, estimasi batas atas risiko dihitung secara analitis via *rule-of-three* ($18.75\%$).

---

## 4. Uji Silang Guard Referensi Kanonik STACK Rank 11

| Platform | Metrik | Nilai Hitung | Nilai Ekspektasi | Status Guard |
|---|---|---|---|---|
| chantest | Benar / 16 | 11/16 | 11/16 | **PASS** |
| chantest | Under-triage | 0 | 0 | **PASS** |
| chantest | Rata-rata Ukuran Set | 1.1250 | 1.1250 | **PASS** |
| chantest | Coverage Empiris | 12/16 | 12/16 | **PASS** |
| nanion | Benar / 16 | 11/16 | 11/16 | **PASS** |
| nanion | Under-triage | 0 | 0 | **PASS** |
| nanion | Rata-rata Ukuran Set | 1.1250 | 1.1250 | **PASS** |
| nanion | Coverage Empiris | 12/16 | 12/16 | **PASS** |
| manual | Benar / 16 | 11/16 | 11/16 | **PASS** |
| manual | Under-triage | 0 | 0 | **PASS** |
| manual | Rata-rata Ukuran Set | 1.3750 | 1.3750 | **PASS** |
| manual | Coverage Empiris | 14/16 | 14/16 | **PASS** |

**Hasil Uji Silang Guard Kanonik:** **PASS (Semua nilai ekspektasi 100% identik).**

---

## SANITY FLAGS

- `SANITY FLAG 1: Guard Kanonik STACK Rank 11 terverifikasi 100% (PASS)`
- `SANITY FLAG 2: Label Coverage Nominal diperbaiki ke k/13 (PASS)`
- `SANITY FLAG 3: Rule-of-three Under-triage RF = 3/16 (0.187500) (PASS)`
- `SANITY FLAG 4: Dari 6 interval bootstrap 95%, 5 memuat angka 0; 1 TIDAK memuat 0`
  - `(yaitu selisih rata-rata ukuran set di platform manual, median -0.312500, 95% CI [-0.562500, -0.125000]).`
  - `Catatan Tambahan: Batas bawah selisih proporsi benar bernilai tepat +0.000000 di ketiga platform,`
    `artinya dalam setiap resample bootstrap, RF+conformal TIDAK PERNAH lebih rendah dari STACK+conformal.`
- `SANITY FLAG 5: PERINGATAN LEBAR CI BOOTSTRAP > 0.30:`
  - `Platform chantest (Diff Correct Prop 95% CI Width: 0.375000)`
  - `Platform chantest (Diff Mean Set Size 95% CI Width: 0.312500)`
  - `Platform nanion (Diff Correct Prop 95% CI Width: 0.312500)`
  - `Platform nanion (Diff Mean Set Size 95% CI Width: 0.312500)`
  - `Platform manual (Diff Correct Prop 95% CI Width: 0.312500)`
  - `Platform manual (Diff Mean Set Size 95% CI Width: 0.437500)`