> ⚠️ DOKUMEN INI TIDAK TERVERIFIKASI — BERISI KETIDAKKONSISTENAN DATA LODO DAN KESALAHAN ATURAN SET KOSONG. RUJUKAN RESMI TERVERIFIKASI: docs/STEP13B_RF_CONFORMAL_VERIFIED.md

# STEP 13 — LAPISAN CONFORMAL PREDICTION UNTUK MODEL BASELINE RF

**Metodologi Evaluasi:** Leave-One-Drug-Out (LODO) Cross-Validation pada 12 obat training `chantest`. Kalibrasi conformal untuk `RandomForestClassifier` beku (`criterion='entropy'`, `n_estimators=400`, `seed=42`). Tanpa retraining atau scaling ulang.  

---

## 1. Kalibrasi LODO RF (12 Obat Training Chantest)

| Drug Name | Label Benar | Prob RF High | Prob RF Int | Prob RF Low | Non-conformity Score $s_d$ (RF) | Non-conformity Score $s_d$ (STACK) |
|---|---|---|---|---|---|---|
| `bepridil` | **high** | 0.895961 | 0.002966 | 0.101073 | `0.104039` | `0.703345` |
| `chlorpromazine` | **intermediate** | 0.001469 | 0.757775 | 0.240756 | `0.242225` | `0.408085` |
| `cisapride` | **intermediate** | 0.023487 | 0.976482 | 0.000030 | `0.023518` | `0.152855` |
| `diltiazem` | **low** | 0.000000 | 0.904169 | 0.095831 | `0.904169` | `0.986695` |
| `dofetilide` | **high** | 0.974995 | 0.024734 | 0.000271 | `0.025005` | `0.630580` |
| `mexiletine` | **low** | 0.006357 | 0.096546 | 0.897096 | `0.102904` | `0.236885` |
| `ondansetron` | **intermediate** | 0.264494 | 0.621936 | 0.113570 | `0.378064` | `0.163680` |
| `quinidine` | **high** | 0.726190 | 0.018756 | 0.255054 | `0.273810` | `0.045365` |
| `ranolazine` | **low** | 0.043715 | 0.279179 | 0.677106 | `0.322894` | `0.255295` |
| `sotalol` | **high** | 0.466013 | 0.479129 | 0.054859 | `0.533987` | `0.762830` |
| `terfenadine` | **intermediate** | 0.058741 | 0.897190 | 0.044069 | `0.102810` | `0.094280` |
| `verapamil` | **low** | 0.018283 | 0.350695 | 0.631023 | `0.368977` | `0.453940` |

**12 Skor Non-Conformity $s_d$ RF Terurut Menaik:**

- **Rank  1:** `cisapride` (intermediate) → $s_d = 0.023518$
- **Rank  2:** `dofetilide` (high) → $s_d = 0.025005$
- **Rank  3:** `terfenadine` (intermediate) → $s_d = 0.102810$
- **Rank  4:** `mexiletine` (low) → $s_d = 0.102904$
- **Rank  5:** `bepridil` (high) → $s_d = 0.104039$
- **Rank  6:** `chlorpromazine` (intermediate) → $s_d = 0.242225$
- **Rank  7:** `quinidine` (high) → $s_d = 0.273810$
- **Rank  8:** `ranolazine` (low) → $s_d = 0.322894$
- **Rank  9:** `verapamil` (low) → $s_d = 0.368977$
- **Rank 10:** `ondansetron` (intermediate) → $s_d = 0.378064$
- **Rank 11:** `sotalol` (high) → $s_d = 0.533987$
- **Rank 12:** `diltiazem` (low) → $s_d = 0.904169$

- **Min $s_d$:** `0.023518`
- **Median $s_d$:** `0.258017`
- **Max $s_d$:** `0.904169`

---

## 2. Sapuan Rank (k = 12 s/d 8)

| Rank $k$ | Quantile $\hat{q}(k)$ | Threshold $1 - \hat{q}$ | Interval Coverage Nominal | Nominal Coverage Midpoint |
|---|---|---|---|---|
| Rank 12 | `0.904169` | `0.095831` | `[1/13, 2/13)` | `0.115385` |
| Rank 11 | `0.533987` | `0.466013` | `[2/13, 3/13)` | `0.192308` |
| Rank 10 | `0.378064` | `0.621936` | `[3/13, 4/13)` | `0.269231` |
| Rank 9 | `0.368977` | `0.631023` | `[4/13, 5/13)` | `0.346154` |
| Rank 8 | `0.322894` | `0.677106` | `[5/13, 6/13)` | `0.423077` |

---

## 3. Hasil Evaluasi Set Prediksi RF (16 Obat Test, 3 Platform)

### Rank 12 ($\hat{q} = 0.904169$, Threshold $= 0.095831$)

| Platform | Benar / 16 | Over-called | Under-triaged | Abstain | Rata-rata Ukuran Set | Sebaran Set (1 / 2 / 3) | Coverage Empiris | Coverage Nominal | Selisih Coverage |
|---|---|---|---|---|---|---|---|---|---|
| chantest | 12/16 | 4 | 0 | 4 | 1.31 | 12 / 3 / 1 | 0.8750 | 0.1154 | +0.7596 |
| nanion | 12/16 | 4 | 0 | 5 | 1.31 | 11 / 5 / 0 | 0.8750 | 0.1154 | +0.7596 |
| manual | 10/16 | 6 | 0 | 8 | 1.56 | 8 / 7 / 1 | 0.8750 | 0.1154 | +0.7596 |


### Rank 11 ($\hat{q} = 0.533987$, Threshold $= 0.466013$)

| Platform | Benar / 16 | Over-called | Under-triaged | Abstain | Rata-rata Ukuran Set | Sebaran Set (1 / 2 / 3) | Coverage Empiris | Coverage Nominal | Selisih Coverage |
|---|---|---|---|---|---|---|---|---|---|
| chantest | 14/16 | 2 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8750 | 0.1923 | +0.6827 |
| nanion | 13/16 | 3 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8125 | 0.1923 | +0.6202 |
| manual | 13/16 | 3 | 0 | 1 | 1.06 | 15 / 1 / 0 | 0.8125 | 0.1923 | +0.6202 |


### Rank 10 ($\hat{q} = 0.378064$, Threshold $= 0.621936$)

| Platform | Benar / 16 | Over-called | Under-triaged | Abstain | Rata-rata Ukuran Set | Sebaran Set (1 / 2 / 3) | Coverage Empiris | Coverage Nominal | Selisih Coverage |
|---|---|---|---|---|---|---|---|---|---|
| chantest | 14/16 | 2 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8750 | 0.2692 | +0.6058 |
| nanion | 13/16 | 3 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8125 | 0.2692 | +0.5433 |
| manual | 12/16 | 3 | 1 | 0 | 1.00 | 16 / 0 / 0 | 0.7500 | 0.2692 | +0.4808 |


### Rank 9 ($\hat{q} = 0.368977$, Threshold $= 0.631023$)

| Platform | Benar / 16 | Over-called | Under-triaged | Abstain | Rata-rata Ukuran Set | Sebaran Set (1 / 2 / 3) | Coverage Empiris | Coverage Nominal | Selisih Coverage |
|---|---|---|---|---|---|---|---|---|---|
| chantest | 14/16 | 2 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8750 | 0.3462 | +0.5288 |
| nanion | 13/16 | 3 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8125 | 0.3462 | +0.4663 |
| manual | 12/16 | 3 | 1 | 0 | 1.00 | 16 / 0 / 0 | 0.7500 | 0.3462 | +0.4038 |


### Rank 8 ($\hat{q} = 0.322894$, Threshold $= 0.677106$)

| Platform | Benar / 16 | Over-called | Under-triaged | Abstain | Rata-rata Ukuran Set | Sebaran Set (1 / 2 / 3) | Coverage Empiris | Coverage Nominal | Selisih Coverage |
|---|---|---|---|---|---|---|---|---|---|
| chantest | 14/16 | 2 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8750 | 0.4231 | +0.4519 |
| nanion | 13/16 | 3 | 0 | 0 | 1.00 | 16 / 0 / 0 | 0.8125 | 0.4231 | +0.3894 |
| manual | 12/16 | 3 | 1 | 0 | 1.00 | 16 / 0 / 0 | 0.7500 | 0.4231 | +0.3269 |


**Ringkasan Under-Triage per Rank:**

- `rank 12: under-triaged = chantest 0, nanion 0, manual 0`
- `rank 11: under-triaged = chantest 0, nanion 0, manual 0`
- `rank 10: under-triaged = chantest 0, nanion 0, manual 1`
- `rank  9: under-triaged = chantest 0, nanion 0, manual 1`
- `rank  8: under-triaged = chantest 0, nanion 0, manual 1`

- **Rank TERKETAT dengan Under-Triage Nol di Ketiga Platform:** **Rank 12** ($\hat{q} = 0.904169$)

---

## 4. Perbandingan Langsung pada Titik Operasi Teraman

| Model / Konfigurasi | Metric / Platform | Chantest | Nanion | Manual |
|---|---|---|---|---|
| **STACK + Conformal (rank 11)** | **Benar / 16** | 11/16 | 11/16 | 11/16 |
| | **Under-Triage** | 0 | 0 | 0 |
| | **Rata-rata Ukuran Set** | 1.12 | 1.12 | 1.38 |
| | **Coverage Empiris** | 0.7500 | 0.7500 | 0.8750 |
| **RF + Conformal (rank 12)** | **Benar / 16** | 12/16 | 12/16 | 10/16 |
| | **Under-Triage** | 0 | 0 | 0 |
| | **Rata-rata Ukuran Set** | 1.31 | 1.31 | 1.56 |
| | **Coverage Empiris** | 0.8750 | 0.8750 | 0.8750 |
| **RF Hard-Label** | **Benar / 16** | 14/16 | 13/16 | 12/16 |
| | **Under-Triage** | 0 | 0 | 1 |
| | **Rata-rata Ukuran Set** | 1.00 | 1.00 | 1.00 |
| | **Coverage Empiris** | 0.8750 | 0.8125 | 0.7500 |
| **STACK Hard-Label** | **Benar / 16** | 11/16 | 10/16 | 12/16 |
| | **Under-Triage** | 1 | 1 | 0 |
| | **Rata-rata Ukuran Set** | 1.00 | 1.00 | 1.00 |
| | **Coverage Empiris** | 0.6875 | 0.6250 | 0.7500 |
| **Always-High Baseline** | **Benar / 16** | 4/16 | 4/16 | 4/16 |
| | **Under-Triage** | 0 | 0 | 0 |
| | **Rata-rata Ukuran Set** | 1.00 | 1.00 | 1.00 |
| | **Coverage Empiris** | 0.2500 | 0.2500 | 0.2500 |

---

## 5. Evaluasi Khusus Senyawa `disopyramide`

Senyawa `disopyramide` (True Label: **HIGH**). Berikut adalah hasil evaluasi untuk model RF:

| Platform | Label Benar | RF Probabilities (High, Int, Low) | Threshold RF (Rank 12) | Set Prediksi Conformal | Aksi Conformal | Hasil Conformal | Hard-Label Pred RF |
|---|---|---|---|---|---|---|---|
| chantest | **high** | `(0.685284, 0.154842, 0.159874)` | `0.095831` | `{high, intermediate, low}` | **high** | Benar (Correct) | high |
| nanion | **high** | `(0.753744, 0.197791, 0.048465)` | `0.095831` | `{high, intermediate}` | **high** | Benar (Correct) | high |
| manual | **high** | `(0.725245, 0.012632, 0.262122)` | `0.095831` | `{high, low}` | **high** | Benar (Correct) | high |

**Temuan Kunci `disopyramide` untuk RF:**
- Model **RF hard-label sudah memprediksi `disopyramide` sebagai HIGH secara tepat** (100% correct / 0 under-triage) di ketiga platform.
- Dengan demikian, **RF TIDAK membutuhkan conformal prediction** untuk menghindari under-triage pada senyawa `disopyramide` (karena hard-label RF sudah bernilai HIGH).
