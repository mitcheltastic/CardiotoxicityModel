# STEP 12 — DIAGNOSTIK PREDIKSI STACK VS BASE LEARNERS (RF & XGB)

**Metodologi Audit:** Read-only inspection dari `models/chantest_stacking_model_rev.joblib` pada data testing (16 obat per platform). Tanpa retraining atau scaling ulang.  

---

## 1. Tabel Prediksi Per Obat — Chantest Testing (16 Obat)

| Drug Name | Label Benar | Pred RF | Pred XGB | Pred STACK | Prob RF (H, I, L) | Prob XGB (H, I, L) | Prob STACK (H, I, L) |
|---|---|---|---|---|---|---|---|
| `astemizole` | **intermediate** | intermediate | intermediate | intermediate | `(0.006555, 0.993441, 0.000004)` | `(0.003612, 0.996377, 0.000011)` | `(0.039572, 0.960353, 0.000075)` |
| `azimilide` | **high** | high | high | high | `(0.999997, 0.000001, 0.000001)` | `(0.999986, 0.000011, 0.000002)` | `(0.989995, 0.000000, 0.010005)` |
| `clarithromycin` | **intermediate** | intermediate | intermediate | intermediate | `(0.000216, 0.997521, 0.002263)` | `(0.000017, 0.999560, 0.000423)` | `(0.015209, 0.984211, 0.000580)` |
| `clozapine` | **intermediate** | intermediate | intermediate | intermediate | `(0.000003, 0.936959, 0.063039)` | `(0.000017, 0.981136, 0.018847)` | `(0.078595, 0.897620, 0.023785)` |
| `disopyramide` | **high** | high | high | intermediate | `(0.685284, 0.154842, 0.159874)` | `(0.862995, 0.115813, 0.021192)` | `(0.438730, 0.495815, 0.065455)` |
| `domperidone` | **intermediate** | high | high | high | `(0.999788, 0.000099, 0.000114)` | `(0.999971, 0.000026, 0.000003)` | `(0.986030, 0.003185, 0.010785)` |
| `droperidol` | **intermediate** | intermediate | intermediate | intermediate | `(0.067308, 0.932523, 0.000170)` | `(0.045931, 0.954054, 0.000016)` | `(0.057251, 0.941604, 0.001145)` |
| `ibutilide` | **high** | high | high | high | `(0.998726, 0.000379, 0.000895)` | `(0.999951, 0.000009, 0.000039)` | `(0.911960, 0.048745, 0.039295)` |
| `loratadine` | **low** | low | low | intermediate | `(0.000000, 0.077575, 0.922425)` | `(0.000450, 0.022501, 0.977050)` | `(0.148550, 0.629450, 0.222000)` |
| `metoprolol` | **low** | low | low | low | `(0.000182, 0.344671, 0.655146)` | `(0.000024, 0.192717, 0.807259)` | `(0.003472, 0.262423, 0.734105)` |
| `nifedipine` | **low** | low | low | low | `(0.001344, 0.009939, 0.988718)` | `(0.000004, 0.000020, 0.999976)` | `(0.000000, 0.000000, 1.000000)` |
| `nitrendipine` | **low** | low | low | low | `(0.000000, 0.029106, 0.970894)` | `(0.000020, 0.001017, 0.998963)` | `(0.029895, 0.089335, 0.880770)` |
| `pimozide` | **intermediate** | high | high | high | `(0.951236, 0.048745, 0.000019)` | `(0.949631, 0.050357, 0.000012)` | `(0.944341, 0.046999, 0.008660)` |
| `risperidone` | **intermediate** | intermediate | intermediate | intermediate | `(0.000000, 0.684213, 0.315787)` | `(0.000173, 0.906352, 0.093476)` | `(0.006530, 0.978645, 0.014825)` |
| `tamoxifen` | **low** | low | intermediate | intermediate | `(0.000003, 0.383595, 0.616403)` | `(0.000794, 0.624335, 0.374871)` | `(0.008780, 0.930675, 0.060545)` |
| `vandetanib` | **high** | high | high | high | `(0.993006, 0.002962, 0.004031)` | `(0.999925, 0.000032, 0.000043)` | `(0.927865, 0.052095, 0.020040)` |

---

## 2 & 3. Analisis Obat: RF BENAR tapi STACK SALAH

### Platform: CHANTEST

**Total Obat (RF BENAR, STACK SALAH):** `3` obat  

| Drug Name | Label Benar | Pred RF | Pred XGB (Benar?) | Pred STACK | Prob RF (H, I, L) | Prob STACK (H, I, L) | Kategori Meta-Learner |
|---|---|---|---|---|---|---|---|
| `disopyramide` | **high** | high | Ya | intermediate | `(0.685284, 0.154842, 0.159874)` | `(0.438730, 0.495815, 0.065455)` | Kategori A (Menimpa Kedua Base Benar) |
| `loratadine` | **low** | low | Ya | intermediate | `(0.000000, 0.077575, 0.922425)` | `(0.148550, 0.629450, 0.222000)` | Kategori A (Menimpa Kedua Base Benar) |
| `tamoxifen` | **low** | low | Tidak (intermediate) | intermediate | `(0.000003, 0.383595, 0.616403)` | `(0.008780, 0.930675, 0.060545)` | Kategori B (Mengikuti Base Salah) |

**Ringkasan Kategori Meta-Learner (CHANTEST):**
- **Kategori A (Menimpa Kedua Base yang BENAR):** `2` obat ['disopyramide', 'loratadine']
- **Kategori B (Mengikuti Base yang SALAH):** `1` obat ['tamoxifen']

### Platform: NANION

**Total Obat (RF BENAR, STACK SALAH):** `3` obat  

| Drug Name | Label Benar | Pred RF | Pred XGB (Benar?) | Pred STACK | Prob RF (H, I, L) | Prob STACK (H, I, L) | Kategori Meta-Learner |
|---|---|---|---|---|---|---|---|
| `disopyramide` | **high** | high | Ya | intermediate | `(0.753744, 0.197791, 0.048465)` | `(0.264455, 0.710815, 0.024730)` | Kategori A (Menimpa Kedua Base Benar) |
| `loratadine` | **low** | low | Ya | intermediate | `(0.000000, 0.069672, 0.930327)` | `(0.146145, 0.608570, 0.245285)` | Kategori A (Menimpa Kedua Base Benar) |
| `tamoxifen` | **low** | low | Tidak (intermediate) | intermediate | `(0.000004, 0.371094, 0.628903)` | `(0.004140, 0.936830, 0.059030)` | Kategori B (Mengikuti Base Salah) |

**Ringkasan Kategori Meta-Learner (NANION):**
- **Kategori A (Menimpa Kedua Base yang BENAR):** `2` obat ['disopyramide', 'loratadine']
- **Kategori B (Mengikuti Base yang SALAH):** `1` obat ['tamoxifen']

### Platform: MANUAL

**Total Obat (RF BENAR, STACK SALAH):** `1` obat  

| Drug Name | Label Benar | Pred RF | Pred XGB (Benar?) | Pred STACK | Prob RF (H, I, L) | Prob STACK (H, I, L) | Kategori Meta-Learner |
|---|---|---|---|---|---|---|---|
| `tamoxifen` | **low** | low | Ya | intermediate | `(0.000000, 0.317312, 0.682688)` | `(0.013279, 0.839711, 0.147010)` | Kategori A (Menimpa Kedua Base Benar) |

**Ringkasan Kategori Meta-Learner (MANUAL):**
- **Kategori A (Menimpa Kedua Base yang BENAR):** `1` obat ['tamoxifen']
- **Kategori B (Mengikuti Base yang SALAH):** `0` obat []

---

## 4. Kesepakatan Baseline RF & XGB vs STACK (Chantest 16 Obat)

- **Jumlah Obat di mana RF dan XGB SEPAKAT:** `15` dari 16 obat
  - Daftar Obat Sepakat: ['astemizole', 'azimilide', 'clarithromycin', 'clozapine', 'disopyramide', 'domperidone', 'droperidol', 'ibutilide', 'loratadine', 'metoprolol', 'nifedipine', 'nitrendipine', 'pimozide', 'risperidone', 'vandetanib']
- **Dari obat yang disepakati RF & XGB, jumlah obat di mana STACK BERBEDA:** `2` obat
  - Daftar Obat STACK Berbeda dari Kesepakatan Base: ['disopyramide', 'loratadine']

---

## 5. Evaluasi Khusus Senyawa `disopyramide` (Under-triage STACK)

Senyawa `disopyramide` memiliki label benar **HIGH** (0). Berikut adalah perbandingan probabilitas dan prediksi per platform:

| Platform | Label Benar | Model | Prediksi | RF / XGB / STACK Probabilitas (High, Int, Low) | Apakah Base Memprediksi HIGH dengan Benar? |
|---|---|---|---|---|---|
| chantest | **high** | RF | high | `(0.685284, 0.154842, 0.159874)` | Ya (HIGH) |
| chantest | **high** | XGB | high | `(0.862995, 0.115813, 0.021192)` | Ya (HIGH) |
| chantest | **high** | STACK | intermediate | `(0.438730, 0.495815, 0.065455)` | Tidak (intermediate) |
| nanion | **high** | RF | high | `(0.753744, 0.197791, 0.048465)` | Ya (HIGH) |
| nanion | **high** | XGB | high | `(0.813760, 0.185273, 0.000966)` | Ya (HIGH) |
| nanion | **high** | STACK | intermediate | `(0.264455, 0.710815, 0.024730)` | Tidak (intermediate) |
| manual | **high** | RF | high | `(0.725245, 0.012632, 0.262122)` | Ya (HIGH) |
| manual | **high** | XGB | high | `(0.994922, 0.000753, 0.004325)` | Ya (HIGH) |
| manual | **high** | STACK | high | `(0.823025, 0.149885, 0.027090)` | Ya (HIGH) |