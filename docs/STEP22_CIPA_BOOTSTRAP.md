# STEP 22 — CIPA BOOTSTRAP PROTOCOL (LI ET AL. & FUADAH ET AL.)

**Execution Date**: 2026-07-31  
**Protocol**: 10,000 paired iterations resampling 1 row per drug across 16 test drugs.  
**Stacking Model SHA-256 Hash**: `A93B9868FDE30C7C93B2FB86EC5C1E594C0AA664F8F700C96BB7D9E6E7F2A5D2` (Assertion PASS)

## Executive Summary

Sesuai protokol CiPA (Li et al. & Fuadah et al.), bootstrap ini mengevaluasi variabilitas simulasi dengan me-resample 1 sampel acak dari 2.000 baris simulasi untuk masing-masing dari 16 obat uji di setiap iterasi. Seluruh 16 obat selalu terwakili di setiap iterasi (paired across models), sehingga variabilitas berasal murni dari simulasi ion-channel, bukan dari perubahan komposisi panel obat.

## Point Estimate Guard Assertions

Pengujian titik estimasi level obat (mean probabilitas 2.000 baris per obat) mengonfirmasi nilai persis:

| Model | Chantest | Nanion | Manual | Guard Status |
| :--- | :--- | :--- | :--- | :--- |
| Stacking | 0.694444 | 0.621032 | 0.754762 | PASS |
| RF | 0.877778 | 0.819373 | 0.755556 | PASS |
| LR | 0.819373 | 0.754762 | 0.819373 | PASS |
| LR-qNet | 0.811966 | 0.756614 | 0.751515 | PASS |

---

## Tabel Results — Platform: `CHANTEST`

| Metric | Stacking | RF | LR | LR-qNet |
| :--- | :--- | :--- | :--- | :--- |
| Macro-F1 | 0.694 (0.568-0.755) | 0.819 (0.691-0.878) | 0.819 (0.694-0.878) | 0.747 (0.611-0.878) |
| AUC-High | 0.854 (0.771-0.938) | 0.875 (0.854-0.938) | 0.938 (0.917-0.979) | 0.917 (0.875-0.958) |
| AUC-Int | 0.746 (0.651-0.849) | 0.746 (0.683-0.873) | 0.810 (0.746-0.889) | 0.778 (0.683-0.841) |
| AUC-Low | 0.918 (0.709-1.000) | 1.000 (0.927-1.000) | 1.000 (0.945-1.000) | 1.000 (0.945-1.000) |
| Accuracy-High | 0.812 (0.750-0.875) | 0.875 (0.812-0.938) | 0.875 (0.812-0.938) | 0.812 (0.688-0.875) |
| Accuracy-Int | 0.688 (0.562-0.750) | 0.812 (0.688-0.875) | 0.812 (0.688-0.875) | 0.750 (0.625-0.875) |
| Accuracy-Low | 0.875 (0.812-0.875) | 0.938 (0.812-1.000) | 0.938 (0.875-1.000) | 0.938 (0.812-1.000) |
| LR+-High | 4.500 (3.000-9.000) | 6.000 (4.000-12.000) | 6.000 (4.500-12.000) | 4.500 (2.250-6.000) |
| LR+-Int | 2.143 (1.286-3.214) | 6.429 (2.571-6.429) [terdefinisi di 39.1% iterasi] | 6.429 (2.571-7.714) [terdefinisi di 77.1% iterasi] | 5.143 (1.929-6.429) [terdefinisi di 70.2% iterasi] |
| LR+-Low | 6.600 (4.400-6.600) [terdefinisi di 2.1% iterasi] | 11.000 (4.400-11.000) [terdefinisi di 43.6% iterasi] | 8.800 (4.400-11.000) [terdefinisi di 18.1% iterasi] | 11.000 (4.400-11.000) [terdefinisi di 59.3% iterasi] |
| LR--High | 0.300 (0.000-0.600) | 0.000 (0.000-0.300) | 0.000 (0.000-0.300) | 0.300 (0.000-0.375) |
| LR--Int | 0.429 (0.214-0.771) | 0.321 (0.161-0.571) | 0.321 (0.161-0.551) | 0.482 (0.286-0.804) |
| LR--Low | 0.400 (0.400-0.600) | 0.000 (0.000-0.400) | 0.200 (0.000-0.400) | 0.000 (0.000-0.244) |


## Tabel Results — Platform: `NANION`

| Metric | Stacking | RF | LR | LR-qNet |
| :--- | :--- | :--- | :--- | :--- |
| Macro-F1 | 0.621 (0.515-0.755) | 0.756 (0.633-0.878) | 0.755 (0.694-0.878) | 0.701 (0.570-0.819) |
| AUC-High | 0.854 (0.760-0.906) | 0.875 (0.854-0.938) | 0.938 (0.917-0.979) | 0.917 (0.875-0.958) |
| AUC-Int | 0.698 (0.603-0.794) | 0.722 (0.643-0.833) | 0.810 (0.730-0.873) | 0.746 (0.651-0.841) |
| AUC-Low | 0.818 (0.655-1.000) | 0.945 (0.836-1.000) | 0.964 (0.909-1.000) | 0.964 (0.909-1.000) |
| Accuracy-High | 0.812 (0.750-0.875) | 0.875 (0.812-0.938) | 0.875 (0.812-0.938) | 0.812 (0.688-0.875) |
| Accuracy-Int | 0.625 (0.500-0.750) | 0.750 (0.625-0.875) | 0.750 (0.688-0.875) | 0.688 (0.562-0.812) |
| Accuracy-Low | 0.812 (0.750-0.875) | 0.875 (0.812-1.000) | 0.875 (0.812-1.000) | 0.938 (0.812-1.000) |
| LR+-High | 4.500 (3.000-9.000) | 6.000 (4.000-12.000) | 6.000 (4.500-12.000) | 4.500 (2.250-6.000) |
| LR+-Int | 1.607 (1.029-3.214) | 5.143 (2.571-6.429) [terdefinisi di 93.3% iterasi] | 3.214 (2.143-6.429) [terdefinisi di 95.3% iterasi] | 3.214 (1.286-6.429) [terdefinisi di 94.2% iterasi] |
| LR+-Low | 4.400 (4.400-6.600) [terdefinisi di 9.6% iterasi] | 8.800 (3.300-11.000) [terdefinisi di 50.5% iterasi] | 6.600 (3.300-11.000) [terdefinisi di 21.9% iterasi] | 8.800 (4.400-11.000) [terdefinisi di 46.9% iterasi] |
| LR--High | 0.300 (0.000-0.600) | 0.000 (0.000-0.000) | 0.000 (0.000-0.300) | 0.300 (0.000-0.375) |
| LR--Int | 0.514 (0.257-0.964) | 0.429 (0.286-0.735) | 0.367 (0.184-0.551) | 0.551 (0.321-0.918) |
| LR--Low | 0.600 (0.400-0.660) | 0.220 (0.000-0.440) | 0.400 (0.000-0.440) | 0.200 (0.000-0.440) |


## Tabel Results — Platform: `MANUAL`

| Metric | Stacking | RF | LR | LR-qNet |
| :--- | :--- | :--- | :--- | :--- |
| Macro-F1 | 0.694 (0.557-0.816) | 0.756 (0.622-0.819) | 0.756 (0.633-0.819) | 0.700 (0.539-0.819) |
| AUC-High | 0.938 (0.823-0.979) | 0.958 (0.896-1.000) | 0.958 (0.917-0.979) | 0.958 (0.917-1.000) |
| AUC-Int | 0.794 (0.643-0.921) | 0.730 (0.643-0.833) | 0.730 (0.667-0.794) | 0.841 (0.746-0.921) |
| AUC-Low | 0.827 (0.655-1.000) | 0.927 (0.855-1.000) | 0.909 (0.855-0.945) | 0.909 (0.873-0.945) |
| Accuracy-High | 0.812 (0.688-0.938) | 0.875 (0.812-0.938) | 0.875 (0.875-0.938) | 0.812 (0.688-0.875) |
| Accuracy-Int | 0.688 (0.562-0.875) | 0.750 (0.625-0.812) | 0.750 (0.625-0.812) | 0.750 (0.625-0.875) |
| Accuracy-Low | 0.812 (0.750-0.938) | 0.875 (0.750-0.938) | 0.875 (0.750-0.938) | 0.875 (0.812-0.938) |
| LR+-High | 4.500 (2.250-12.000) | 6.000 (4.000-12.000) | 6.000 (6.000-12.000) | 4.000 (2.400-6.000) |
| LR+-Int | 2.571 (1.286-6.429) [terdefinisi di 98.2% iterasi] | 5.143 (1.929-6.429) [terdefinisi di 90.9% iterasi] | 5.143 (1.929-6.429) [terdefinisi di 100.0% iterasi] | 3.857 (1.286-6.429) [terdefinisi di 48.9% iterasi] |
| LR+-Low | 6.600 (3.300-8.800) [terdefinisi di 45.9% iterasi] | 8.800 (2.933-8.800) [terdefinisi di 78.6% iterasi] | 8.800 (3.300-8.800) [terdefinisi di 73.7% iterasi] | 8.800 (3.300-8.800) [terdefinisi di 70.9% iterasi] |
| LR--High | 0.000 (0.000-0.600) | 0.000 (0.000-0.300) | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) |
| LR--Int | 0.429 (0.184-0.857) | 0.482 (0.321-0.804) | 0.482 (0.321-0.735) | 0.571 (0.286-0.857) |
| LR--Low | 0.440 (0.200-0.660) | 0.220 (0.200-0.489) | 0.220 (0.200-0.489) | 0.220 (0.200-0.440) |


## SANITY FLAGS

Daftar sel metrik yang terdefinisi di bawah 50% iterasi atau memiliki lebar Confidence Interval (CI) > 0.40:

| Platform | Model | Metric | Cell Output | Flag Reason |
| :--- | :--- | :--- | :--- | :--- |
| chantest | Stacking | LR+-High | `4.500 (3.000-9.000)` | lebar CI > 0.40 (6.000) |
| chantest | RF | LR+-High | `6.000 (4.000-12.000)` | lebar CI > 0.40 (8.000) |
| chantest | LR | LR+-High | `6.000 (4.500-12.000)` | lebar CI > 0.40 (7.500) |
| chantest | LR-qNet | LR+-High | `4.500 (2.250-6.000)` | lebar CI > 0.40 (3.750) |
| chantest | Stacking | LR+-Int | `2.143 (1.286-3.214)` | lebar CI > 0.40 (1.929) |
| chantest | RF | LR+-Int | `6.429 (2.571-6.429) [terdefinisi di 39.1% iterasi]` | terdefinisi < 50% (39.1%); lebar CI > 0.40 (3.857) |
| chantest | LR | LR+-Int | `6.429 (2.571-7.714) [terdefinisi di 77.1% iterasi]` | lebar CI > 0.40 (5.143) |
| chantest | LR-qNet | LR+-Int | `5.143 (1.929-6.429) [terdefinisi di 70.2% iterasi]` | lebar CI > 0.40 (4.500) |
| chantest | Stacking | LR+-Low | `6.600 (4.400-6.600) [terdefinisi di 2.1% iterasi]` | terdefinisi < 50% (2.1%); lebar CI > 0.40 (2.200) |
| chantest | RF | LR+-Low | `11.000 (4.400-11.000) [terdefinisi di 43.6% iterasi]` | terdefinisi < 50% (43.6%); lebar CI > 0.40 (6.600) |
| chantest | LR | LR+-Low | `8.800 (4.400-11.000) [terdefinisi di 18.1% iterasi]` | terdefinisi < 50% (18.1%); lebar CI > 0.40 (6.600) |
| chantest | LR-qNet | LR+-Low | `11.000 (4.400-11.000) [terdefinisi di 59.3% iterasi]` | lebar CI > 0.40 (6.600) |
| chantest | Stacking | LR--High | `0.300 (0.000-0.600)` | lebar CI > 0.40 (0.600) |
| chantest | Stacking | LR--Int | `0.429 (0.214-0.771)` | lebar CI > 0.40 (0.557) |
| chantest | RF | LR--Int | `0.321 (0.161-0.571)` | lebar CI > 0.40 (0.411) |
| chantest | LR-qNet | LR--Int | `0.482 (0.286-0.804)` | lebar CI > 0.40 (0.518) |
| nanion | Stacking | LR+-High | `4.500 (3.000-9.000)` | lebar CI > 0.40 (6.000) |
| nanion | RF | LR+-High | `6.000 (4.000-12.000)` | lebar CI > 0.40 (8.000) |
| nanion | LR | LR+-High | `6.000 (4.500-12.000)` | lebar CI > 0.40 (7.500) |
| nanion | LR-qNet | LR+-High | `4.500 (2.250-6.000)` | lebar CI > 0.40 (3.750) |
| nanion | Stacking | LR+-Int | `1.607 (1.029-3.214)` | lebar CI > 0.40 (2.186) |
| nanion | RF | LR+-Int | `5.143 (2.571-6.429) [terdefinisi di 93.3% iterasi]` | lebar CI > 0.40 (3.857) |
| nanion | LR | LR+-Int | `3.214 (2.143-6.429) [terdefinisi di 95.3% iterasi]` | lebar CI > 0.40 (4.286) |
| nanion | LR-qNet | LR+-Int | `3.214 (1.286-6.429) [terdefinisi di 94.2% iterasi]` | lebar CI > 0.40 (5.143) |
| nanion | Stacking | LR+-Low | `4.400 (4.400-6.600) [terdefinisi di 9.6% iterasi]` | terdefinisi < 50% (9.6%); lebar CI > 0.40 (2.200) |
| nanion | RF | LR+-Low | `8.800 (3.300-11.000) [terdefinisi di 50.5% iterasi]` | lebar CI > 0.40 (7.700) |
| nanion | LR | LR+-Low | `6.600 (3.300-11.000) [terdefinisi di 21.9% iterasi]` | terdefinisi < 50% (21.9%); lebar CI > 0.40 (7.700) |
| nanion | LR-qNet | LR+-Low | `8.800 (4.400-11.000) [terdefinisi di 46.9% iterasi]` | terdefinisi < 50% (46.9%); lebar CI > 0.40 (6.600) |
| nanion | Stacking | LR--High | `0.300 (0.000-0.600)` | lebar CI > 0.40 (0.600) |
| nanion | Stacking | LR--Int | `0.514 (0.257-0.964)` | lebar CI > 0.40 (0.707) |
| nanion | RF | LR--Int | `0.429 (0.286-0.735)` | lebar CI > 0.40 (0.449) |
| nanion | LR-qNet | LR--Int | `0.551 (0.321-0.918)` | lebar CI > 0.40 (0.597) |
| nanion | RF | LR--Low | `0.220 (0.000-0.440)` | lebar CI > 0.40 (0.440) |
| nanion | LR | LR--Low | `0.400 (0.000-0.440)` | lebar CI > 0.40 (0.440) |
| nanion | LR-qNet | LR--Low | `0.200 (0.000-0.440)` | lebar CI > 0.40 (0.440) |
| manual | Stacking | LR+-High | `4.500 (2.250-12.000)` | lebar CI > 0.40 (9.750) |
| manual | RF | LR+-High | `6.000 (4.000-12.000)` | lebar CI > 0.40 (8.000) |
| manual | LR | LR+-High | `6.000 (6.000-12.000)` | lebar CI > 0.40 (6.000) |
| manual | LR-qNet | LR+-High | `4.000 (2.400-6.000)` | lebar CI > 0.40 (3.600) |
| manual | Stacking | LR+-Int | `2.571 (1.286-6.429) [terdefinisi di 98.2% iterasi]` | lebar CI > 0.40 (5.143) |
| manual | RF | LR+-Int | `5.143 (1.929-6.429) [terdefinisi di 90.9% iterasi]` | lebar CI > 0.40 (4.500) |
| manual | LR | LR+-Int | `5.143 (1.929-6.429) [terdefinisi di 100.0% iterasi]` | lebar CI > 0.40 (4.500) |
| manual | LR-qNet | LR+-Int | `3.857 (1.286-6.429) [terdefinisi di 48.9% iterasi]` | terdefinisi < 50% (48.9%); lebar CI > 0.40 (5.143) |
| manual | Stacking | LR+-Low | `6.600 (3.300-8.800) [terdefinisi di 45.9% iterasi]` | terdefinisi < 50% (45.9%); lebar CI > 0.40 (5.500) |
| manual | RF | LR+-Low | `8.800 (2.933-8.800) [terdefinisi di 78.6% iterasi]` | lebar CI > 0.40 (5.867) |
| manual | LR | LR+-Low | `8.800 (3.300-8.800) [terdefinisi di 73.7% iterasi]` | lebar CI > 0.40 (5.500) |
| manual | LR-qNet | LR+-Low | `8.800 (3.300-8.800) [terdefinisi di 70.9% iterasi]` | lebar CI > 0.40 (5.500) |
| manual | Stacking | LR--High | `0.000 (0.000-0.600)` | lebar CI > 0.40 (0.600) |
| manual | Stacking | LR--Int | `0.429 (0.184-0.857)` | lebar CI > 0.40 (0.673) |
| manual | RF | LR--Int | `0.482 (0.321-0.804)` | lebar CI > 0.40 (0.482) |
| manual | LR | LR--Int | `0.482 (0.321-0.735)` | lebar CI > 0.40 (0.413) |
| manual | LR-qNet | LR--Int | `0.571 (0.286-0.857)` | lebar CI > 0.40 (0.571) |
| manual | Stacking | LR--Low | `0.440 (0.200-0.660)` | lebar CI > 0.40 (0.460) |