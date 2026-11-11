import os
import sys
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

# Environment setup
trainai_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\TrainAI'
outer_docs_dir = r'd:\1. COLLEGE\1. HERE WE GO\8. TUGAS AKHIR\1. RISET ML\5. Here We Go Bismillah\docs'

if trainai_dir not in sys.path:
    sys.path.insert(0, trainai_dir)

from src.data_loader_rev import load_rev

output_lines = []
def log(msg=""):
    output_lines.append(str(msg))
    print(msg)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()

def get_auc_and_pairs(y_true_binary, scores):
    pos_mask = (y_true_binary == 1)
    neg_mask = (y_true_binary == 0)
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    if n_pos == 0 or n_neg == 0:
        return np.nan, 0, 0, 0, 0
    
    conc = 0
    disc = 0
    tied = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                conc += 1
            elif p < n:
                disc += 1
            else:
                tied += 1
                
    auc = (conc + 0.5 * tied) / (n_pos * n_neg)
    return auc, conc, disc, tied, n_pos, n_neg

# ----------------------------------------------------
# LANGKAH 0 - DEFINISI
# ----------------------------------------------------
log("==========================================")
log("=== STEP 18A — ORDINAL BINARY AUC (POINT ESTIMATES ONLY) ===")
log("==========================================")
log("LANGKAH 0 - DEFINISI:")
log("Peta label repo: high=0, intermediate=1, low=2.")
log("Dua ambang ordinal:")
log("  Ambang 1, P(risk > low)          = p_high + p_intermediate")
log("  Ambang 2, P(risk > intermediate) = p_high")
log("Label biner sebenarnya:")
log("  Ambang 1: positif kalau kelas benar high ATAU intermediate; negatif kalau low")
log("  Ambang 2: positif kalau kelas benar high; negatif kalau intermediate atau low")
log("==========================================\n")

# Load model for RF and XGB base learners
model_path = os.path.join(trainai_dir, 'models', 'chantest_stacking_model_rev.joblib')
stack_model = joblib.load(model_path)
rf_model = stack_model.named_estimators_['rf']
xgb_model = stack_model.named_estimators_['xgb']

models = {
    'STACK': stack_model,
    'RF': rf_model,
    'XGB': xgb_model
}

platforms = ['chantest', 'nanion', 'manual']

# ----------------------------------------------------
# LANGKAH 1 - GUARD
# ----------------------------------------------------
log("==========================================")
log("=== LANGKAH 1 — GUARD CHECK & CLASS BREAKDOWN ===")
log("==========================================")

test_data = {}
sanity_flags = []

for p in platforms:
    fpath = os.path.join(trainai_dir, 'results', f'cross_platform_probs_{p}_rev.csv')
    df = pd.read_csv(fpath)
    df_test = df[df['split'] == 'testing'].reset_index(drop=True)
    
    n_drugs = len(df_test)
    prob_sum = df_test['prob_high'] + df_test['prob_int'] + df_test['prob_low']
    sum_check = np.allclose(prob_sum, 1.0, atol=1e-6)
    
    guard_status = "PASS" if (n_drugs == 16 and sum_check) else "FAIL"
    log(f"Platform {p.upper():<8}: n_drugs={n_drugs}, sum(p)=1.0 (tol 1e-6) -> {guard_status}")
    if guard_status == "FAIL":
        log(f"ERROR: Guard check failed for platform {p}")
        sys.exit(1)
        
    counts = df_test['true_label'].value_counts().to_dict()
    n_high = counts.get(0, 0)
    n_int = counts.get(1, 0)
    n_low = counts.get(2, 0)
    
    t1_pos = n_high + n_int
    t1_neg = n_low
    t2_pos = n_high
    t2_neg = n_int + n_low
    
    log(f"  Obat per kelas (16 obat uji): High(0)={n_high}, Intermediate(1)={n_int}, Low(2)={n_low}")
    log(f"  Ambang 1: Positif={t1_pos}, Negatif={t1_neg}")
    log(f"  Ambang 2: Positif={t2_pos}, Negatif={t2_neg}")
    
    if t1_pos < 5:
        msg = f"PERINGATAN: positif sangat sedikit, AUC tidak stabil (Platform {p}, Ambang 1 Pos: {t1_pos})"
        log(f"  {msg}")
        sanity_flags.append(msg)
    if t2_pos < 5:
        msg = f"PERINGATAN: positif sangat sedikit, AUC tidak stabil (Platform {p}, Ambang 2 Pos: {t2_pos})"
        log(f"  {msg}")
        sanity_flags.append(msg)
        
    test_data[p] = df_test

# Load row-level datasets for evaluation
row_data = {}
for p in platforms:
    X, y, groups, _ = load_rev(p, 'testing', data_dir=os.path.join(trainai_dir, 'data'))
    row_data[p] = (X, y, groups)

# ----------------------------------------------------
# LANGKAH 2 - AUC LEVEL OBAT
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 2 — AUC LEVEL OBAT (UTAMA) ===")
log("==========================================")

csv_records = []
drug_results = {}

for m_name in ['STACK', 'RF', 'XGB']:
    drug_results[m_name] = {}
    for p in platforms:
        if m_name == 'STACK':
            df_test = test_data[p]
            p_high = df_test['prob_high'].values
            p_int = df_test['prob_int'].values
            p_low = df_test['prob_low'].values
            y_true = df_test['true_label'].values
        else:
            X, y, groups = row_data[p]
            m_obj = models[m_name]
            proba = m_obj.predict_proba(X)
            df_eval = pd.DataFrame(proba, columns=['prob_high', 'prob_int', 'prob_low'])
            df_eval['drug_name'] = groups.values
            df_eval['true_label'] = y.values
            
            drug_summary = df_eval.groupby('drug_name').agg({
                'true_label': 'first',
                'prob_high': 'mean',
                'prob_int': 'mean',
                'prob_low': 'mean'
            }).loc[test_data[p]['drug_name']].reset_index()
            
            p_high = drug_summary['prob_high'].values
            p_int = drug_summary['prob_int'].values
            p_low = drug_summary['prob_low'].values
            y_true = drug_summary['true_label'].values

        # Ambang 1
        s1 = p_high + p_int
        y1 = np.where((y_true == 0) | (y_true == 1), 1, 0)
        auc1, c1, d1, t1, pos1, neg1 = get_auc_and_pairs(y1, s1)
        
        # Ambang 2
        s2 = p_high
        y2 = np.where(y_true == 0, 1, 0)
        auc2, c2, d2, t2, pos2, neg2 = get_auc_and_pairs(y2, s2)
        
        drug_results[m_name][p] = {
            'auc1': auc1, 'c1': c1, 'd1': d1, 't1': t1, 'pos1': pos1, 'neg1': neg1,
            'auc2': auc2, 'c2': c2, 'd2': d2, 't2': t2, 'pos2': pos2, 'neg2': neg2
        }
        
        csv_records.append({
            'level': 'drug',
            'model': m_name,
            'platform': p,
            'ambang': 'ambang_1',
            'auc': f"{auc1:.6f}",
            'n_positif': pos1,
            'n_negatif': neg1,
            'konkordan': c1,
            'diskordan': d1,
            'ties': t1
        })
        
        csv_records.append({
            'level': 'drug',
            'model': m_name,
            'platform': p,
            'ambang': 'ambang_2',
            'auc': f"{auc2:.6f}",
            'n_positif': pos2,
            'n_negatif': neg2,
            'konkordan': c2,
            'diskordan': d2,
            'ties': t2
        })
        
        if auc1 < 0.5:
            sanity_flags.append(f"FLAG: {m_name} {p} Ambang 1 AUC ({auc1:.6f}) < 0.5")
        if auc2 < 0.5:
            sanity_flags.append(f"FLAG: {m_name} {p} Ambang 2 AUC ({auc2:.6f}) < 0.5")

log("TABEL AUC LEVEL OBAT (baris = model, kolom = platform, isi = Ambang 1 / Ambang 2):")
log(f"{'Model':<8} | {'chantest':<23} | {'nanion':<23} | {'manual':<23}")
log("-" * 84)
for m_name in ['STACK', 'RF', 'XGB']:
    r_c = drug_results[m_name]['chantest']
    r_n = drug_results[m_name]['nanion']
    r_m = drug_results[m_name]['manual']
    log(f"{m_name:<8} | {r_c['auc1']:.6f} / {r_c['auc2']:.6f} | {r_n['auc1']:.6f} / {r_n['auc2']:.6f} | {r_m['auc1']:.6f} / {r_m['auc2']:.6f}")

log("\nTABEL RINCIAN PASANGAN LEVEL OBAT (Konkordan / Diskordan / Ties / Total Pairs):")
for m_name in ['STACK', 'RF', 'XGB']:
    log(f"\nModel: {m_name}")
    log(f"  {'Platform':<10} | Ambang 1 (C/D/T/Total) | Ambang 2 (C/D/T/Total)")
    log("  " + "-" * 60)
    for p in platforms:
        r = drug_results[m_name][p]
        log(f"  {p:<10} | {r['c1']}/{r['d1']}/{r['t1']}/{r['pos1']*r['neg1']} | {r['c2']}/{r['d2']}/{r['t2']}/{r['pos2']*r['neg2']}")

# SEGERA SIMPAN RESULTS CSV LANGKAH 2
csv_path_main = os.path.join(trainai_dir, 'results', 'ordinal_binary_auc_rev.csv')
os.makedirs(os.path.dirname(csv_path_main), exist_ok=True)
df_csv = pd.DataFrame(csv_records)
df_csv.to_csv(csv_path_main, index=False)

csv_size_step2 = os.path.getsize(csv_path_main)
log(f"\n[SEGERA DITULIS] TrainAI/results/ordinal_binary_auc_rev.csv terbuat setelah Langkah 2.")
log(f"  Path Absolut : {csv_path_main}")
log(f"  Ukuran File  : {csv_size_step2} bytes ({len(df_csv)} baris)")

# ----------------------------------------------------
# LANGKAH 3 - AUC LEVEL BARIS (SEKUNDER)
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 3 — AUC LEVEL BARIS (SEKUNDER) ===")
log("==========================================")
log("Catatan Penting: Evaluasi level baris memperhitungkan seluruh variasi sampel simulasi individual.")
log("Angka utama paper adalah LEVEL OBAT karena kategorisasi risiko klinis TdP dan keputusan regulatory")
log("diaplikasikan per senyawa obat (drug-level mean probability).\n")

row_results = {}
for m_name in ['STACK', 'RF', 'XGB']:
    row_results[m_name] = {}
    for p in platforms:
        X, y, groups = row_data[p]
        m_obj = models[m_name]
        proba = m_obj.predict_proba(X)
        p_high = proba[:, 0]
        p_int = proba[:, 1]
        p_low = proba[:, 2]
        
        # Ambang 1
        s1 = p_high + p_int
        y1 = np.where((y.values == 0) | (y.values == 1), 1, 0)
        auc1, c1, d1, t1, pos1, neg1 = get_auc_and_pairs(y1, s1)
        
        # Ambang 2
        s2 = p_high
        y2 = np.where(y.values == 0, 1, 0)
        auc2, c2, d2, t2, pos2, neg2 = get_auc_and_pairs(y2, s2)
        
        row_results[m_name][p] = {
            'auc1': auc1, 'c1': c1, 'd1': d1, 't1': t1, 'pos1': pos1, 'neg1': neg1,
            'auc2': auc2, 'c2': c2, 'd2': d2, 't2': t2, 'pos2': pos2, 'neg2': neg2,
            'n_rows': len(y)
        }
        
        csv_records.append({
            'level': 'row',
            'model': m_name,
            'platform': p,
            'ambang': 'ambang_1',
            'auc': f"{auc1:.6f}",
            'n_positif': pos1,
            'n_negatif': neg1,
            'konkordan': c1,
            'diskordan': d1,
            'ties': t1
        })
        
        csv_records.append({
            'level': 'row',
            'model': m_name,
            'platform': p,
            'ambang': 'ambang_2',
            'auc': f"{auc2:.6f}",
            'n_positif': pos2,
            'n_negatif': neg2,
            'konkordan': c2,
            'diskordan': d2,
            'ties': t2
        })

log("TABEL AUC LEVEL BARIS (baris = model, kolom = platform, isi = Ambang 1 / Ambang 2):")
log(f"{'Model':<8} | {'chantest':<23} | {'nanion':<23} | {'manual':<23}")
log("-" * 84)
for m_name in ['STACK', 'RF', 'XGB']:
    r_c = row_results[m_name]['chantest']
    r_n = row_results[m_name]['nanion']
    r_m = row_results[m_name]['manual']
    log(f"{m_name:<8} | {r_c['auc1']:.6f} / {r_c['auc2']:.6f} | {r_n['auc1']:.6f} / {r_n['auc2']:.6f} | {r_m['auc1']:.6f} / {r_m['auc2']:.6f}")

# UPDATE CSV HASIL SETELAH LANGKAH 3
df_csv_final = pd.DataFrame(csv_records)
df_csv_final.to_csv(csv_path_main, index=False)

csv_size_final = os.path.getsize(csv_path_main)
log(f"\n[UPDATED CSV] TrainAI/results/ordinal_binary_auc_rev.csv diperbarui setelah Langkah 3.")
log(f"  Path Absolut : {csv_path_main}")
log(f"  Ukuran Final : {csv_size_final} bytes ({len(df_csv_final)} baris total)")

# ----------------------------------------------------
# LANGKAH 4 - NILAI RUJUKAN LI ET AL 2019
# ----------------------------------------------------
log("\n==========================================")
log("=== LANGKAH 4 — NILAI RUJUKAN LI ET AL 2019 (SITASI) ===")
log("==========================================")
log("Catatan Rujukan qNet dari Tabel PPT Pak Ali:")
log("  - Ambang 1 (P(risk > low))          : 8.4E-01 (0.84)")
log("  - Ambang 2 (P(risk > intermediate)) : 9.2E-01 (0.92)")
log("TANDAI JELAS: Ini adalah NILAI SITASI untuk pembanding rujukan, BUKAN hasil hitung dari repo ini.")
log("PERINGATAN WAJIB: Nilai rujukan ini harus diverifikasi langsung ke paper Li et al 2019 (doi: 10.1002/CPT.1184)")
log("sebelum dicantumkan ke naskah final paper. JANGAN mencantumkan metrik lain dari tabel PPT tersebut")
log("dan JANGAN menghitung metrik apa pun untuk baris qNet ini.\n")

# ----------------------------------------------------
# DOKUMEN DAN COPY MIRRORING
# ----------------------------------------------------
log("==========================================")
log("=== SANITY FLAGS ===")
log("==========================================")
if sanity_flags:
    for f in sanity_flags:
        log(f)
else:
    log("Semua sanity checks PASS. Tidak ada AUC di bawah 0.5 dan tidak ada ambang dengan positif < 5 selain Ambang 2.")

# Write main markdown document
doc_path_main = os.path.join(trainai_dir, 'docs', 'STEP18A_ORDINAL_BINARY_AUC.md')
os.makedirs(os.path.dirname(doc_path_main), exist_ok=True)

with open(doc_path_main, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

log(f"\nMain Document Written: {doc_path_main}")
log(f"Main Document Size   : {os.path.getsize(doc_path_main)} bytes")

# Copy mirror to outer docs directory
doc_path_mirror = os.path.join(outer_docs_dir, 'STEP18A_ORDINAL_BINARY_AUC.md')
os.makedirs(os.path.dirname(doc_path_mirror), exist_ok=True)

with open(doc_path_main, 'r', encoding='utf-8') as f_in:
    content = f_in.read()

with open(doc_path_mirror, 'w', encoding='utf-8') as f_out:
    f_out.write(content)

log(f"Mirror Document Copy : {doc_path_mirror}")
log(f"Mirror Document Size : {os.path.getsize(doc_path_mirror)} bytes")

# Hash verification
hash_main = compute_sha256(doc_path_main)
hash_mirror = compute_sha256(doc_path_mirror)

log(f"\nSHA-256 Checksum Main  : {hash_main}")
log(f"SHA-256 Checksum Mirror: {hash_mirror}")

if hash_main == hash_mirror:
    log("DOCUMENT HASH MATCH: PASS (Mirror file 100% identik dengan file utama)")
else:
    log("ERROR: Document hash mismatch!")
    sys.exit(1)

log("\nSTEP 18A COMPLETED SUCCESSFULLY. STOP.")
