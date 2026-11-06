# %% [markdown]
# # 07 — Eksperimen pada 12 unit: encoding, transformasi, varian SMA, optimizer baku
#
# **Unit:** 12 unit cv5x1 yang sama dengan uji ketahanan mealpy — Cervical asli (run 0–3), Cervical 6:1 (run 0–3),
# Breast asli (run 0–3). Data, subsampling, seed, partisi CV pencarian, anggaran (300), populasi (20), evaluasi ulang
# (5 partisi baru), dan hold-out IDENTIK dengan eksperimen utama; kode inti diambil langsung dari 01_experiment_v5_2.py
# lewat `core.py`. Hasil eksperimen utama (encoding linier: RS, SMA, SMA_flip, HHO, HHO_flip, default) dan TPE
# dibaca dari `results/units` dan `results/baseline_tpe/units` sebagai pembanding berpasangan.
#
# | Blok | Pertanyaan | Arm (per unit) |
# |---|---|---|
# | A `log` | Encoding log | RS_log, SMA_log, SMA_log_flip, HHO_log, HHO_log_flip, TPE_log |
# | B `partial` | Transformasi kedua (dosis–respons) | SMA/HHO dengan hanya 3 dimensi berpengaruh dicerminkan (flipImp) atau hanya 6 dimensi lain (flipOth) |
# | C `smavar` | Varian SMA invarian | SMAb (kontraksi ke solusi terbaik) & SMAm (ke centroid), asli + dicerminkan |
# | D `std` | Optimizer baku | DE/rand/1/bin dan CMA-ES, asli + dicerminkan |
#
# Encoding log: learning_rate log10 pada 0,01–0,30; gamma, reg_alpha, reg_lambda log1p pada 0–batas atas (rentang tidak
# berubah, hanya kerapatan). Dimensi berpengaruh: min_child_weight, gamma, reg_alpha (Tabel S5 naskah).
#
# **Perkiraan waktu** (1 pencarian ±3–6 menit di Colab, ±0,5 menit di PC cepat): blok A 6, B 4, C 4, D 4 pencarian per
# unit → 18 × 12 = 216 pencarian ≈ 11–20 jam serial, ±6–10 jam dengan 2 worker. Prioritas: A dan C dulu, lalu B dan D.
# Semua tersimpan per (unit, blok) → aman bila Colab terputus; jalankan ulang sel yang sama untuk melanjutkan.
#
# **Cara pakai di Colab:**
# ```
# from google.colab import drive; drive.mount('/content/drive')
# !git clone https://github.com/Thata-ux/noise-landscape-encoding-bias-hpo-cancer.git
# %cd noise-landscape-encoding-bias-hpo-cancer/src
# !pip install -q xgboost==3.4.1 optuna==4.1.0 cma statsmodels
# ```
# Uji cepat dulu (±2–5 menit): set `TEST_MODE = True`, jalankan, lalu kembalikan ke `False`.

# %%
import os, sys, json, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')) if '__file__' not in globals() else os.path.dirname(os.path.abspath(__file__)))
import core as rc

# ---------------- PENGATURAN ----------------
BLOCKS_TO_RUN = ['log', 'smavar', 'partial', 'std']   # urutan prioritas; hapus blok yang belum ingin dijalankan
N_WORKERS = int(os.environ.get('P2_WORKERS', 4 if rc.ON_KAGGLE else 2))   # Kaggle CPU: 4 core; Colab gratis: 2
DEADLINE_HOURS = 10.5 if rc.ON_KAGGLE else None   # Kaggle mematikan sesi pada 12 jam → berhenti mulai pekerjaan baru di 10,5 jam
TEST_MODE = bool(int(os.environ.get('P2_REV_TEST', 0)))  # True = 1 unit, anggaran 40, populasi 10 (uji alur saja)
# --------------------------------------------

BLOCKS = {
    'log': [('RS_log', 'RS', 'log'), ('SMA_log', 'SMA', 'log'), ('SMA_log_flip', 'SMA', 'log_flip'),
            ('HHO_log', 'HHO', 'log'), ('HHO_log_flip', 'HHO', 'log_flip'), ('TPE_log', 'TPE', 'log')],
    'partial': [('SMA_flipImp', 'SMA', 'flipImp'), ('SMA_flipOth', 'SMA', 'flipOth'),
                ('HHO_flipImp', 'HHO', 'flipImp'), ('HHO_flipOth', 'HHO', 'flipOth')],
    'smavar': [('SMAb', 'SMAb', 'lin'), ('SMAb_flip', 'SMAb', 'flip'), ('SMAm', 'SMAm', 'lin'), ('SMAm_flip', 'SMAm', 'flip')],
    'std': [('DE', 'DE', 'lin'), ('DE_flip', 'DE', 'flip'), ('CMAES', 'CMAES', 'lin'), ('CMAES_flip', 'CMAES', 'flip')],
}
OUT = os.path.join(rc.default_output_base(), 'subset_encoding_variants' + ('_TEST' if TEST_MODE else ''))
UNITS12 = rc.SUBSET_12[:1] if TEST_MODE else rc.SUBSET_12
extra = dict(budget=40, pop=10) if TEST_MODE else {}
JOBS = [dict(block=b, arms=BLOCKS[b], ds=ds, ir=ir, run=r, **extra) for b in BLOCKS_TO_RUN for ds, ir, r in UNITS12]
print(f'Output → {OUT}\nBlok: {BLOCKS_TO_RUN} | unit: {len(UNITS12)} | pekerjaan: {len(JOBS)} | worker: {N_WORKERS}')
print('Data eksperimen utama:', rc.MAIN_UNITS_DIR, '→', 'ADA' if os.path.isdir(rc.MAIN_UNITS_DIR) else 'TIDAK DITEMUKAN')

# %%
# ============================================================
# 1. JALANKAN (checkpoint per unit × blok)
# ============================================================
if rc.ON_KAGGLE:
    rc.restore_checkpoints(OUT, os.path.basename(OUT))   # lanjutkan dari output versi sebelumnya (bila ditambahkan sebagai Input)
LEFT = rc.run_blocks_parallel(JOBS, OUT, n_workers=N_WORKERS, deadline_hours=DEADLINE_HOURS)

# %%
# ============================================================
# 2. ANALISIS BERPASANGAN (n = 12 unit; Wilcoxon, median Δ, CI bootstrap, Holm per blok)
# ============================================================
R = rc.load_block_results(OUT)
print('Cek reproduksi (XGBoost default pada partisi pencarian; harus 0):')
for f in glob.glob(os.path.join(OUT, 'units', '*.json')):
    u = json.load(open(f))
    if abs(u.get('repro_check_default_cv_wf1_diff', 0)) > 1e-9:
        print('  ⚠️ ', u['unit'], u['repro_check_default_cv_wf1_diff'])
print('  selesai (tanpa ⚠️ = identik dengan eksperimen utama)')

units = sorted(R.unit.unique())
REF = rc.load_main_reference(units)
LONG = pd.concat([R[['unit', 'arm', 'reeval_cv_wf1', 'reeval_cv_mf1', 'reeval_cv_mcc', 'ho_wf1',
                     'final_real_frac_at_lower', 'final_real_frac_at_upper']], REF], ignore_index=True)
LONG.to_csv(os.path.join(OUT, 'long.csv'), index=False)

PAIRS = {
    'A_log': [('SMA_lin', 'RS_lin', 'SMA − RS (linier; eksperimen utama)'),
              ('SMA_log', 'RS_log', 'SMA − RS (log)'),
              ('HHO_lin', 'RS_lin', 'HHO − RS (linier)'),
              ('HHO_log', 'RS_log', 'HHO − RS (log)'),
              ('TPE_log', 'RS_log', 'TPE − RS (log)'),
              ('SMA_lin', 'SMA_lin_flip', 'flip SMA (linier)'),
              ('SMA_log', 'SMA_log_flip', 'flip SMA (log)'),
              ('HHO_lin', 'HHO_lin_flip', 'flip HHO (linier)'),
              ('HHO_log', 'HHO_log_flip', 'flip HHO (log)'),
              ('RS_log', 'RS_lin', 'RS: log − linier'),
              ('SMA_log', 'SMA_lin', 'SMA: log − linier'),
              ('TPE_log', 'TPE_lin', 'TPE: log − linier')],
    'B_partial': [('SMA_lin', 'SMA_flipOth', 'SMA: asli − flip 6 dimensi lain'),
                  ('SMA_lin', 'SMA_flipImp', 'SMA: asli − flip 3 dimensi berpengaruh'),
                  ('SMA_lin', 'SMA_lin_flip', 'SMA: asli − flip semua 9 dimensi'),
                  ('HHO_lin', 'HHO_flipOth', 'HHO: asli − flip 6 dimensi lain'),
                  ('HHO_lin', 'HHO_flipImp', 'HHO: asli − flip 3 dimensi berpengaruh'),
                  ('HHO_lin', 'HHO_lin_flip', 'HHO: asli − flip semua 9 dimensi')],
    'C_smavar': [('SMAb', 'SMAb_flip', 'flip SMAb (kontraksi ke solusi terbaik)'),
                 ('SMAm', 'SMAm_flip', 'flip SMAm (kontraksi ke centroid)'),
                 ('SMAb', 'RS_lin', 'SMAb − RS'), ('SMAb_flip', 'RS_lin', 'SMAb (dicerminkan) − RS'),
                 ('SMAm', 'RS_lin', 'SMAm − RS'), ('SMAm_flip', 'RS_lin', 'SMAm (dicerminkan) − RS'),
                 ('SMAb', 'TPE_lin', 'SMAb − TPE')],
    'D_std': [('DE', 'DE_flip', 'flip DE'), ('CMAES', 'CMAES_flip', 'flip CMA-ES'),
              ('DE', 'RS_lin', 'DE − RS'), ('CMAES', 'RS_lin', 'CMA-ES − RS'),
              ('DE', 'TPE_lin', 'DE − TPE'), ('CMAES', 'TPE_lin', 'CMA-ES − TPE'),
              ('TPE_lin', 'TPE_lin_flip', 'flip TPE (kontrol)')],
}
tables = []
for metric in ('reeval_cv_wf1', 'reeval_cv_mf1'):
    wide = LONG.pivot_table(index='unit', columns='arm', values=metric)
    for blk, pairs in PAIRS.items():
        T = rc.paired_summary(wide, pairs, metric)
        if len(T):
            T.insert(0, 'block', blk); tables.append(T)
PT = pd.concat(tables, ignore_index=True)
PT.to_csv(os.path.join(OUT, 'paired_tests.csv'), index=False)
pd.set_option('display.width', 220)
print(PT[PT.metric == 'reeval_cv_wf1'][['block', 'comparison', 'n', 'median_delta', 'ci_lo', 'ci_hi', 'n_pos', 'n_neg',
                                        'r_rank_biserial', 'p', 'p_holm']].round(4).to_string(index=False))

# posisi akhir di ruang hyperparameter ASLI (linier ter-normalisasi) per arm
POS = (LONG.groupby('arm')[['final_real_frac_at_lower', 'final_real_frac_at_upper', 'reeval_cv_wf1']].mean()
       .round(4).sort_index())
POS.to_csv(os.path.join(OUT, 'positions_by_arm.csv'))
print('\nRata-rata proporsi dimensi di batas bawah/atas ruang ASLI dan wF1 evaluasi ulang per arm:')
print(POS.to_string())

# %%
# ============================================================
# 3. CARA MEMBACA
# ============================================================
# Blok A — bila 'SMA − RS (log)' jauh lebih kecil daripada 'SMA − RS (linier)', keunggulan SMA sebagian besar berasal
#   dari encoding linier (wilayah reg_alpha≈0 sempit pada skala linier). 'flip SMA (log)' yang tetap positif berarti
#   bias ke titik asal tetap ada pada encoding log (mekanisme tidak bergantung encoding, besaran bergantung encoding).
# Blok B — prediksi mekanisme: flip 3 dimensi berpengaruh menghasilkan Δ yang mendekati flip semua dimensi,
#   sedangkan flip 6 dimensi lain menghasilkan Δ kecil (dosis–respons).
# Blok C — SMAb lolos uji tanpa informasi fitness (06, bagian 5); bila 'flip SMAb' ≈ 0 dan SMAb ≥ RS, kontraksi relatif terhadap
#   solusi terbaik menghilangkan keunggulan akibat encoding tanpa merusak kinerja. SMAm masih bias ke pusat (06, bagian 5).
# Blok D — 'flip DE' dan 'flip CMA-ES' ≈ 0 menunjukkan flip test tidak 'menghukum' optimizer baku yang invarian.
print('Selesai. Keluaran:', OUT)
