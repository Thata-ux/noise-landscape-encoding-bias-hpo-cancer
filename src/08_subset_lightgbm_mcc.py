# %% [markdown]
# # 08 — Eksperimen pada 12 unit: model kedua (LightGBM) dan MCC sebagai fitness pencarian
#
# **Unit:** 12 unit cv5x1 yang sama dengan 07_subset_encoding_variants (Cervical asli, Cervical 6:1, Breast asli; run 0–3).
#
# | Blok | Pertanyaan | Arm (per unit) |
# |---|---|---|
# | E `lgbm` | Generalisasi ke model lain | LightGBM default, RS, SMA, SMA_flip, HHO, HHO_flip, TPE, TPE_flip (fitness wF1) |
# | F `mcc` | Dampak bias pada metrik kelas minoritas | XGBoost dengan fitness −MCC: RS, SMA, SMA_flip, HHO, HHO_flip, TPE |
#
# Ruang LightGBM (9 dimensi, analog Tabel 3): learning_rate 0,01–0,30; max_depth 3–10; min_child_samples 5–50;
# min_split_gain 0–5; subsample 0,5–1 (subsample_freq = 1); colsample_bytree 0,5–1; reg_alpha 0–10; reg_lambda 0–10;
# n_estimators 50–500; scale_pos_weight seperti XGBoost. Preprocessing, partisi, seed, anggaran, dan evaluasi ulang
# identik dengan eksperimen utama.
#
# **Perkiraan waktu:** blok E 7 pencarian, blok F 6 pencarian per unit → 156 pencarian ≈ 8–14 jam serial,
# ±4–7 jam dengan 2 worker (LightGBM biasanya lebih cepat daripada XGBoost pada data sekecil ini).
#
# **Cara pakai di Colab:** sama dengan 07, ditambah `!pip install -q lightgbm` (biasanya sudah terpasang).
# Uji cepat dulu dengan `TEST_MODE = True`.

# %%
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')) if '__file__' not in globals() else os.path.dirname(os.path.abspath(__file__)))
import core as rc

# ---------------- PENGATURAN ----------------
BLOCKS_TO_RUN = ['mcc', 'lgbm']
N_WORKERS = int(os.environ.get('P2_WORKERS', 4 if rc.ON_KAGGLE else 2))   # Kaggle CPU: 4 core; Colab gratis: 2
DEADLINE_HOURS = 10.5 if rc.ON_KAGGLE else None   # Kaggle mematikan sesi pada 12 jam → berhenti mulai pekerjaan baru di 10,5 jam
TEST_MODE = bool(int(os.environ.get('P2_REV_TEST', 0)))
# --------------------------------------------

BLOCKS = {
    'lgbm': dict(family='lgbm', fitness='wf1', include_default=True,
                 arms=[('RS_lgbm', 'RS', 'lin'), ('SMA_lgbm', 'SMA', 'lin'), ('SMA_lgbm_flip', 'SMA', 'flip'),
                       ('HHO_lgbm', 'HHO', 'lin'), ('HHO_lgbm_flip', 'HHO', 'flip'),
                       ('TPE_lgbm', 'TPE', 'lin'), ('TPE_lgbm_flip', 'TPE', 'flip')]),
    'mcc': dict(family='xgb', fitness='mcc', include_default=False,
                arms=[('RS_mcc', 'RS', 'lin'), ('SMA_mcc', 'SMA', 'lin'), ('SMA_mcc_flip', 'SMA', 'flip'),
                      ('HHO_mcc', 'HHO', 'lin'), ('HHO_mcc_flip', 'HHO', 'flip'), ('TPE_mcc', 'TPE', 'lin')]),
}
OUT = os.path.join(rc.default_output_base(), 'subset_lightgbm_mcc' + ('_TEST' if TEST_MODE else ''))
UNITS12 = rc.SUBSET_12[:1] if TEST_MODE else rc.SUBSET_12
extra = dict(budget=40, pop=10) if TEST_MODE else {}
JOBS = [dict(block=b, ds=ds, ir=ir, run=r, **BLOCKS[b], **extra) for b in BLOCKS_TO_RUN for ds, ir, r in UNITS12]
print(f'Output → {OUT}\nBlok: {BLOCKS_TO_RUN} | unit: {len(UNITS12)} | pekerjaan: {len(JOBS)} | worker: {N_WORKERS}')

# %%
# ============================================================
# 1. JALANKAN (checkpoint per unit × blok)
# ============================================================
if rc.ON_KAGGLE:
    rc.restore_checkpoints(OUT, os.path.basename(OUT))   # lanjutkan dari output versi sebelumnya (bila ditambahkan sebagai Input)
LEFT = rc.run_blocks_parallel(JOBS, OUT, n_workers=N_WORKERS, deadline_hours=DEADLINE_HOURS)

# %%
# ============================================================
# 2. ANALISIS
# ============================================================
R = rc.load_block_results(OUT)
units = sorted(R.unit.unique())
REF = rc.load_main_reference(units)
LONG = pd.concat([R[['unit', 'arm', 'reeval_cv_wf1', 'reeval_cv_mf1', 'reeval_cv_mcc', 'ho_wf1',
                     'final_real_frac_at_lower', 'final_real_frac_at_upper']], REF], ignore_index=True)
LONG.to_csv(os.path.join(OUT, 'long.csv'), index=False)
PAIRS = {
    'E_lgbm': [('SMA_lgbm', 'RS_lgbm', 'SMA − RS (LightGBM)'), ('HHO_lgbm', 'RS_lgbm', 'HHO − RS (LightGBM)'),
               ('TPE_lgbm', 'RS_lgbm', 'TPE − RS (LightGBM)'),
               ('SMA_lgbm', 'SMA_lgbm_flip', 'flip SMA (LightGBM)'), ('HHO_lgbm', 'HHO_lgbm_flip', 'flip HHO (LightGBM)'),
               ('TPE_lgbm', 'TPE_lgbm_flip', 'flip TPE (LightGBM, kontrol)'),
               ('RS_lgbm', 'default', 'RS − default (LightGBM)'), ('SMA_lgbm', 'default', 'SMA − default (LightGBM)'),
               ('SMA_lin', 'RS_lin', 'SMA − RS (XGBoost, eksperimen utama)'),
               ('SMA_lin', 'SMA_lin_flip', 'flip SMA (XGBoost, eksperimen utama)')],
    'F_mcc': [('SMA_mcc', 'RS_mcc', 'SMA − RS (fitness MCC)'), ('HHO_mcc', 'RS_mcc', 'HHO − RS (fitness MCC)'),
              ('TPE_mcc', 'RS_mcc', 'TPE − RS (fitness MCC)'),
              ('SMA_mcc', 'SMA_mcc_flip', 'flip SMA (fitness MCC)'), ('HHO_mcc', 'HHO_mcc_flip', 'flip HHO (fitness MCC)'),
              ('SMA_mcc', 'SMA_lin', 'SMA: fitness MCC − fitness wF1'), ('RS_mcc', 'RS_lin', 'RS: fitness MCC − fitness wF1')],
}
# 'default' untuk blok E adalah default LightGBM (arm 'default' dari blok lgbm), bukan default XGBoost
LONG_E = LONG.copy()
LONG_E.loc[LONG_E.arm == 'default_lin', 'arm'] = 'default_xgb'
tables = []
for metric in ('reeval_cv_wf1', 'reeval_cv_mf1', 'reeval_cv_mcc'):
    wide = LONG_E.pivot_table(index='unit', columns='arm', values=metric)
    for blk, pairs in PAIRS.items():
        T = rc.paired_summary(wide, pairs, metric)
        if len(T):
            T.insert(0, 'block', blk); tables.append(T)
PT = pd.concat(tables, ignore_index=True)
PT.to_csv(os.path.join(OUT, 'paired_tests.csv'), index=False)
pd.set_option('display.width', 220)
show = PT[((PT.block == 'E_lgbm') & (PT.metric == 'reeval_cv_wf1')) | ((PT.block == 'F_mcc') & PT.metric.isin(['reeval_cv_mcc', 'reeval_cv_mf1']))]
print(show[['block', 'metric', 'comparison', 'n', 'median_delta', 'ci_lo', 'ci_hi', 'n_pos', 'n_neg', 'p', 'p_holm']].round(4).to_string(index=False))

# Di mana wilayah yang baik pada LightGBM? Spearman wF1 vs tiap hyperparameter dari 300 evaluasi RS per unit
rows = []
for f in glob.glob(os.path.join(OUT, 'units', '*__lgbm.npz')):
    z = np.load(f)
    X, y = z['opt_RS_lgbm_Xreal'], z['opt_RS_lgbm_wf1']
    for j, (n, *_r) in enumerate(rc.PARAM_SPACE_LGBM):
        rows.append(dict(unit=os.path.basename(f), hp=n, rho=stats.spearmanr(X[:, j], y)[0]))
if rows:
    SP = pd.DataFrame(rows).groupby('hp').rho.agg(['median', lambda s: (s < 0).mean()]).rename(
        columns={'<lambda_0>': 'frac_units_negative'}).sort_values('median')
    SP.to_csv(os.path.join(OUT, 'lgbm_spearman_wf1_vs_hp.csv'))
    print('\nLightGBM — median Spearman wF1 vs hyperparameter (negatif = nilai kecil lebih baik):'); print(SP.round(3))

POS = LONG.groupby('arm')[['final_real_frac_at_lower', 'final_real_frac_at_upper']].mean().round(3)
POS.to_csv(os.path.join(OUT, 'positions_by_arm.csv'))
# Cara membaca: bila 'flip SMA (LightGBM)' positif dan reg_alpha juga negatif pada LightGBM, mekanisme yang sama
# berlaku di model lain. Pada blok F, 'flip SMA (fitness MCC)' pada metrik MCC/F1 minoritas menunjukkan apakah bias SMA
# juga memengaruhi deteksi kelas minoritas ketika pencarian dipandu oleh metrik yang berfokus pada minoritas.
print('Selesai. Keluaran:', OUT)
