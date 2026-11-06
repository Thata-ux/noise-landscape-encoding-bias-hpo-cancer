# %% [markdown]
# # 06 — Analisis ulang dari unit yang sudah tersimpan
#
# Tidak ada pelatihan XGBoost. Semua analisis dihitung dari `results/units/*.npz|json`, `results/tidy/*.csv`, dan
# `results/baseline_tpe/`, kecuali bagian 5 dan 6 yang menjalankan optimizer pada fitness sintetis (tanpa model).
#
# | Bagian | Isi | Keluaran utama (results/reanalysis/) |
# |---|---|---|
# | 1 | CI bootstrap SNR, dekomposisi varians dua-arah, ICC(C,1) (Tabel S6) | `1a_snr_per_unit.csv`, `1b_snr_per_condition.csv` |
# | 2 | Rasio neutralitas dengan ambang tetap (Tabel S8) | `2_neutrality_fixed_threshold.csv` |
# | 3 | Korelasi SNR–learnability per dataset dan indikator plateau | `3a_snr_r2_by_dataset.csv`, `3b_plateau_per_condition.csv` |
# | 4 | Gambar 8 (winner's curse, termasuk TPE) | `Fig8_winners_curse_withTPE.png/pdf` |
# | 5 | Uji bias struktural gaya BIAS toolbox, termasuk SMAb, SMAm, DE, CMA-ES (Tabel S7) | `5a_bias_tests.csv`, `5_bias_raw.csv` |
# | 6 | Kontribusi tiap operator HHO terhadap tarikan ke titik asal | `6_hho_operator_log.csv`, `6_hho_final_positions.csv` |
#
# **Cara pakai di Colab:**
# ```
# !git clone https://github.com/Thata-ux/noise-landscape-encoding-bias-hpo-cancer.git
# %cd noise-landscape-encoding-bias-hpo-cancer/src
# !pip install -q xgboost==3.4.1 optuna==4.1.0 cma statsmodels
# ```
# lalu jalankan semua sel. Perkiraan waktu total: ±1–5 menit (tidak ada pelatihan model; bagian 5 dan 6 memakai
# fitness sintetis).
# Hasil disimpan di `results/reanalysis` (atau di Google Drive bila ter-mount).

# %%
# from google.colab import drive; drive.mount('/content/drive')
import os, sys, json, glob, math, time
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')) if '__file__' not in globals() else os.path.dirname(os.path.abspath(__file__)))
import core as rc

REPO = rc.REPO
UNITS = os.path.join(REPO, 'results', 'units')
TIDY = os.path.join(REPO, 'results', 'tidy')
OUT = os.path.join(rc.default_output_base(), 'reanalysis'); os.makedirs(OUT, exist_ok=True)
N_BOOT = 5000
RNG = np.random.default_rng(2026)
noise = pd.read_csv(os.path.join(TIDY, 'noise.csv'))
fla = pd.read_csv(os.path.join(TIDY, 'fla.csv'))
for df in (noise, fla):
    df['cond'] = df['dataset'] + '|' + df['ir_target'].astype(str).map(lambda v: v if v == 'asli' else f'{v}:1')
print('Unit:', len(noise), '| output →', OUT)


def boot_ci(x, func=np.median, B=N_BOOT):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    v = func(x[RNG.integers(0, len(x), (B, len(x)))], axis=1)
    return tuple(np.percentile(v, [2.5, 97.5]))


def paired(d, label, **extra):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    nz = d[d != 0]
    if len(nz) >= 2:
        p = stats.wilcoxon(nz).pvalue
        rk = stats.rankdata(np.abs(nz)); r = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    else:
        p, r = np.nan, np.nan
    lo, hi = boot_ci(d)
    return dict(comparison=label, n=len(d), median_delta=np.median(d), ci_lo=lo, ci_hi=hi,
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), r_rank_biserial=r, p=p, **extra)

# %%
# ============================================================
# 1. SNR: CI bootstrap & dekomposisi varians dua-arah (konfigurasi × partisi)
# ============================================================
# Matriks noise_wf1 (30 konfigurasi × 10 partisi) per unit. Partisi SAMA untuk semua konfigurasi (desain bersilang),
# sehingga varians dapat dipisah menjadi: σ²_c (konfigurasi = sinyal), σ²_p (efek utama partisi), σ²_e (interaksi/residu).
# - SNR (naskah)      = σ²_sinyal / σ²_noise, dengan σ²_noise = rata-rata varians dalam-konfigurasi = σ²_p + σ²_e (kira-kira)
# - SNR_rank          = σ²_c / σ²_e  → noise yang benar-benar dapat mengubah PERINGKAT dalam satu pencarian
# - ICC(1,1) (naskah) = σ²_sinyal / (σ²_sinyal + σ²_noise);  ICC(C,1) = σ²_c / (σ²_c + σ²_e)  (consistency)
# - share_partition   = σ²_p / (σ²_p + σ²_e) → porsi noise naskah yang hanya pergeseran seragam antar-partisi
def decompose(F):
    n, k = F.shape
    g = F.mean(); rm = F.mean(1); cm = F.mean(0)
    SSc = k * ((rm - g) ** 2).sum(); SSp = n * ((cm - g) ** 2).sum()
    SSe = ((F - rm[:, None] - cm[None, :] + g) ** 2).sum()
    MSc, MSp, MSe = SSc / (n - 1), SSp / (k - 1), SSe / ((n - 1) * (k - 1))
    s2c, s2p, s2e = max((MSc - MSe) / k, 0), max((MSp - MSe) / n, 0), MSe
    within = F.var(1, ddof=1).mean()
    sig = max(rm.var(ddof=1) - within / k, 0)
    return dict(snr=sig / within if within > 0 else np.nan, icc1=sig / (sig + within) if (sig + within) > 0 else np.nan,
                snr_rank=s2c / s2e if s2e > 0 else np.nan, icc_c1=s2c / (s2c + s2e) if (s2c + s2e) > 0 else np.nan,
                s2_config=s2c, s2_partition=s2p, s2_resid=s2e, s2_noise_oneway=within,
                share_partition=s2p / (s2p + s2e) if (s2p + s2e) > 0 else np.nan)


def snr_boot_unit(F, B=2000):
    """CI SNR per unit: resampling konfigurasi (baris) dengan pengembalian."""
    n, k = F.shape
    Fb = F[RNG.integers(0, n, (B, n))]                     # (B, n, k)
    within = Fb.var(axis=2, ddof=1).mean(axis=1)
    sig = np.maximum(Fb.mean(axis=2).var(axis=1, ddof=1) - within / k, 0)
    return tuple(np.percentile(sig / within, [2.5, 97.5]))


rows = []
for _, u in noise.iterrows():
    z = np.load(os.path.join(UNITS, u['unit'] + '.npz'))
    F = z['noise_wf1'].astype(float)
    d = decompose(F)
    lo, hi = snr_boot_unit(F)
    rows.append(dict(unit=u['unit'], dataset=u['dataset'], cond=u['cond'], scheme=u['scheme'], run=u['run'],
                     snr_stored=u['snr'], **d, snr_ci_lo=lo, snr_ci_hi=hi))
A = pd.DataFrame(rows)
print('Cek reproduksi SNR (maks |selisih| terhadap noise.csv; selisih kecil karena matriks disimpan float32):', float((A.snr - A.snr_stored).abs().max()))
A.to_csv(os.path.join(OUT, '1a_snr_per_unit.csv'), index=False)

rows = []
for (c, sc), g in A.groupby(['cond', 'scheme'], sort=False):
    lo, hi = boot_ci(g.snr)
    rows.append(dict(cond=c, scheme=sc, n_runs=len(g), snr_median=g.snr.median(), snr_ci_lo=lo, snr_ci_hi=hi,
                     n_units_snr0=int((g.snr == 0).sum()), snr_rank_median=g.snr_rank.median(),
                     icc1_median=g.icc1.median(), icc_c1_median=g.icc_c1.median(),
                     share_partition_median=g.share_partition.median(),
                     unit_ci_width_median=(g.snr_ci_hi - g.snr_ci_lo).median()))
Bc = pd.DataFrame(rows)
Bc.to_csv(os.path.join(OUT, '1b_snr_per_condition.csv'), index=False)
print(Bc.round(3).to_string(index=False))
KEY = dict(
    share_partition_median_all=float(A.share_partition.median()),
    share_partition_median_by_scheme=A.groupby('scheme').share_partition.median().round(3).to_dict(),
    snr_rank_over_snr_median=float((A.snr_rank / A.snr.replace(0, np.nan)).median()),
    spearman_snr_vs_snr_rank=float(stats.spearmanr(A.snr, A.snr_rank)[0]),
    n_units_snr0=int((A.snr == 0).sum()))
print(KEY)

# %%
# ============================================================
# 2. Rasio neutralitas dengan ambang TETAP (bukan σ_noise skema masing-masing)
# ============================================================
# Ambang naskah: σ_noise unit itu sendiri → pada cv5x3 ambang ikut mengecil (mekanis).
# Di sini: (i) ambang naskah (reproduksi), (ii) ambang = σ_noise unit cv5x1 pasangannya untuk KEDUA skema,
# (iii) ambang absolut 0,001 / 0,0025 / 0,005 wF1.
ABS = [0.001, 0.0025, 0.005]
sig1 = noise[noise.scheme == 'cv5x1'].set_index(['dataset', 'ir_target', 'run']).sigma_noise
rows = []
for _, u in noise.iterrows():
    walk = np.load(os.path.join(UNITS, u['unit'] + '.npz'))['fla_walk_wf1'].astype(float)
    steps = np.abs(np.diff(walk))
    r = dict(unit=u['unit'], dataset=u['dataset'], cond=u['cond'], ir_target=u['ir_target'], scheme=u['scheme'],
             run=u['run'], neutral_own=np.mean(steps < u['sigma_noise']),
             neutral_fixed_cv5x1=np.mean(steps < sig1.loc[(u['dataset'], u['ir_target'], u['run'])]))
    for a in ABS:
        r[f'neutral_abs_{a}'] = np.mean(steps < a)
    rows.append(r)
N = pd.DataFrame(rows)
chk = N.merge(fla[['unit', 'neutral_ratio']], on='unit')
print('Cek reproduksi neutralitas (maks |selisih|):', float((chk.neutral_own - chk.neutral_ratio).abs().max()))
pv = N.pivot_table(index=['dataset', 'cond', 'run'], columns='scheme',
                   values=['neutral_own', 'neutral_fixed_cv5x1'] + [f'neutral_abs_{a}' for a in ABS])
res = []
for col, lab in [('neutral_own', 'ambang σ_noise skema sendiri (naskah)'),
                 ('neutral_fixed_cv5x1', 'ambang tetap = σ_noise cv5x1 pasangan')] + \
                [(f'neutral_abs_{a}', f'ambang absolut {a}') for a in ABS]:
    d = (pv[(col, 'cv5x3')] - pv[(col, 'cv5x1')]).values
    by_cond = (pv[(col, 'cv5x3')] - pv[(col, 'cv5x1')]).groupby(level='cond').mean()
    res.append(paired(d, 'cv5x3 − cv5x1', threshold=lab, conds_negative=f'{int((by_cond < 0).sum())}/{len(by_cond)}'))
R2 = pd.DataFrame(res)
R2.to_csv(os.path.join(OUT, '2_neutrality_fixed_threshold.csv'), index=False)
N.to_csv(os.path.join(OUT, '2_neutrality_per_unit.csv'), index=False)
print(R2.round(4).to_string(index=False))

# %%
# ============================================================
# 3. Anomali Thyroid: korelasi SNR–learnability per dataset & indikator plateau
# ============================================================
M = noise[['unit', 'dataset', 'cond', 'scheme', 'run', 'snr']].merge(fla[['unit', 'rf_oob_r2', 'f_range']], on='unit')
rows = []
for (ds, sc), g in list(M.groupby(['dataset', 'scheme'])) + [((ds, 'pooled'), g) for ds, g in M.groupby('dataset')]:
    rho, p = stats.spearmanr(g.snr, g.rf_oob_r2)
    rows.append(dict(dataset=ds, scheme=sc, n_units=len(g), spearman_rho=rho, p=p))
rho, p = stats.spearmanr(M.snr, M.rf_oob_r2)
rows.append(dict(dataset='ALL', scheme='pooled', n_units=len(M), spearman_rho=rho, p=p))
C3 = pd.DataFrame(rows); C3.to_csv(os.path.join(OUT, '3a_snr_r2_by_dataset.csv'), index=False)
print(C3.round(3).to_string(index=False))

# indikator plateau dari 200 sampel FLA: proporsi nilai wF1 unik & porsi nilai modus (dataran terbesar)
rows = []
for _, u in M.iterrows():
    w = np.round(np.load(os.path.join(UNITS, u['unit'] + '.npz'))['fla_sample_wf1'].astype(float), 8)
    vals, cnt = np.unique(w, return_counts=True)
    top10 = np.sort(w)[-20:]
    rows.append(dict(unit=u['unit'], dataset=u['dataset'], cond=u['cond'], scheme=u['scheme'], snr=u['snr'],
                     rf_oob_r2=u['rf_oob_r2'], frac_unique=len(vals) / len(w), modal_share=cnt.max() / len(w),
                     n_unique_top10pct=len(np.unique(top10)),
                     skew=stats.skew(w)))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, '3b_plateau_per_unit.csv'), index=False)
Pc = P.groupby(['dataset', 'cond', 'scheme']).agg(snr_median=('snr', 'median'), r2_mean=('rf_oob_r2', 'mean'),
                                                  frac_unique=('frac_unique', 'mean'), modal_share=('modal_share', 'mean'),
                                                  skew=('skew', 'mean')).reset_index()
Pc.to_csv(os.path.join(OUT, '3b_plateau_per_condition.csv'), index=False)
print(Pc.round(3).to_string(index=False))
rho_u, p_u = stats.spearmanr(P.frac_unique, P.rf_oob_r2)
print(f'Spearman (proporsi nilai unik vs R² surrogate), semua unit: ρ={rho_u:.2f}, p={p_u:.3g}')
KEY['thyroid_frac_unique_mean'] = float(P[P.dataset == 'Thyroid'].frac_unique.mean())
KEY['other_frac_unique_mean'] = float(P[P.dataset != 'Thyroid'].frac_unique.mean())
KEY['rho_unique_vs_r2'] = float(rho_u)

# %%
# ============================================================
# 4. Gambar 8 dengan TPE (cv5x1 saja)
# ============================================================
import matplotlib.pyplot as plt
_A = open(os.path.join(REPO, 'src', '02_analysis.py'), encoding='utf-8').read()
_s = lambda a, b: _A[_A.index(a):_A.index(b, _A.index(a))]
exec(_s("INK, INK2, MUTED", "FIG_NAME_EN = {"))
exec(_s("from matplotlib.ticker import", "# %%"))
LANG = os.environ.get('P2_LANG', 'en')
L = (lambda id_, en: en) if LANG == 'en' else (lambda id_, en: id_)

opt = pd.read_csv(os.path.join(TIDY, 'optimizers.csv'))
W = opt[(opt.flipped == False) & opt.optimizer.isin(OPT_ORDER)][['unit', 'scheme', 'optimizer', 'winners_curse']]
tpe = pd.read_csv(os.path.join(REPO, 'results', 'baseline_tpe', 'tpe_long.csv'))
Wt = tpe[tpe.metode == 'TPE'][['unit', 'winners_curse']].assign(scheme='cv5x1', optimizer='TPE')
W = pd.concat([W, Wt]).merge(noise[['unit', 'snr']], on='unit')
W['log_snr'] = np.log(W.snr.clip(lower=1e-4))
ORD = OPT_ORDER + ['TPE']
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
ax = axes[0]; x = np.arange(len(ORD)); w = 0.36
for j, sc in enumerate(['cv5x1', 'cv5x3']):
    for i, o in enumerate(ORD):
        g = W[(W.optimizer == o) & (W.scheme == sc)].winners_curse
        if not len(g):
            continue
        m_ = g.mean(); lo, hi = boot_ci(g, np.mean)
        xx = x[i] + (j - 0.5) * w if o != 'TPE' else x[i]
        ax.bar(xx, m_, w * 0.92, color=OPT_COLOR[o] if sc == 'cv5x1' else 'white', edgecolor=OPT_COLOR[o], lw=1.3,
               hatch=None if sc == 'cv5x1' else '///')
        ax.errorbar(xx, m_, yerr=[[m_ - lo], [hi - m_]], fmt='none', ecolor=INK2, lw=0.9, capsize=2)
ax.set_xticks(x); ax.set_xticklabels([o if o != 'TPE' else 'TPE\n(cv5x1)' for o in ORD])
ax.set_ylabel(L("Winner's curse (wF1)\nCV terbaik − evaluasi ulang", "Winner's curse (wF1)\nbest CV − re-evaluation"))
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec=MUTED), plt.Rectangle((0, 0), 1, 1, fc='white', ec=MUTED, hatch='///')],
          labels=['cv5x1', 'cv5x3'], loc='upper left'); ax.grid(axis='x', visible=False); panel(ax, 'a')
ax = axes[1]
base = W[W.optimizer.isin(OPT_ORDER)]
_, edges = pd.qcut(base.log_snr, 5, retbins=True); edges[0], edges[-1] = -np.inf, np.inf
W['snr_bin'] = pd.cut(W.log_snr, edges, labels=False)
# TPE TIDAK diplot di panel b: TPE hanya ada pada cv5x1, sedangkan garis lain menggabungkan cv5x1 dan cv5x3,
# sehingga perbandingan per kuintil SNR tidak setara.
for o in OPT_ORDER:
    g = W[W.optimizer == o].groupby('snr_bin').agg(x=('snr', 'median'), y=('winners_curse', 'mean'))
    ax.plot(g.x, g.y, marker='o', ms=4, color=OPT_COLOR[o], label=o, lw=1.8, ls='--' if o == 'RS' else '-')
ax.set_xscale('log'); ax.set_xlabel(L('SNR (median per kuintil, skala log)', 'SNR (median per quintile, log scale)'))
ax.set_ylabel(L("Winner's curse (rata-rata)", "Winner's curse (mean)")); logticks(ax, 'x', [0.3, 1, 3, 10, 30]); panel(ax, 'b')
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=5, bbox_to_anchor=(0.72, 1.07))
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'Fig8_winners_curse_withTPE.{ext}'), bbox_inches='tight', pad_inches=0.08, dpi=300)
plt.close(fig)
print('🖼  Fig8_winners_curse_withTPE disimpan (TPE di panel a saja; panel b sama dengan naskah).')

# %%
# ============================================================
# 5. Uji bias struktural gaya BIAS toolbox (Kononova dkk., 2015; Vermetten dkk., 2022)
# ============================================================
# Fitness acak i.i.d. U(0,1) (dan konstan). Algoritma tanpa bias → posisi solusi terbaik akhir per ulangan
# tersebar seragam di [0,1] pada setiap dimensi. Diuji per dimensi dengan Kolmogorov–Smirnov dan Cramér–von Mises
# terhadap U(0,1), koreksi Holm atas 9 dimensi; dilaporkan juga centroid populasi akhir (seperti Tabel S3).
# RS adalah kontrol negatif (harus lolos). Optimizer yang dipakai pada 07_subset_encoding_variants (SMAm, SMAb, DE,
# CMAES) ikut diuji di sini.
from statsmodels.stats.multitest import multipletests
BIAS_REPS = int(os.environ.get('P2_BIAS_REPS', 200))
BIAS_METHODS = ['RS', 'BFO', 'HHO', 'SMA', 'SMAc', 'SMAm', 'SMAb', 'DE', 'CMAES']
BIAS_FITNESS = ['acak', 'konstan']


class FakeEval:
    def __init__(self, kind, rng):
        self.kind, self.rng = kind, rng

    def __call__(self, x):
        f = float(self.rng.random()) if self.kind == 'acak' else 0.5
        return dict(f=f, wf1=-f, mf1=0.0, mcc=0.0)


def bias_run(method, kind, rep):
    obj = rc.Objective(FakeEval(kind, np.random.default_rng([rep, 99])), rc.S['opt_budget'])
    tr = rc.Tracker(obj)
    try:
        rc.ALL_METHODS[method](obj, np.random.default_rng([rep, 7]), rc.S['pop_size'], tr)
    except rc.BudgetExhausted:
        pass
    # 'solusi terbaik' pada fitness konstan = titik pertama (tidak informatif) → pakai solusi terakhir yang dievaluasi
    best = obj.best_x if kind == 'acak' else np.array(obj.X_log[-1])
    # RS tidak memiliki populasi → 20 titik terakhir yang dievaluasi dipakai sebagai 'populasi akhir'
    c = tr.last_pop.mean(axis=0) if tr.last_pop is not None else np.array(obj.X_log[-rc.S['pop_size']:]).mean(axis=0)
    return best, c


raw, t0 = [], time.time()
for kind in BIAS_FITNESS:
    for m in BIAS_METHODS:
        _t = time.time()
        for rep in range(BIAS_REPS):
            best, c = bias_run(m, kind, rep)
            raw.append(dict(fitness=kind, method=m, rep=rep, **{f'best_{j}': v for j, v in enumerate(best)},
                            **{f'cen_{j}': v for j, v in enumerate(c)}))
        print(f'   {kind:<8} {m:<6} {BIAS_REPS} ulangan, {time.time() - _t:.0f} dtk', flush=True)
RAWB = pd.DataFrame(raw); RAWB.to_csv(os.path.join(OUT, '5_bias_raw.csv'), index=False)
rows = []
for (kind, m), g in RAWB.groupby(['fitness', 'method'], sort=False):
    B_ = g[[f'best_{j}' for j in range(rc.D)]].values; C_ = g[[f'cen_{j}' for j in range(rc.D)]].values
    p_ks = [stats.kstest(B_[:, j], 'uniform').pvalue for j in range(rc.D)]
    p_cvm = [stats.cramervonmises(B_[:, j], 'uniform').pvalue for j in range(rc.D)]
    rej_ks = multipletests(p_ks, method='holm')[0]; rej_cvm = multipletests(p_cvm, method='holm')[0]
    cm = C_.mean(axis=0)
    rows.append(dict(fitness=kind, method=m, reps=len(g), dims_reject_KS=int(rej_ks.sum()), dims_reject_CvM=int(rej_cvm.sum()),
                     min_p_KS=float(min(p_ks)), mean_best_pos=float(B_.mean()),
                     centroid_mean=float(C_.mean()), frac_dims_lower=float((C_ < 0.02).mean()),
                     frac_dims_upper=float((C_ > 0.98).mean()), frac_dims_center=float((np.abs(C_ - 0.5) < 0.02).mean()),
                     verdict='bias' if rej_cvm.sum() > 0 else 'tidak terdeteksi'))
BT = pd.DataFrame(rows)
BT.to_csv(os.path.join(OUT, '5a_bias_tests.csv'), index=False)
print(BT.round(3).to_string(index=False))
print(f'Bagian 5 selesai dalam {(time.time() - t0) / 60:.1f} menit')

# %%
# ============================================================
# 6. HHO: operator mana yang menarik populasi ke titik asal?
# ============================================================
# Salinan run_hho v5.2 dengan pencatatan per pembaruan posisi (operator, posisi sebelum/sesudah). Kesetaraan dengan
# kode asli diperiksa: dengan seed yang sama, jejak evaluasinya harus identik.
def run_hho_logged(obj, rng, pop_size, track, log):
    D = rc.D
    P = rng.random((pop_size, D)); F = np.array([obj(x) for x in P]); track(P)
    while True:
        b = np.argmin(F); Xr, Fr = P[b].copy(), F[b]
        for i in range(pop_size):
            before = P[i].copy()
            E = 2 * (2 * rng.random() - 1) * (1 - obj.progress)
            if abs(E) >= 1:
                if rng.random() >= 0.5:
                    Xk = P[rng.integers(pop_size)]
                    P[i] = Xk - rng.random() * np.abs(Xk - 2 * rng.random() * P[i]); op = 'eksplorasi-1 (acak)'
                else:
                    P[i] = (Xr - P.mean(axis=0)) - rng.random() * rng.random(D); op = 'eksplorasi-2 (Xr − Xm − r(LB+rUB))'
                P[i] = np.clip(P[i], 0, 1); F[i] = obj(P[i])
            else:
                r, Jp = rng.random(), 2 * (1 - rng.random())
                if r >= 0.5 and abs(E) >= 0.5:
                    P[i] = np.clip((Xr - P[i]) - E * np.abs(Jp * Xr - P[i]), 0, 1); F[i] = obj(P[i]); op = 'soft besiege'
                elif r >= 0.5:
                    P[i] = np.clip(Xr - E * np.abs(Xr - P[i]), 0, 1); F[i] = obj(P[i]); op = 'hard besiege'
                else:
                    ref = P[i] if abs(E) >= 0.5 else P.mean(axis=0)
                    Y = np.clip(Xr - E * np.abs(Jp * Xr - ref), 0, 1); fY = obj(Y); op = 'rapid dives (ditolak)'
                    if fY < F[i]:
                        P[i], F[i] = Y, fY; op = 'rapid dives (diterima)'
                    else:
                        Z = np.clip(Y + rng.random(D) * rc._levy(rng), 0, 1); fZ = obj(Z)
                        if fZ < F[i]:
                            P[i], F[i] = Z, fZ; op = 'rapid dives (diterima)'
            log.append(dict(op=op, progress=obj.progress, mean_before=before.mean(), mean_after=P[i].mean(),
                            frac_low_after=float(np.mean(P[i] < 0.02))))
            if F[i] < Fr:
                Xr, Fr = P[i].copy(), F[i]
        track(P)


class SphereEval:
    """Fitness informatif sintetis: jarak kuadrat ke optimum c (minimisasi)."""
    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        f = float(np.sum((np.asarray(x) - self.c) ** 2))
        return dict(f=f, wf1=-f, mf1=0.0, mcc=0.0)


# cek kesetaraan dengan run_hho asli
o1 = rc.Objective(FakeEval('acak', np.random.default_rng([0, 99])), 300); t1 = rc.Tracker(o1)
o2 = rc.Objective(FakeEval('acak', np.random.default_rng([0, 99])), 300); t2 = rc.Tracker(o2)
for fn, o, t in ((rc.run_hho, o1, t1), (lambda a, b, c, d: run_hho_logged(a, b, c, d, []), o2, t2)):
    try:
        fn(o, np.random.default_rng([0, 7]), 20, t)
    except rc.BudgetExhausted:
        pass
print('Salinan HHO identik dengan v5.2:', bool(np.allclose(np.array(o1.X_log), np.array(o2.X_log))))

HHO_REPS = int(os.environ.get('P2_HHO_REPS', 100))
logs, fin = [], []
for kind in ('acak', 'konstan', 'sphere_c0.7', 'sphere_c0.3'):
    for rep in range(HHO_REPS):
        if kind.startswith('sphere'):
            ev = SphereEval(np.full(rc.D, float(kind.split('c')[1])))
        else:
            ev = FakeEval(kind, np.random.default_rng([rep, 99]))
        obj = rc.Objective(ev, 300); tr = rc.Tracker(obj); lg = []
        try:
            run_hho_logged(obj, np.random.default_rng([rep, 7]), 20, tr, lg)
        except rc.BudgetExhausted:
            pass
        for r in lg:
            r.update(fitness=kind, rep=rep)
        logs += lg
        c = tr.last_pop.mean(axis=0)
        fin.append(dict(fitness=kind, rep=rep, centroid=c.mean(), frac_lower=np.mean(c < 0.02)))
LG = pd.DataFrame(logs)
LG['shift'] = LG.mean_after - LG.mean_before
LG['phase'] = np.where(LG.progress < 0.5, 'paruh awal', 'paruh akhir')
T6 = (LG.groupby(['fitness', 'op']).agg(n=('shift', 'size'), share=('shift', 'size'), mean_shift=('shift', 'mean'),
                                        mean_pos_after=('mean_after', 'mean'), frac_low_after=('frac_low_after', 'mean'))
      .reset_index())
T6['share'] = T6.groupby('fitness').n.transform(lambda s: s / s.sum())
T6.to_csv(os.path.join(OUT, '6_hho_operator_log.csv'), index=False)
FIN = pd.DataFrame(fin).groupby('fitness').agg(centroid=('centroid', 'mean'), frac_lower=('frac_lower', 'mean')).reset_index()
FIN.to_csv(os.path.join(OUT, '6_hho_final_positions.csv'), index=False)
print(T6.round(4).to_string(index=False)); print(FIN.round(3).to_string(index=False))

# %%
# ============================================================
# 7. Ringkasan angka kunci
# ============================================================
KEY['neutrality'] = R2[['threshold', 'median_delta', 'n_neg', 'n', 'p']].round(5).to_dict('records')
KEY['bias_verdicts'] = BT[['fitness', 'method', 'dims_reject_CvM', 'centroid_mean', 'frac_dims_lower', 'frac_dims_center']].round(3).to_dict('records')
json.dump(KEY, open(os.path.join(OUT, 'key_numbers.json'), 'w'), indent=1, default=float)
print(json.dumps(KEY, indent=1, default=float)[:3000])
print('\nSelesai. Semua keluaran ada di', OUT)
