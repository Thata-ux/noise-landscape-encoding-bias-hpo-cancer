# %% [markdown]
# # Paper 2 — Analisis Final
# ## Noise-Aware Fitness Landscape & Search Behavior Analysis of XGBoost HPO on Small Imbalanced Cancer Data
#
# Notebook ini **tidak menjalankan eksperimen**. Ia membaca 280 unit hasil (`units/*.json` + `units/*.npz`)
# dari `paper2_noisefla_searchbehavior_v5_2`, lalu menghasilkan semua **tabel** dan **gambar** untuk artikel.
#
# | Bagian | Isi | Keluaran |
# |---|---|---|
# | 1 | Desain & kondisi eksperimen | Tabel 1 |
# | 2 | **RQ1** noise evaluasi & komposisi kelas | Tabel 2–3, Gambar 1–2 |
# | 3 | **RQ2a** karakteristik landscape | Tabel 4–6, Gambar 3–4 |
# | 4 | **RQ2b** perilaku pencarian & bias encoding (flip test) | Tabel 7, Gambar 5–6 |
# | 5 | **RQ3** reliabilitas optimasi | Tabel 8–9, Gambar 7–8 |
# | 6 | Biaya komputasi | Tabel 10 |
#
# Semua uji berpasangan memakai Wilcoxon signed-rank + rank-biserial r + bootstrap 95% CI (median);
# koreksi perbandingan berganda Holm dalam setiap keluarga uji. Model regresi: mixed model (intersep acak
# per dataset) dengan pemeriksaan robustness efek-tetap + galat baku ter-klaster per kondisi.

# %%
# ============================================================
# 0. KONFIGURASI
# ============================================================
# Di Colab: jalankan sel mount Drive dulu, lalu (sekali) `!pip install -q statsmodels openpyxl` bila perlu.
import os, glob, json, math, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Bahasa keluaran tabel & gambar: 'id' (naskah Indonesia) atau 'en' (repo GitHub & submisi berbahasa Inggris)
LANG = os.environ.get('P2_LANG', 'en')
def L(id_text, en_text):
    return en_text if LANG == 'en' else id_text

UNITS_DIR_CANDIDATES = [
    os.environ.get('P2_UNITS_DIR', ''),
    '/content/drive/MyDrive/Colab Notebooks/Paper 2/outputs/v5.2/full/thyRespDrop_cervTestsDrop/units',
    './paper2_outputs/v5.2/full/thyRespDrop_cervTestsDrop/units',
    './units',
    '../results/units',                      # struktur repositori GitHub
    './results/units',
]
UNITS_DIR = next(p for p in UNITS_DIR_CANDIDATES if p and os.path.isdir(p))
OUT_DIR = os.environ.get('P2_ANALYSIS_DIR', os.path.join(os.path.dirname(UNITS_DIR.rstrip('/')), L('analysis_final', 'analysis_final_en')))
FIG_DIR, TAB_DIR = os.path.join(OUT_DIR, 'figures'), os.path.join(OUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
N_BOOT, SEED = 2000, 2026
RNG = np.random.default_rng(SEED)
print('units  →', UNITS_DIR)
print('output →', OUT_DIR)

# %%
# ============================================================
# 0b. GAYA GAMBAR (siap jurnal) — palet tervalidasi (CVD & kontras), warna mengikuti entitas, bukan peringkat
# ============================================================
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
OPT_ORDER = ['RS', 'BFO', 'HHO', 'SMA', 'SMAc']
OPT_COLOR = {'RS': '#898781', 'BFO': '#2a78d6', 'HHO': '#eb6834', 'SMA': '#1baf7a', 'SMAc': '#eda100', 'TPE': '#7b3fb0'}
DS_ORDER = ['Lung', 'Thyroid', 'Breast', 'Cervical']
DS_COLOR = {'Lung': '#6250d6', 'Thyroid': '#008300', 'Breast': '#e87ba4', 'Cervical': '#52514e'}
DS_MARK = {'Lung': 'o', 'Thyroid': 's', 'Breast': '^', 'Cervical': 'D'}
SEQ_BLUE = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5, 'axes.labelcolor': INK2,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 2, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})


FIG_NAME_EN = {
    'Gambar1_mekanisme_noise_sinyal': 'Fig2_noise_signal_mechanism',
    'Gambar2_manipulasi_noise': 'Fig3_noise_manipulation',
    'Gambar3_landscape_noise': 'Fig4_landscape_vs_noise',
    'Gambar4_kepentingan_hyperparameter': 'Fig5_hyperparameter_importance',
    'Gambar5_perilaku_pencarian': 'Fig6_search_behaviour',
    'Gambar6_flip_test_bias_batas': 'Fig7_flip_test_boundary_bias',
    'Gambar7_winners_curse': 'Fig8_winners_curse',
    'Gambar8_forest_vs_random_search': 'Fig9_forest_vs_random_search',
}
TAB_NAME_EN = {
    'Tabel1_desain_kondisi': 'Table2_condition_design',
    'Tabel2_noise_per_kondisi_numerik': 'Table4_noise_per_condition_numeric',
    'Tabel2_noise_per_kondisi': 'Table4_noise_per_condition',
    'Tabel3_model_noise_sinyal_snr': 'Table5_noise_signal_snr_models',
    'Tabel4_landscape_per_kondisi': 'TableS4_landscape_per_condition',
    'Tabel5_efek_noise_pada_landscape': 'Table6a_noise_effect_on_landscape',
    'Tabel5b_korelasi_snr_landscape': 'Table6b_snr_landscape_correlation',
    'Tabel6_kepentingan_hyperparameter': 'TableS5_hyperparameter_importance',
    'Tabel7_flip_test': 'Table7_flip_test',
    'Tabel8a_winners_curse': 'Table9a_winners_curse',
    'Tabel8b_efek_cv5x3_pada_winners_curse': 'Table9b_cv5x3_effect_on_winners_curse',
    'Tabel8c_model_winners_curse': 'TableS6_winners_curse_model',
    'Tabel9_perbandingan_vs_RS_default': 'Table10_vs_random_search_and_default',
    'Tabel10_biaya_komputasi': 'TableS7_compute_cost',
    'Tabel10b_waktu_pencarian_per_optimizer': 'TableS8_search_time_per_optimizer',
}
# Terjemahan judul kolom & isi sel tabel (diterapkan saat menyimpan; logika analisis tetap memakai label internal)
EN = {
    'Level IR': 'IR level', 'Seri*': 'Series*', 'IR aktual (dev)': 'Actual IR (dev)', 'Positif (dev)': 'Positives (dev)',
    'Negatif (dev)': 'Negatives (dev)', 'Positif (test)': 'Positives (test)', 'Unit (2 skema × 10 run)': 'Units (2 schemes × 10 runs)',
    'asli': 'original', 'σ sinyal': 'σ signal', 'Respons': 'Response', 'Prediktor': 'Predictor',
    'β (FE-klaster)': 'β (FE, cluster-robust)', 'log σ sinyal': 'log σ signal',
    'log(jumlah positif)': 'log(no. of positives)', 'log(jumlah negatif)': 'log(no. of negatives)',
    'Klaster top-10%': 'Top-10% clusters', 'Panjang korelasi': 'Correlation length', 'Rasio neutralitas': 'Neutrality ratio',
    'Rentang wF1': 'wF1 range', 'R² surrogate (learnability)': 'Surrogate R² (learnability)', 'metrik': 'metric', 'cakupan': 'scope', 'Gabungan': 'Pooled', 'F1 minoritas': 'Minority F1',
    'skema': 'scheme', 'rata-rata [95% CI]': 'mean [95% CI]', 'rasio rata-rata cv5x3/cv5x1': 'mean ratio cv5x3/cv5x1',
    'Δ median (cv5x1 − cv5x3) [95% CI]': 'median Δ (cv5x1 − cv5x3) [95% CI]', 'Δ median [95% CI]': 'median Δ [95% CI]',
    'pembanding': 'reference', 'menit/unit (rata-rata)': 'min/unit (mean)', 'unit': 'units',
    'menit per pencarian (300 evaluasi), rata-rata': 'min per search (300 evaluations), mean',
    'BFO_flip': 'BFO (mirrored)', 'HHO_flip': 'HHO (mirrored)', 'SMA_flip': 'SMA (mirrored)',
}


def to_en(df):
    df = df.copy()
    df.columns = [EN.get(c, c) for c in df.columns]
    if df.index.name:
        df.index.name = EN.get(df.index.name, df.index.name)
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda v: EN.get(v, v) if isinstance(v, str) else v)
    return df


def savefig(fig, name):
    name = FIG_NAME_EN.get(name, name) if LANG == 'en' else name
    for ext in ('png', 'pdf'):
        # bbox eksplisit: backend inline Colab tidak selalu memakai rcParams savefig.bbox → teks tepi terpotong
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.08)
    print('🖼  ', name)


from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator


def logticks(ax, axis, ticks):
    """Sumbu log dengan label angka biasa (tanpa notasi 10^x yang saling bertumpuk)."""
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_major_locator(FixedLocator(ticks)); a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); a.set_minor_formatter(NullFormatter())


def panel(ax, letter):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK, va='bottom')

# %%
# ============================================================
# 0c. FUNGSI STATISTIK
# ============================================================
def boot_ci(x, func=np.mean, B=N_BOOT):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), (B, len(x)))
    v = np.apply_along_axis(func, 1, x[idx])
    return tuple(np.percentile(v, [2.5, 97.5]))


def paired_test(d):
    """Wilcoxon signed-rank untuk selisih berpasangan d; rank-biserial r; bootstrap CI median."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    nz = d[d != 0]
    if len(nz) < 2:
        return dict(n=len(d), median=np.median(d) if len(d) else np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    n_pos=int((d > 0).sum()), p=np.nan, r_rb=np.nan)
    rk = stats.rankdata(np.abs(nz))
    r_rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
    lo, hi = boot_ci(d, np.median)
    return dict(n=len(d), median=float(np.median(d)), ci_lo=lo, ci_hi=hi, n_pos=int((d > 0).sum()),
                p=float(stats.wilcoxon(nz).pvalue), r_rb=float(r_rb))


def holm(p):
    p = np.asarray(p, float); m = np.sum(~np.isnan(p)); order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan); running = 0
    for k, i in enumerate(order[:m]):
        running = max(running, (m - k) * p[i]); adj[i] = min(1.0, running)
    return adj


def fmt_ci(m, lo, hi, nd=4):
    return f'{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]'


def fmt_p(p):
    return 'n/a' if pd.isna(p) else ('< 0.001' if p < 0.001 else f'{p:.3f}')


TABLES = {}


def save_table(df, name, index=False):
    if LANG == 'en':
        df, name = to_en(df), TAB_NAME_EN.get(name, name)
    df.to_csv(os.path.join(TAB_DIR, f'{name}.csv'), index=index)
    TABLES[name] = df.reset_index() if index else df
    print('📄 ', name, df.shape)

# %%
# ============================================================
# 1. MUAT 280 UNIT → TABEL TIDY
# ============================================================
UNITS = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(UNITS_DIR, '*.json')))]
print('unit dimuat:', len(UNITS), pd.Series([u['dataset'] for u in UNITS]).value_counts().to_dict())
assert len({u['meta']['code_version'] for u in UNITS}) == 1, 'versi kode tercampur!'


def ir_lab(u):
    return 'asli' if u['ir_target'] is None else str(u['ir_target'])   # label internal (diterjemahkan saat simpan)


def base_row(u):
    return dict(unit=u['unit'], dataset=u['dataset'], ir_level=ir_lab(u), ir=u['ir_actual'], scheme=u['scheme'],
                run=u['run'], n_dev=u['n_dev'], pos=u['dev_pos'], neg=u['n_dev'] - u['dev_pos'],
                n_test=u['n_test'], test_pos=u['test_pos'])


NOISE = pd.DataFrame([{**base_row(u), **u['noise']} for u in UNITS])
FLA = pd.DataFrame([{**base_row(u), **u['fla']} for u in UNITS])
BASE = pd.DataFrame([{**base_row(u), 'baseline': k, **u[k]} for u in UNITS for k in ('default', 'corner', 'corner_upper')])
OPT = pd.DataFrame([{**base_row(u), 'optimizer': n,
                     **{k: v for k, v in o.items() if k not in ('trace_best', 'div', 'div_evals', 'best_x', 'best_params', 'behavior')},
                     **{k: v for k, v in o['behavior'].items() if k != 'final_centroid'}}
                    for u in UNITS for n, o in u['optimizers'].items()])
TIME = pd.DataFrame([{**base_row(u), 'runtime_min': u['runtime_sec'] / 60, **u.get('timing', {})} for u in UNITS])

# Urutan kondisi: per dataset, dari mayoritas paling sedikit → paling banyak
COND = (NOISE.groupby(['dataset', 'ir_level']).agg(pos=('pos', 'first'), neg=('neg', 'first'), ir=('ir', 'first'))
        .reset_index())
COND['ds_ord'] = COND.dataset.map({d: i for i, d in enumerate(DS_ORDER)})
COND = COND.sort_values(['ds_ord', 'ir']).reset_index(drop=True)
SERIES = {('Lung', 'asli'): 'A', ('Lung', '3'): 'A', ('Thyroid', 'asli'): 'B', ('Thyroid', '3'): 'B',
          ('Thyroid', '6'): 'B', ('Breast', 'asli'): 'B', ('Breast', '3'): 'B', ('Breast', '6'): 'B',
          ('Breast', '12'): 'B', ('Cervical', '3'): 'A', ('Cervical', '6'): 'A', ('Cervical', '12'): 'A',
          ('Cervical', 'asli'): 'A/B', ('Cervical', '20'): 'B'}
COND['series'] = [SERIES[(d, l)] for d, l in zip(COND.dataset, COND.ir_level)]
COND['label'] = [f"{d} {(L('asli ', 'orig. ') if l == 'asli' else '')}{ir:.1f}:1" for d, l, ir in zip(COND.dataset, COND.ir_level, COND.ir)]
CORD = {(d, l): i for i, (d, l) in enumerate(zip(COND.dataset, COND.ir_level))}
for df in (NOISE, FLA, BASE, OPT, TIME):
    df['cond'] = [CORD[(d, l)] for d, l in zip(df.dataset, df.ir_level)]
OPT_BASE = OPT[~OPT.flipped].copy()
print('kondisi:', len(COND))

# ---- Baseline Bayesian Optimization (TPE) — dimuat bila tersedia (notebook Paper2_Baseline_TPE) ----
# Dianalisis sebagai keluarga uji tersendiri dengan generator acak terpisah, sehingga angka analisis utama tidak berubah.
TPE_CANDIDATES = [os.environ.get('P2_TPE_UNITS', ''),
                  '/content/drive/MyDrive/Colab Notebooks/Paper 2/outputs/baseline_tpe/v5.2/full/thyRespDrop_cervTestsDrop/units',
                  '../results/baseline_tpe/units', './results/baseline_tpe/units']
TPE_DIR = next((p for p in TPE_CANDIDATES if p and os.path.isdir(p)), None)
RNG_TPE = np.random.default_rng(SEED + 1)
OPT_TPE = pd.DataFrame()
if TPE_DIR:
    _b = NOISE.set_index('unit')
    _rows = []
    for f in sorted(glob.glob(os.path.join(TPE_DIR, '*.json'))):
        u = json.load(open(f))
        if u['unit'] not in _b.index:
            continue
        for n, o in u['optimizers'].items():
            _rows.append(dict(unit=u['unit'], dataset=u['dataset'], ir_level=_b.loc[u['unit'], 'ir_level'], scheme=u['scheme'],
                              run=u['run'], cond=_b.loc[u['unit'], 'cond'], optimizer=n, base_optimizer='TPE', flipped=o['flipped'],
                              reeval_cv_wf1=o['reeval_cv_wf1'], reeval_cv_mf1=o['reeval_cv_mf1'], winners_curse=o['winners_curse'],
                              final_real_frac_at_lower=o['behavior']['final_real_frac_at_lower'],
                              final_real_frac_at_upper=o['behavior']['final_real_frac_at_upper']))
    OPT_TPE = pd.DataFrame(_rows)
HAS_TPE = len(OPT_TPE) > 0
print('TPE:', f"{OPT_TPE.unit.nunique()} unit dari {TPE_DIR}" if HAS_TPE else 'tidak ditemukan (analisis tanpa TPE)')


def with_tpe_rng(fn, *a, **k):
    """Jalankan fn dengan RNG terpisah agar bootstrap analisis utama tidak bergeser."""
    global RNG
    _save = RNG; RNG = RNG_TPE
    try:
        return fn(*a, **k)
    finally:
        RNG = _save

# %%
# ============================================================
# TABEL 1 — Desain eksperimen
# ============================================================
t1 = COND.copy()
cnt = NOISE.groupby(['dataset', 'ir_level']).agg(units=('unit', 'count'), n_dev=('n_dev', 'first'),
                                                  n_test=('n_test', 'first'), test_pos=('test_pos', 'first')).reset_index()
t1 = t1.merge(cnt, on=['dataset', 'ir_level'])
t1 = t1[['dataset', 'ir_level', 'series', 'ir', 'pos', 'neg', 'n_dev', 'n_test', 'test_pos', 'units']]
t1.columns = ['Dataset', 'Level IR', 'Seri*', 'IR aktual (dev)', 'Positif (dev)', 'Negatif (dev)', 'n dev',
              'n test', 'Positif (test)', 'Unit (2 skema × 10 run)']
t1['IR aktual (dev)'] = t1['IR aktual (dev)'].round(2)
save_table(t1, 'Tabel1_desain_kondisi')
print('* Seri A = jumlah minoritas tetap (mayoritas dikurangi); Seri B = jumlah mayoritas tetap (minoritas dikurangi).')
t1

# %% [markdown]
# ## 2. RQ1 — Noise evaluasi dan komposisi kelas

# %%
# ============================================================
# TABEL 2 — Noise, sinyal, SNR per kondisi × skema (rata-rata [95% CI bootstrap antar-run])
# ============================================================
rows = []
for (c, sc), g in NOISE.groupby(['cond', 'scheme']):
    r = dict(cond=c, scheme=sc)
    for col in ['sigma_noise', 'sigma_signal', 'icc', 'rank_stability_tau']:
        m = g[col].mean(); lo, hi = boot_ci(g[col]); r[col] = m; r[col + '_lo'] = lo; r[col + '_hi'] = hi
    r['snr_median'] = g.snr.median(); lo, hi = boot_ci(g.snr, np.median); r['snr_lo'], r['snr_hi'] = lo, hi
    rows.append(r)
T2 = COND[['dataset', 'ir_level', 'series', 'pos', 'neg']].reset_index().rename(columns={'index': 'cond'}).merge(pd.DataFrame(rows), on='cond')
save_table(T2.drop(columns='cond'), 'Tabel2_noise_per_kondisi_numerik')
T2f = T2[['dataset', 'ir_level', 'scheme', 'pos', 'neg']].copy()
T2f['σ noise'] = [fmt_ci(a, b, c) for a, b, c in zip(T2.sigma_noise, T2.sigma_noise_lo, T2.sigma_noise_hi)]
T2f['σ sinyal'] = [fmt_ci(a, b, c) for a, b, c in zip(T2.sigma_signal, T2.sigma_signal_lo, T2.sigma_signal_hi)]
T2f['SNR (median)'] = [fmt_ci(a, b, c, 2) for a, b, c in zip(T2.snr_median, T2.snr_lo, T2.snr_hi)]
T2f['ICC'] = [fmt_ci(a, b, c, 2) for a, b, c in zip(T2.icc, T2.icc_lo, T2.icc_hi)]
T2f['Kendall τ'] = [fmt_ci(a, b, c, 2) for a, b, c in zip(T2.rank_stability_tau, T2.rank_stability_tau_lo, T2.rank_stability_tau_hi)]
save_table(T2f, 'Tabel2_noise_per_kondisi')

# Rasio noise cv5x3/cv5x1 (berpasangan: dataset × level × run) vs teori 1/√3
pv = NOISE.pivot_table(index=['dataset', 'ir_level', 'run'], columns='scheme', values='sigma_noise')
NOISE_RATIO = (pv.cv5x3 / pv.cv5x1).dropna()
KEY = {}
KEY['noise_ratio_median'] = float(NOISE_RATIO.median())
KEY['noise_ratio_ci'] = boot_ci(NOISE_RATIO, np.median)
KEY['noise_ratio_theory'] = 1 / math.sqrt(3)
print(f"Rasio σ noise cv5x3/cv5x1: median {KEY['noise_ratio_median']:.3f} "
      f"[{KEY['noise_ratio_ci'][0]:.3f}, {KEY['noise_ratio_ci'][1]:.3f}] (teori {KEY['noise_ratio_theory']:.3f}, n={len(NOISE_RATIO)})")
T2f

# %%
# ============================================================
# TABEL 3 — Model: log(noise), log(sinyal), log(SNR) ~ log(positif) + log(negatif) + cv5x3
# ============================================================
import statsmodels.formula.api as smf

M = NOISE.copy()
M['log_noise'] = np.log(M.sigma_noise); M['log_signal'] = np.log(M.sigma_signal.clip(lower=1e-6))
M['log_snr'] = np.log(M.snr.clip(lower=1e-4)); M['log_pos'] = np.log(M.pos); M['log_neg'] = np.log(M.neg)
M['cv5x3'] = (M.scheme == 'cv5x3').astype(int); M['clu'] = M.dataset + '_' + M.ir_level
rows = []
MODELS = {}
for y, lab in [('log_noise', 'log σ noise'), ('log_signal', 'log σ sinyal'), ('log_snr', 'log SNR')]:
    mm = smf.mixedlm(f'{y} ~ log_pos + log_neg + cv5x3', M, groups=M['dataset']).fit(reml=True)
    fe = smf.ols(f'{y} ~ log_pos + log_neg + cv5x3 + C(dataset)', M).fit(cov_type='cluster', cov_kwds={'groups': M['clu']})
    MODELS[y] = mm
    ci, cif = mm.conf_int(), fe.conf_int()
    for term, tl in [('log_pos', 'log(jumlah positif)'), ('log_neg', 'log(jumlah negatif)'), ('cv5x3', 'cv5x3 (vs cv5x1)')]:
        rows.append({'Respons': lab, 'Prediktor': tl,
                     'β (mixed)': round(mm.params[term], 3), '95% CI (mixed)': f'[{ci.loc[term, 0]:.2f}, {ci.loc[term, 1]:.2f}]',
                     'p (mixed)': fmt_p(mm.pvalues[term]),
                     'β (FE-klaster)': round(fe.params[term], 3), '95% CI (FE)': f'[{cif.loc[term, 0]:.2f}, {cif.loc[term, 1]:.2f}]',
                     'p (FE)': fmt_p(fe.pvalues[term])})
T3 = pd.DataFrame(rows)
save_table(T3, 'Tabel3_model_noise_sinyal_snr')
KEY['beta'] = {y: {t: float(MODELS[y].params[t]) for t in ['log_pos', 'log_neg', 'cv5x3']} for y in MODELS}
print('n =', len(M), 'unit;', M.dataset.nunique(), 'dataset (intersep acak);', M.clu.nunique(), 'klaster kondisi')
T3

# %%
# ============================================================
# GAMBAR 1 — Mekanisme: noise ditentukan mayoritas, sinyal ditentukan minoritas (cv5x1)
# ============================================================
cm = (NOISE[NOISE.scheme == 'cv5x1'].groupby(['dataset', 'ir_level'])
      .agg(pos=('pos', 'first'), neg=('neg', 'first'), noise=('sigma_noise', 'mean'), signal=('sigma_signal', 'mean'),
           snr=('snr', 'median')).reset_index())
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (x, y, xl, yl, term_resp) in zip(axes, [
        ('neg', 'noise', L('Jumlah negatif (mayoritas), data dev', 'No. of negatives (majority), dev set'), L('σ noise evaluasi (wF1)', 'Evaluation σ noise (wF1)'), ('log_noise', 'log_neg')),
        ('pos', 'signal', L('Jumlah positif (minoritas), data dev', 'No. of positives (minority), dev set'), L('σ sinyal antar-konfigurasi', 'Between-config. σ signal'), ('log_signal', 'log_pos')),
        ('neg', 'snr', L('Jumlah negatif (mayoritas), data dev', 'No. of negatives (majority), dev set'), 'SNR (median)', ('log_snr', 'log_neg'))]):
    for ds in DS_ORDER:
        g = cm[cm.dataset == ds]
        ax.scatter(g[x], g[y], s=34, marker=DS_MARK[ds], color=DS_COLOR[ds], edgecolor='white', linewidth=1.2, label=ds, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(xl); ax.set_ylabel(yl)
    b = MODELS[term_resp[0]].params[term_resp[1]]
    ax.set_title(f"{L('β model', 'model β')} = {b:+.2f}", fontsize=8, fontweight='normal', color=INK2, loc='right')
    logticks(ax, 'x', [100, 200, 400, 700] if x == 'neg' else [25, 50, 100, 200])
logticks(axes[0], 'y', [0.005, 0.01, 0.02]); logticks(axes[1], 'y', [0.003, 0.01, 0.03])
logticks(axes[2], 'y', [0.1, 0.3, 1, 3, 10])
axes[2].axhline(1, color=MUTED, lw=0.8, ls='--')
for ax, l in zip(axes, 'abc'):
    panel(ax, l)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2)
fig.tight_layout()
savefig(fig, 'Gambar1_mekanisme_noise_sinyal')
plt.show()

# %%
# ============================================================
# GAMBAR 2 — Manipulasi noise: rasio cv5x3/cv5x1 vs teori; SNR berpasangan per kondisi
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.5]))
ax = axes[0]
ax.hist(NOISE_RATIO, bins=24, color='#2a78d6', edgecolor='white', linewidth=1)
ax.axvline(KEY['noise_ratio_theory'], color=INK, lw=1.2, ls='--')
ax.axvline(KEY['noise_ratio_median'], color='#eb6834', lw=1.6)
ax.text(KEY['noise_ratio_theory'] + 0.01, ax.get_ylim()[1] * 0.92, '1/√3', color=INK, fontsize=7.5)
ax.text(KEY['noise_ratio_median'] + 0.01, ax.get_ylim()[1] * 0.78, f"median {KEY['noise_ratio_median']:.3f}", color='#eb6834', fontsize=7.5)
ax.set_xlabel('σ noise cv5x3 / σ noise cv5x1'); ax.set_ylabel(L('Jumlah pasangan', 'No. of pairs'))
panel(ax, 'a')
ax = axes[1]
snr_c = NOISE.groupby(['cond', 'scheme']).snr.median().unstack()
yy = np.arange(len(COND))[::-1]
for i, c in enumerate(COND.index):
    ax.plot([snr_c.loc[c, 'cv5x1'], snr_c.loc[c, 'cv5x3']], [yy[i]] * 2, color=AXIS, lw=1.5, zorder=1)
ax.scatter(snr_c.loc[COND.index, 'cv5x1'], yy, s=26, color='#898781', label='cv5x1', zorder=3, edgecolor='white')
ax.scatter(snr_c.loc[COND.index, 'cv5x3'], yy, s=26, color='#2a78d6', label='cv5x3', zorder=3, edgecolor='white')
ax.set_yticks(yy); ax.set_yticklabels(COND.label, fontsize=7); ax.set_xscale('log'); ax.axvline(1, color=MUTED, lw=0.8, ls='--')
logticks(ax, 'x', [0.1, 0.3, 1, 3, 10, 30])
ax.set_xlabel(L('SNR (median antar-run, skala log)', 'SNR (median over runs, log scale)')); ax.legend(loc='upper left'); ax.grid(axis='y', visible=False)
panel(ax, 'b')
fig.tight_layout()
savefig(fig, 'Gambar2_manipulasi_noise')
plt.show()

# %% [markdown]
# ## 3. RQ2a — Karakteristik fitness landscape

# %%
# ============================================================
# TABEL 4 — Metrik landscape per kondisi × skema (rata-rata antar-run)
# ============================================================
LCOLS = {'fdc': 'FDC', 'dispersion': 'Dispersion', 'top_clusters': 'Klaster top-10%', 'rf_oob_r2': 'R² surrogate (learnability)',
         'rho1': 'ρ₁ random walk', 'corr_length': 'Panjang korelasi', 'neutral_ratio': 'Rasio neutralitas', 'f_range': 'Rentang wF1'}
T4 = FLA.groupby(['cond', 'scheme'])[list(LCOLS)].mean().reset_index()
T4 = COND[['dataset', 'ir_level']].reset_index().rename(columns={'index': 'cond'}).merge(T4, on='cond').drop(columns='cond')
T4 = T4.rename(columns=LCOLS).round(3)
save_table(T4, 'Tabel4_landscape_per_kondisi')
T4

# %%
# ============================================================
# TABEL 5 — Efek penurunan noise (cv5x3 − cv5x1) pada metrik landscape, berpasangan per run
# ============================================================
rows = []
for col in ['rho1', 'rf_oob_r2', 'fdc', 'neutral_ratio', 'top_clusters', 'dispersion', 'f_range']:
    pv = FLA.pivot_table(index=['dataset', 'ir_level', 'run'], columns='scheme', values=col).dropna()
    r = paired_test(pv.cv5x3 - pv.cv5x1); r['metrik'] = LCOLS[col]; rows.append(r)
T5 = pd.DataFrame(rows)
T5['p_holm'] = holm(T5.p)
T5f = T5[['metrik', 'n', 'median', 'ci_lo', 'ci_hi', 'n_pos', 'r_rb', 'p', 'p_holm']].copy()
T5f['Δ median [95% CI]'] = [fmt_ci(a, b, c, 3) for a, b, c in zip(T5['median'], T5.ci_lo, T5.ci_hi)]
T5f['Δ > 0'] = [f'{a}/{b}' for a, b in zip(T5.n_pos, T5.n)]
T5f['r (rank-biserial)'] = T5.r_rb.round(2); T5f['p Holm'] = [fmt_p(p) for p in T5.p_holm]
T5f = T5f[['metrik', 'Δ median [95% CI]', 'Δ > 0', 'r (rank-biserial)', 'p Holm']]
save_table(T5f, 'Tabel5_efek_noise_pada_landscape')

# Korelasi SNR dengan metrik landscape (Spearman, level unit)
FS = FLA.merge(NOISE[['unit', 'snr']], on='unit')
rows = []
for col in ['rf_oob_r2', 'rho1', 'fdc', 'neutral_ratio', 'f_range']:
    rr = stats.spearmanr(np.log(FS.snr.clip(lower=1e-4)), FS[col], nan_policy='omit')
    rows.append({'metrik': LCOLS[col], 'Spearman ρ (log SNR)': round(rr.correlation, 2), 'p': rr.pvalue})
T5b = pd.DataFrame(rows); T5b['p Holm'] = [fmt_p(p) for p in holm(T5b.p)]; T5b = T5b.drop(columns='p')
save_table(T5b, 'Tabel5b_korelasi_snr_landscape')
T5f

# %%
# ============================================================
# TABEL 6 — Kepentingan hyperparameter (permutation importance surrogate RF, dinormalisasi)
# ============================================================
IMP = [c for c in FLA.columns if c.startswith('imp_')]
HP = [c.replace('imp_', '') for c in IMP]
T6 = FLA.groupby('cond')[IMP].mean()
T6.index = COND.label.values[T6.index]
T6.columns = HP
T6.index.name = L(None, 'condition')
dom = FLA[IMP].idxmax(axis=1).str.replace('imp_', '').value_counts()
KEY['dominant_hp_counts'] = dom.to_dict()
save_table(T6.round(3), 'Tabel6_kepentingan_hyperparameter', index=True)
print('Hyperparameter paling penting per unit:', dom.to_dict())

# GAMBAR 3 — learnability vs SNR, dan efek berpasangan cv5x3
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
for ds in DS_ORDER:
    g = FS[FS.dataset == ds]
    ax.scatter(g.snr.clip(lower=1e-2), g.rf_oob_r2, s=12, marker=DS_MARK[ds], color=DS_COLOR[ds], alpha=0.75, edgecolor='none', label=ds)
ax.set_xscale('log'); ax.set_xlabel(L('SNR (skala log; < 0,01 dipotong ke 0,01)', 'SNR (log scale; values < 0.01 set to 0.01)')); ax.set_ylabel(L('R² surrogate (learnability)', 'Surrogate R² (learnability)'))
logticks(ax, 'x', [0.01, 0.1, 1, 10, 100])
rr = T5b.loc[T5b.metrik == LCOLS['rf_oob_r2'], 'Spearman ρ (log SNR)'].iloc[0]
ax.text(0.03, 0.95, f'Spearman ρ = {rr:.2f}', transform=ax.transAxes, va='top', fontsize=8, color=INK)
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.3, 1.08), handletextpad=0.2, markerscale=1.6)
panel(ax, 'a')
ax = axes[1]
cols = [('rho1', 'ρ₁'), ('rf_oob_r2', 'R²'), ('fdc', 'FDC'), ('neutral_ratio', L('Neutral.', 'Neutrality'))]
data = []
for col, _ in cols:
    pv = FLA.pivot_table(index=['dataset', 'ir_level', 'run'], columns='scheme', values=col).dropna()
    data.append((pv.cv5x3 - pv.cv5x1).values)
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
for p_ in bp['boxes']:
    p_.set(facecolor='#cde2fb', edgecolor='#2a78d6')
for i, d in enumerate(data, 1):
    ax.scatter(i + RNG.uniform(-0.15, 0.15, len(d)), d, s=4, color='#2a78d6', alpha=0.35, edgecolor='none')
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(1, len(cols) + 1)); ax.set_xticklabels([c[1] for c in cols])
ax.set_ylabel(L('Δ (cv5x3 − cv5x1), berpasangan', 'Paired Δ (cv5x3 − cv5x1)')); panel(ax, 'b')
fig.tight_layout(); savefig(fig, 'Gambar3_landscape_noise'); plt.show()

# GAMBAR 4 — heatmap kepentingan hyperparameter
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seqblue', SEQ_BLUE)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(T6.values, cmap=cmap, vmin=0, vmax=max(0.7, T6.values.max()), aspect='auto')
ax.set_xticks(range(len(HP))); ax.set_xticklabels(HP, rotation=35, ha='right')
ax.set_yticks(range(len(T6))); ax.set_yticklabels(T6.index, fontsize=7.5); ax.grid(False)
for i in range(T6.shape[0]):
    for j in range(T6.shape[1]):
        v = T6.values[i, j]
        if v >= 0.10:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5, color='white' if v > 0.35 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label(L('Kepentingan relatif', 'Relative importance')); cb.outline.set_visible(False)
fig.tight_layout(); savefig(fig, 'Gambar4_kepentingan_hyperparameter'); plt.show()
T6.round(2)

# %% [markdown]
# ## 4. RQ2b — Perilaku pencarian & bias encoding

# %%
# ============================================================
# GAMBAR 5 — Eksplorasi (% diversity) & konvergensi (gap ke solusi terbaik unit), cv5x1, orientasi asli
# ============================================================
GRIDE = np.arange(1, 301)
XPL = {o: [] for o in OPT_ORDER if o != 'RS'}
GAP = {o: [] for o in OPT_ORDER}
for u in UNITS:
    if u['scheme'] != 'cv5x1':
        continue
    z = np.load(os.path.join(UNITS_DIR, u['unit'] + '.npz'))
    curves = {o: np.maximum.accumulate(z[f'opt_{o}_wf1']) for o in OPT_ORDER}
    best_any = max(c[-1] for c in curves.values())
    for o in OPT_ORDER:
        GAP[o].append(best_any - curves[o])
    for o in XPL:
        ev, dv = z[f'opt_{o}_div_evals'], z[f'opt_{o}_div']
        if len(ev) > 1 and dv.max() > 0:
            XPL[o].append(np.interp(GRIDE, ev, 100 * dv / dv.max()))
    z.close()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
for o in XPL:
    a = np.array(XPL[o]); med = np.median(a, 0)
    ax.fill_between(GRIDE, np.percentile(a, 25, 0), np.percentile(a, 75, 0), color=OPT_COLOR[o], alpha=0.15, lw=0)
    ax.plot(GRIDE, med, color=OPT_COLOR[o], label=o)
ax.set_xlabel(L('Evaluasi objektif', 'Objective evaluations')); ax.set_ylabel(L('Eksplorasi (% diversity maks.)\nmedian & IQR', 'Exploration (% of max. diversity)\nmedian & IQR')); ax.set_ylim(0, 105); panel(ax, 'a')
ax = axes[1]
for o in OPT_ORDER:
    a = np.array(GAP[o]); mean_ = a.mean(0)
    ax.plot(GRIDE, mean_, color=OPT_COLOR[o], label=o, lw=2 if o != 'RS' else 1.6, ls='-' if o != 'RS' else '--')
ax.set_xlabel(L('Evaluasi objektif', 'Objective evaluations')); ax.set_ylabel(L('Gap wF1 ke solusi terbaik unit\n(rata-rata, CV pencarian)', 'wF1 gap to best solution in unit\n(mean, search CV)'))
ax.set_ylim(bottom=0); panel(ax, 'b')
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(); savefig(fig, 'Gambar5_perilaku_pencarian'); plt.show()

# %%
# ============================================================
# TABEL 7 — Flip test (asli − dicerminkan), evaluasi ulang CV, cv5x1
# ============================================================
FL = OPT[OPT.flipped].merge(OPT_BASE, on=['unit', 'base_optimizer'], suffixes=('_f', '_a'))
FL['d_wf1'] = FL.reeval_cv_wf1_a - FL.reeval_cv_wf1_f
FL['d_mf1'] = FL.reeval_cv_mf1_a - FL.reeval_cv_mf1_f
FL['dataset'] = FL.dataset_a
rows = []
for b in ['BFO', 'HHO', 'SMA']:
    g = FL[FL.base_optimizer == b]
    for scope, gg in [('Gabungan', g)] + [(ds, g[g.dataset == ds]) for ds in DS_ORDER]:
        for met in ['d_wf1', 'd_mf1']:
            r = paired_test(gg[met]); r.update(optimizer=b, cakupan=scope, metrik='wF1' if met == 'd_wf1' else 'F1 minoritas')
            rows.append(r)
T7 = pd.DataFrame(rows)
T7['p_holm'] = np.nan
for (sc, met), idx in T7.groupby([T7.cakupan == 'Gabungan', 'metrik']).groups.items():
    T7.loc[idx, 'p_holm'] = holm(T7.loc[idx, 'p'])
if HAS_TPE:
    FLt = OPT_TPE[OPT_TPE.flipped].merge(OPT_TPE[~OPT_TPE.flipped], on=['unit'], suffixes=('_f', '_a'))
    FLt['d_wf1'] = FLt.reeval_cv_wf1_a - FLt.reeval_cv_wf1_f; FLt['d_mf1'] = FLt.reeval_cv_mf1_a - FLt.reeval_cv_mf1_f
    FLt['dataset'] = FLt.dataset_a
    _r = []
    for scope, gg in [('Gabungan', FLt)] + [(ds, FLt[FLt.dataset == ds]) for ds in DS_ORDER]:
        for met in ['d_wf1', 'd_mf1']:
            r = with_tpe_rng(paired_test, gg[met]); r.update(optimizer='TPE', cakupan=scope, metrik='wF1' if met == 'd_wf1' else 'F1 minoritas')
            _r.append(r)
    _r = pd.DataFrame(_r); _r['p_holm'] = np.nan
    for (sc, met), idx in _r.groupby([_r.cakupan == 'Gabungan', 'metrik']).groups.items():   # keluarga uji TPE tersendiri
        _r.loc[idx, 'p_holm'] = holm(_r.loc[idx, 'p'])
    T7 = pd.concat([T7, _r], ignore_index=True)
T7f = T7[['optimizer', 'cakupan', 'metrik']].copy()
T7f['Δ median [95% CI]'] = [fmt_ci(a, b, c) for a, b, c in zip(T7['median'], T7.ci_lo, T7.ci_hi)]
T7f['Δ > 0'] = [f'{a}/{b}' for a, b in zip(T7.n_pos, T7.n)]
T7f['r'] = T7.r_rb.round(2); T7f['p Holm'] = [fmt_p(p) for p in T7.p_holm]
save_table(T7f, 'Tabel7_flip_test')
KEY['flip'] = {b: T7[(T7.optimizer == b) & (T7.cakupan == 'Gabungan') & (T7.metrik == 'wF1')].iloc[0][['median', 'n_pos', 'n', 'p', 'r_rb']].to_dict() for b in (['BFO', 'HHO', 'SMA', 'TPE'] if HAS_TPE else ['BFO', 'HHO', 'SMA'])}
T7f[T7f.metrik == 'wF1']

# %%
# ============================================================
# GAMBAR 6 — Flip test & tarikan ke batas ruang
# ============================================================
FLIP_OPT = ['BFO', 'HHO', 'SMA'] + (['TPE'] if HAS_TPE else [])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
ax = axes[0]
for i, b in enumerate(FLIP_OPT):
    d = (FLt if b == 'TPE' else FL[FL.base_optimizer == b]).d_wf1.values
    ax.boxplot([d], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor=OPT_COLOR[b], lw=1.4), medianprops=dict(color=OPT_COLOR[b], lw=2),
               whiskerprops=dict(color=OPT_COLOR[b]), capprops=dict(color=OPT_COLOR[b]))
    _rng = RNG_TPE if b == 'TPE' else RNG
    ax.scatter(i + _rng.uniform(-0.18, 0.18, len(d)), d, s=6, color=OPT_COLOR[b], alpha=0.45, edgecolor='none')
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(len(FLIP_OPT))); ax.set_xticklabels(FLIP_OPT)
ax.set_ylabel(L('Δ wF1 evaluasi ulang\n(asli − dicerminkan)', 'Δ re-evaluated wF1\n(original − mirrored)')); ax.grid(axis='x', visible=False)
yt = ax.get_ylim()[1]
for i, b in enumerate(FLIP_OPT):
    pr = T7[(T7.optimizer == b) & (T7.cakupan == 'Gabungan') & (T7.metrik == 'wF1')].iloc[0]
    ax.text(i, yt, f"p {fmt_p(pr.p_holm)}\nr = {pr.r_rb:.2f}", ha='center', va='bottom', fontsize=6.8, color=INK2)
panel(ax, 'a')
ax = axes[1]
bb = pd.concat([OPT[OPT.base_optimizer.isin(['BFO', 'HHO', 'SMA']) & (OPT.scheme == 'cv5x1')], OPT_TPE], ignore_index=True)
agg = bb.groupby(['base_optimizer', 'flipped'])[['final_real_frac_at_lower', 'final_real_frac_at_upper']].mean()
x = np.arange(len(FLIP_OPT)); w = 0.2
for j, (fl_, col, lab, hatch) in enumerate([(False, 'final_real_frac_at_lower', L('asli: batas bawah', 'original: lower bound'), None),
                                             (False, 'final_real_frac_at_upper', L('asli: batas atas', 'original: upper bound'), None),
                                             (True, 'final_real_frac_at_lower', L('dicerminkan: batas bawah', 'mirrored: lower bound'), '///'),
                                             (True, 'final_real_frac_at_upper', L('dicerminkan: batas atas', 'mirrored: upper bound'), '///')]):
    vals = [agg.loc[(b, fl_), col] for b in FLIP_OPT]
    shade = INK2 if col.endswith('lower') else '#b4b2a9'   # netral: warna optimizer dipakai di panel a
    ax.bar(x + (j - 1.5) * w, vals, w * 0.9, color=shade if hatch is None else 'white', edgecolor=shade, hatch=hatch, lw=1.2, label=lab)
ax.set_xticks(x); ax.set_xticklabels(FLIP_OPT); ax.set_ylabel(L('Proporsi dimensi di batas\n(posisi akhir, ruang asli)', 'Fraction of dimensions at bound\n(final position, original space)'))
ax.set_ylim(0, 1); ax.legend(fontsize=6.8, loc='upper left'); ax.grid(axis='x', visible=False); panel(ax, 'b')
fig.tight_layout(); savefig(fig, 'Gambar6_flip_test_bias_batas'); plt.show()

# %% [markdown]
# ## 5. RQ3 — Reliabilitas optimasi

# %%
# ============================================================
# TABEL 8 — Winner's curse (CV terbaik − evaluasi ulang) per optimizer × skema + model
# ============================================================
W = OPT_BASE.merge(NOISE[['unit', 'snr']], on='unit')
W['log_snr'] = np.log(W.snr.clip(lower=1e-4)); W['cv5x3'] = (W.scheme == 'cv5x3').astype(int)
rows = []
for o in OPT_ORDER:
    for sc in ['cv5x1', 'cv5x3']:
        g = W[(W.optimizer == o) & (W.scheme == sc)].winners_curse
        lo, hi = boot_ci(g)
        rows.append({'optimizer': o, 'skema': sc, 'rata-rata [95% CI]': fmt_ci(g.mean(), lo, hi), 'median': round(g.median(), 4), 'n': len(g)})
if HAS_TPE:
    g = OPT_TPE[~OPT_TPE.flipped].winners_curse
    lo, hi = with_tpe_rng(boot_ci, g)
    rows.append({'optimizer': 'TPE', 'skema': 'cv5x1', 'rata-rata [95% CI]': fmt_ci(g.mean(), lo, hi), 'median': round(g.median(), 4), 'n': len(g)})
T8a = pd.DataFrame(rows)
pv = W.pivot_table(index=['dataset', 'ir_level', 'run', 'optimizer'], columns='scheme', values='winners_curse').dropna()
red = []
for o in OPT_ORDER:
    g = pv.xs(o, level='optimizer'); r = paired_test(g.cv5x1 - g.cv5x3)
    red.append({'optimizer': o, 'rasio rata-rata cv5x3/cv5x1': round(g.cv5x3.mean() / g.cv5x1.mean(), 2),
                'Δ median (cv5x1 − cv5x3) [95% CI]': fmt_ci(r['median'], r['ci_lo'], r['ci_hi']), 'p': r['p']})
T8b = pd.DataFrame(red); T8b['p Holm'] = [fmt_p(p) for p in holm(T8b.p)]; T8b = T8b.drop(columns='p')
mw = smf.mixedlm('winners_curse ~ C(optimizer, Treatment("RS")) + log_snr + cv5x3', W, groups=W['dataset']).fit()
T8c = pd.DataFrame({'β': mw.params, 'SE': mw.bse, 'p': mw.pvalues}).drop(index='Group Var', errors='ignore')
T8c.index = [i.replace('C(optimizer, Treatment("RS"))[T.', 'optimizer=').replace(']', '') for i in T8c.index]
T8c.index.name = 'term'
T8c['p'] = [fmt_p(p) for p in T8c.p]; T8c = T8c.round(4)
save_table(T8a, 'Tabel8a_winners_curse'); save_table(T8b, 'Tabel8b_efek_cv5x3_pada_winners_curse')
save_table(T8c, 'Tabel8c_model_winners_curse', index=True)
KEY['wc_ratio'] = dict(zip(T8b.optimizer, T8b['rasio rata-rata cv5x3/cv5x1']))
T8c

# %%
# ============================================================
# TABEL 9 — Perbandingan evaluasi ulang (wF1 & F1 minoritas) terhadap Random Search & XGBoost default
#           (cv5x1, berpasangan per unit; termasuk versi dicerminkan)
# ============================================================
X1 = OPT[OPT.scheme == 'cv5x1'].pivot_table(index='unit', columns='optimizer', values='reeval_cv_wf1')
X1m = OPT[OPT.scheme == 'cv5x1'].pivot_table(index='unit', columns='optimizer', values='reeval_cv_mf1')
DEF = BASE[(BASE.baseline == 'default') & (BASE.scheme == 'cv5x1')].set_index('unit')
rows = []
for ref, refw, refm in [('Random Search', X1['RS'], X1m['RS']), ('XGBoost default', DEF.reeval_cv_wf1.reindex(X1.index), DEF.reeval_cv_mf1.reindex(X1.index))]:
    for o in ['RS', 'BFO', 'BFO_flip', 'HHO', 'HHO_flip', 'SMA', 'SMA_flip', 'SMAc']:
        if ref == 'Random Search' and o == 'RS':
            continue
        for met, A, R in [('wF1', X1[o], refw), ('F1 minoritas', X1m[o], refm)]:
            r = paired_test(A - R); r.update(pembanding=ref, optimizer=o, metrik=met); rows.append(r)
T9 = pd.DataFrame(rows)
T9['p_holm'] = np.nan
for _, idx in T9.groupby(['pembanding', 'metrik']).groups.items():
    T9.loc[idx, 'p_holm'] = holm(T9.loc[idx, 'p'])
if HAS_TPE:
    Xt = OPT_TPE.pivot_table(index='unit', columns='optimizer', values='reeval_cv_wf1').reindex(X1.index)
    Xtm = OPT_TPE.pivot_table(index='unit', columns='optimizer', values='reeval_cv_mf1').reindex(X1.index)
    _r = []
    for ref, refw, refm in [('Random Search', X1['RS'], X1m['RS']), ('XGBoost default', DEF.reeval_cv_wf1.reindex(X1.index), DEF.reeval_cv_mf1.reindex(X1.index))]:
        for o in ['TPE', 'TPE_flip']:
            for met, A, Rr in [('wF1', Xt[o], refw), ('F1 minoritas', Xtm[o], refm)]:
                r = with_tpe_rng(paired_test, A - Rr); r.update(pembanding=ref, optimizer=o, metrik=met); _r.append(r)
    _r = pd.DataFrame(_r); _r['p_holm'] = np.nan
    for _, idx in _r.groupby(['pembanding', 'metrik']).groups.items():          # keluarga uji TPE tersendiri
        _r.loc[idx, 'p_holm'] = holm(_r.loc[idx, 'p'])
    T9 = pd.concat([T9, _r], ignore_index=True)
T9f = T9[['pembanding', 'optimizer', 'metrik']].copy()
T9f['Δ median [95% CI]'] = [fmt_ci(a, b, c) for a, b, c in zip(T9['median'], T9.ci_lo, T9.ci_hi)]
T9f['Δ > 0'] = [f'{a}/{b}' for a, b in zip(T9.n_pos, T9.n)]
T9f['r'] = T9.r_rb.round(2); T9f['p Holm'] = [fmt_p(p) for p in T9.p_holm]
save_table(T9f, 'Tabel9_perbandingan_vs_RS_default')

# selisih antar-optimizer vs winner's curse per kondisi
G = OPT_BASE.groupby(['cond', 'scheme', 'optimizer']).reeval_cv_wf1.mean().unstack()
spread = G[['RS', 'BFO', 'HHO', 'SMA']].max(axis=1) - G[['RS', 'BFO', 'HHO', 'SMA']].min(axis=1)
wc_c = OPT_BASE[OPT_BASE.optimizer.isin(['RS', 'BFO', 'HHO', 'SMA'])].groupby(['cond', 'scheme']).winners_curse.mean()
KEY['spread_median'], KEY['wc_median'] = float(spread.median()), float(wc_c.median())
KEY['spread_lt_wc'] = f'{int((spread < wc_c).sum())}/{len(wc_c)}'
gain = OPT_BASE[OPT_BASE.optimizer.isin(['RS', 'BFO', 'HHO', 'SMA'])].groupby('unit').reeval_cv_wf1.max() - \
       BASE[BASE.baseline == 'default'].set_index('unit').reeval_cv_wf1
KEY['gain_best_vs_default_by_ds'] = gain.groupby(NOISE.set_index('unit').dataset).median().round(4).to_dict()
print(f"Selisih antar-optimizer per kondisi (median) {KEY['spread_median']:.4f} vs winner's curse {KEY['wc_median']:.4f}; "
      f"selisih < WC pada {KEY['spread_lt_wc']} kondisi")
print('Keuntungan optimizer terbaik vs default (median per dataset):', KEY['gain_best_vs_default_by_ds'])
T9f[T9f.metrik == 'wF1']

# %%
# ============================================================
# GAMBAR 7 — Winner's curse: per optimizer × skema, dan sebagai fungsi SNR
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
x = np.arange(len(OPT_ORDER)); w = 0.36
for j, sc in enumerate(['cv5x1', 'cv5x3']):
    m_ = [W[(W.optimizer == o) & (W.scheme == sc)].winners_curse.mean() for o in OPT_ORDER]
    ci = [boot_ci(W[(W.optimizer == o) & (W.scheme == sc)].winners_curse) for o in OPT_ORDER]
    cols_ = [OPT_COLOR[o] for o in OPT_ORDER]
    ax.bar(x + (j - 0.5) * w, m_, w * 0.92, color=cols_ if sc == 'cv5x1' else 'white', edgecolor=cols_, lw=1.3,
           hatch=None if sc == 'cv5x1' else '///')
    ax.errorbar(x + (j - 0.5) * w, m_, yerr=np.array([[a - l, h - a] for a, (l, h) in zip(m_, ci)]).T, fmt='none', ecolor=INK2, lw=0.9, capsize=2)
ax.set_xticks(x); ax.set_xticklabels(OPT_ORDER); ax.set_ylabel(L("Winner's curse (wF1)\nCV terbaik − evaluasi ulang", "Winner's curse (wF1)\nbest CV − re-evaluation"))
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec=MUTED), plt.Rectangle((0, 0), 1, 1, fc='white', ec=MUTED, hatch='///')],
          labels=['cv5x1', 'cv5x3'], loc='upper right'); ax.grid(axis='x', visible=False); panel(ax, 'a')
ax = axes[1]
W['snr_bin'] = pd.qcut(W.log_snr, 5, labels=False)
for o in OPT_ORDER:
    g = W[W.optimizer == o].groupby('snr_bin').agg(x=('snr', 'median'), y=('winners_curse', 'mean'))
    ax.plot(g.x, g.y, marker='o', ms=4, color=OPT_COLOR[o], label=o, lw=1.8, ls='--' if o == 'RS' else '-')
ax.set_xscale('log'); ax.set_xlabel(L('SNR (median per kuintil, skala log)', 'SNR (median per quintile, log scale)')); ax.set_ylabel(L("Winner's curse (rata-rata)", "Winner's curse (mean)"))
logticks(ax, 'x', [0.3, 1, 3, 10, 30])
panel(ax, 'b')
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=5, bbox_to_anchor=(0.72, 1.07))
fig.tight_layout(); savefig(fig, 'Gambar7_winners_curse'); plt.show()

# %%
# ============================================================
# GAMBAR 8 — Forest plot: selisih evaluasi ulang wF1 terhadap Random Search, asli vs dicerminkan (cv5x1)
# ============================================================
sub = T9[(T9.pembanding == 'Random Search') & (T9.metrik == 'wF1')].set_index('optimizer')
order = (['TPE', 'TPE_flip'] if HAS_TPE else []) + ['BFO', 'BFO_flip', 'HHO', 'HHO_flip', 'SMA', 'SMA_flip', 'SMAc']
fig, ax = plt.subplots(figsize=(4.6, 2.9 + 0.3 * HAS_TPE))
for i, o in enumerate(order[::-1]):
    r = sub.loc[o]; b = o.replace('_flip', '')
    hollow = o.endswith('_flip')
    ax.errorbar(r['median'], i, xerr=[[r['median'] - r.ci_lo], [r.ci_hi - r['median']]], fmt='none', ecolor=OPT_COLOR[b], lw=1.6, capsize=2)
    ax.scatter(r['median'], i, s=42, color='white' if hollow else OPT_COLOR[b], edgecolor=OPT_COLOR[b], lw=1.6, zorder=3)
ax.axvline(0, color=MUTED, lw=0.9)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o.replace('_flip', L(' (dicerminkan)', ' (mirrored)')) for o in order[::-1]])
ax.set_xlabel(L('Δ median wF1 evaluasi ulang vs Random Search [95% CI]\n(cv5x1, n = 140 unit berpasangan)', 'Median Δ re-evaluated wF1 vs Random Search [95% CI]\n(cv5x1, n = 140 paired units)')); ax.grid(axis='y', visible=False)
ax.xaxis.set_major_locator(FixedLocator([-0.005, 0, 0.005, 0.01])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+.3f}' if v else '0'))
fig.tight_layout(); savefig(fig, 'Gambar8_forest_vs_random_search'); plt.show()

# %% [markdown]
# ## 6. Biaya komputasi & ekspor

# %%
# ============================================================
# TABEL 10 — Biaya komputasi
# ============================================================
T10 = TIME.groupby(['dataset', 'scheme']).runtime_min.agg(['mean', 'std', 'count']).round(1).reset_index()
T10.columns = ['dataset', 'skema', 'menit/unit (rata-rata)', 'SD', 'unit']
srch = TIME.filter(like='_search_sec').mean().div(60).round(2)
T10b = srch.rename(lambda s: s.replace('_search_sec', '')).rename('menit per pencarian (300 evaluasi), rata-rata').reset_index().rename(columns={'index': 'optimizer'})
save_table(T10, 'Tabel10_biaya_komputasi'); save_table(T10b, 'Tabel10b_waktu_pencarian_per_optimizer')
KEY['total_compute_hours'] = round(TIME.runtime_min.sum() / 60, 1)
print('Total waktu komputasi (jumlah semua unit):', KEY['total_compute_hours'], 'jam')

# Ekspor: satu workbook Excel + ringkasan angka kunci (JSON) untuk penulisan Results
with pd.ExcelWriter(os.path.join(OUT_DIR, L('Paper2_tabel_lengkap.xlsx', 'Paper2_all_tables.xlsx'))) as xw:
    for name, df in TABLES.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)
json.dump(KEY, open(os.path.join(OUT_DIR, L('angka_kunci.json', 'key_numbers.json')), 'w'), indent=1, default=float)
print('✅ Selesai →', OUT_DIR)

# %% [markdown]
# ## Draf keterangan gambar (untuk artikel)
#
# **Gambar 1.** Mekanisme noise evaluasi pada weighted F1. (a) σ noise evaluasi menurun dengan jumlah kelas mayoritas;
# (b) σ sinyal antar-konfigurasi meningkat dengan jumlah kelas minoritas; (c) akibatnya SNR meningkat dengan ukuran
# kedua kelas. Titik = rata-rata kondisi (cv5x1, 10 run); β = koefisien log–log dari mixed model (Tabel 3), yang
# mengontrol jumlah kelas lain, skema CV, dan dataset — sehingga pola mentah per panel tidak selalu tampak linier.
#
# **Gambar 2.** Validasi manipulasi noise. (a) Rasio σ noise cv5x3/cv5x1 pada 140 pasangan berpusat di sekitar nilai
# teoretis 1/√3. (b) SNR median per kondisi pada cv5x1 dan cv5x3.
#
# **Gambar 3.** Noise mengubah karakter landscape yang terukur. (a) Learnability landscape (R² surrogate random forest,
# OOB) terhadap SNR, per unit. (b) Selisih berpasangan (cv5x3 − cv5x1) untuk ρ₁ random walk, R² surrogate, FDC, dan
# rasio neutralitas.
#
# **Gambar 4.** Kepentingan relatif hyperparameter (permutation importance surrogate RF, dinormalisasi; rata-rata
# 20 unit per kondisi). Nilai ≥ 0,10 ditampilkan.
#
# **Gambar 5.** Perilaku pencarian (cv5x1, orientasi asli, n = 140 unit). (a) Eksplorasi sebagai persentase diversity
# populasi maksimum (median dan IQR). (b) Rata-rata selisih antara skor CV terbaik yang telah ditemukan dan skor CV
# terbaik yang dicapai metode mana pun pada unit yang sama.
#
# **Gambar 6.** Uji invariansi encoding. (a) Selisih wF1 evaluasi ulang antara orientasi asli dan ruang yang dicerminkan
# (seed sama; 140 pasangan); p Holm dan rank-biserial r. (b) Proporsi dimensi hyperparameter yang berada di batas ruang
# (posisi akhir populasi, dinyatakan dalam ruang asli).
#
# **Gambar 7.** Winner's curse (CV terbaik − rata-rata 5 evaluasi ulang dengan partisi CV baru). (a) Per optimizer dan
# skema CV (rata-rata, 95% CI bootstrap). (b) Sebagai fungsi SNR (kuintil).
#
# **Gambar 8.** Selisih median wF1 evaluasi ulang terhadap Random Search (95% CI bootstrap), dalam orientasi asli
# (titik penuh) dan ruang yang dicerminkan (titik kosong).
