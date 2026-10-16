"""B. Noise/SNR dan landscape dihitung ulang dengan F1 minoritas dan MCC sebagai fitness (dari data mentah .npz, tanpa running ulang)."""
import os, json, glob, math, itertools, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
U = os.path.join(REPO, 'results', 'units'); OUT = os.path.join(REPO, 'results', 'sensitivity'); os.makedirs(OUT, exist_ok=True)


def noise_stats(F):
    nrep = F.shape[1]
    vw = float(F.var(axis=1, ddof=1).mean()); vm = float(F.mean(axis=1).var(ddof=1))
    vs = max(vm - vw / nrep, 0.0)
    taus = [stats.kendalltau(F[:, a], F[:, b])[0] for a, b in itertools.combinations(range(nrep), 2)]
    return dict(sigma_noise=math.sqrt(vw), sigma_signal=math.sqrt(vs), snr=vs / vw if vw > 0 else np.nan,
                icc=vs / (vs + vw) if vs + vw > 0 else np.nan, tau=float(np.nanmean(taus)))


def fla_stats(X, fs, walk, sigma_noise, seed):
    best = X[np.argmin(fs)]; dist = np.linalg.norm(X - best, axis=1)
    fdc = float(stats.pearsonr(fs, dist)[0]) if np.std(fs) > 0 else np.nan
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, oob_score=True, random_state=seed).fit(X, fs)
    rho1 = float(pd.Series(walk).autocorr(1)) if np.std(walk) > 0 else np.nan
    steps = np.abs(np.diff(walk))
    return dict(fdc=fdc, rf_oob_r2=float(rf.oob_score_), rho1=rho1,
                neutral_ratio=float(np.mean(steps < sigma_noise)) if sigma_noise > 0 else float(np.mean(steps == 0)))


def one(js):
    u = json.load(open(js)); z = np.load(js[:-5] + '.npz'); out = []
    for m in ('wf1', 'mf1', 'mcc'):
        ns = noise_stats(z[f'noise_{m}'].astype(float))
        fl = fla_stats(z['fla_sample_X'].astype(float), -z[f'fla_sample_{m}'].astype(float), -z[f'fla_walk_{m}'].astype(float),
                       ns['sigma_noise'], u['seed'])
        out.append(dict(unit=u['unit'], dataset=u['dataset'], ir=('asli' if u['ir_target'] is None else str(u['ir_target'])),
                        scheme=u['scheme'], run=u['run'], pos=u['dev_pos'], neg=u['n_dev'] - u['dev_pos'], metric=m, **ns, **fl))
    return out


rows = Parallel(n_jobs=2)(delayed(one)(f) for f in sorted(glob.glob(f'{U}/*.json')))
D = pd.DataFrame([r for rr in rows for r in rr])
D.to_csv(f'{OUT}/unit_metrics_wf1_mf1_mcc.csv', index=False)
print('selesai', D.shape)


# ---- analisis ----
import numpy as np, pandas as pd, warnings
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
D = pd.read_csv(f'{OUT}/unit_metrics_wf1_mf1_mcc.csv')
N = pd.read_csv(os.path.join(REPO, 'results', 'tidy', 'noise.csv')); Fo = pd.read_csv(os.path.join(REPO, 'results', 'tidy', 'fla.csv'))
w = D[D.metric == 'wf1'].merge(N[['unit', 'sigma_noise', 'snr']], on='unit', suffixes=('', '_orig')).merge(Fo[['unit', 'rf_oob_r2', 'rho1']], on='unit', suffixes=('', '_orig'))
print('cek wF1 vs asli: max|Δσ|', np.abs(w.sigma_noise - w.sigma_noise_orig).max(), ' max|ΔR²|', np.abs(w.rf_oob_r2 - w.rf_oob_r2_orig).max(),
      ' max|Δρ1|', np.abs(w.rho1 - w.rho1_orig).max())
LAB = {'wf1': 'Weighted F1', 'mf1': 'F1 minoritas', 'mcc': 'MCC'}
rows_m, rows_l, rows_s = [], [], []
for m in ['wf1', 'mf1', 'mcc']:
    d = D[D.metric == m].copy()
    d['log_noise'] = np.log(d.sigma_noise); d['log_signal'] = np.log(d.sigma_signal.clip(lower=1e-6)); d['log_snr'] = np.log(d.snr.clip(lower=1e-4))
    d['log_pos'] = np.log(d.pos); d['log_neg'] = np.log(d.neg); d['cv5x3'] = (d.scheme == 'cv5x3').astype(int)
    r = {'Fitness': LAB[m]}
    for y, yl in [('log_noise', 'σnoise'), ('log_signal', 'σsinyal'), ('log_snr', 'SNR')]:
        mm = smf.mixedlm(f'{y} ~ log_pos + log_neg + cv5x3', d, groups=d['dataset']).fit(reml=True)
        for t, tl in [('log_pos', 'log n positif'), ('log_neg', 'log n negatif'), ('cv5x3', 'cv5x3')]:
            r[f'{yl}: {tl}'] = f"{mm.params[t]:+.2f}{'*' if mm.pvalues[t] < 0.05 else ''}"
    pv = d.pivot_table(index=['dataset', 'ir', 'run'], columns='scheme', values='sigma_noise').dropna()
    r['rasio σnoise cv5x3/cv5x1 (median)'] = round(float((pv.cv5x3 / pv.cv5x1).median()), 3)
    r['SNR cv5x1 (median, rentang kondisi)'] = (lambda s: f"{s.median():.2f} [{s.min():.2f}–{s.max():.2f}]")(
        d[d.scheme == 'cv5x1'].groupby(['dataset', 'ir']).snr.median())
    rows_m.append(r)
    for col, cl in [('rho1', 'ρ₁ random walk'), ('rf_oob_r2', 'R² surrogate'), ('fdc', 'FDC'), ('neutral_ratio', 'Rasio neutralitas')]:
        pv = d.pivot_table(index=['dataset', 'ir', 'run'], columns='scheme', values=col).dropna()
        dd = pv.cv5x3 - pv.cv5x1; nz = dd[dd != 0]
        rk = stats.rankdata(np.abs(nz)); rrb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
        rho = stats.spearmanr(d.log_snr, d[col], nan_policy='omit').correlation
        rows_l.append({'Fitness': LAB[m], 'Metrik': cl, 'Median Δ (cv5x3 − cv5x1)': round(float(dd.median()), 3),
                       'Δ > 0': f'{int((dd > 0).sum())}/{len(dd)}', 'r': round(float(rrb), 2),
                       'p Wilcoxon': stats.wilcoxon(nz).pvalue, 'Spearman ρ (log SNR)': round(float(rho), 2)})
# kesesuaian SNR antar-metrik (tingkat kondisi × skema)
c = D.groupby(['metric', 'dataset', 'ir', 'scheme']).snr.median().unstack('metric')
for a, b in [('wf1', 'mf1'), ('wf1', 'mcc'), ('mf1', 'mcc')]:
    rows_s.append({'Pasangan': f'{LAB[a]} vs {LAB[b]}', 'Spearman ρ SNR (28 kondisi×skema)': round(stats.spearmanr(c[a], c[b]).correlation, 2)})
TM, TL, TS = pd.DataFrame(rows_m), pd.DataFrame(rows_l), pd.DataFrame(rows_s)
TL['p Wilcoxon'] = TL['p Wilcoxon'].map(lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}')
TM.to_csv(f'{OUT}/TableS2a_noise_model_by_metric.csv', index=False); TL.to_csv(f'{OUT}/TableS2b_landscape_by_metric.csv', index=False); TS.to_csv(f'{OUT}/TableS2c_snr_agreement.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(TM.T.to_string()); print(TL.to_string(index=False)); print(TS.to_string(index=False))
