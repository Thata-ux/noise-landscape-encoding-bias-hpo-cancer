"""A. Struktur ketergantungan: bootstrap berjenjang (dataset → kondisi → run), analisis tingkat kondisi, leave-one-dataset-out."""
import os, numpy as np, pandas as pd
from scipy import stats
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(REPO, 'results', 'sensitivity'); os.makedirs(OUT, exist_ok=True)
T_ = os.path.join(REPO, 'results', 'tidy')
O = pd.read_csv(f'{T_}/optimizers.csv'); F = pd.read_csv(f'{T_}/fla.csv'); B = pd.read_csv(f'{T_}/baselines.csv')
for d in (O, F, B):
    d['cond'] = d.dataset + '|' + d.ir_target.astype(str)
rng = np.random.default_rng(2026); NB = 4000


def hboot(df, val):
    """Bootstrap berjenjang untuk median selisih: resample dataset, lalu kondisi dalam dataset, lalu run dalam kondisi."""
    g = {ds: {c: gg[val].values for c, gg in d.groupby('cond')} for ds, d in df.groupby('dataset')}
    dss = list(g); out = np.empty(NB)
    for b in range(NB):
        vals = []
        for ds in rng.choice(dss, len(dss)):
            conds = list(g[ds])
            for c in rng.choice(conds, len(conds)):
                v = g[ds][c]; vals.append(rng.choice(v, len(v)))
        out[b] = np.median(np.concatenate(vals))
    return np.percentile(out, [2.5, 97.5]), float((out > 0).mean())


def summarize(df, val, label):
    d = df[val].dropna()
    w = stats.wilcoxon(d[d != 0]).pvalue
    (lo, hi), ppos = hboot(df.dropna(subset=[val]), val)
    cm = df.groupby('cond')[val].mean()
    pc = stats.wilcoxon(cm).pvalue
    lodo = []
    for ds in sorted(df.dataset.unique()):
        dd = df[df.dataset != ds][val].dropna()
        lodo.append(f"{np.median(dd):+.4f}")
    return {'Perbandingan': label, 'n unit': len(d), 'Median Δ (unit)': round(float(np.median(d)), 4),
            'p Wilcoxon (unit)': w, '95% CI bootstrap berjenjang': f'[{lo:.4f}, {hi:.4f}]', 'P(Δ>0) bootstrap': round(ppos, 3),
            'Kondisi Δ̄>0': f'{int((cm > 0).sum())}/{len(cm)}', 'p Wilcoxon (kondisi)': pc,
            'Median Δ tanpa B/C/L/T': ' '.join(lodo)}


rows = []
# flip test (cv5x1)
x = O[O.scheme == 'cv5x1']
P = x.pivot_table(index=['unit', 'dataset', 'cond'], columns='optimizer', values='reeval_cv_wf1').reset_index()
PM = x.pivot_table(index=['unit', 'dataset', 'cond'], columns='optimizer', values='reeval_cv_mf1').reset_index()
for b in ['SMA', 'HHO', 'BFO']:
    P['d'] = P[b] - P[b + '_flip']; rows.append(summarize(P, 'd', f'Flip test {b}: asli − dicerminkan (wF1)'))
for b in ['SMA', 'HHO', 'SMAc', 'SMA_flip', 'BFO']:
    P['d'] = P[b] - P['RS']; rows.append(summarize(P, 'd', f'{b} − RS (wF1)'))
for b in ['SMA']:
    PM['d'] = PM[b] - PM[b + '_flip']; rows.append(summarize(PM, 'd', f'Flip test {b}: asli − dicerminkan (F1 minoritas)'))
# efek noise pada landscape (cv5x3 − cv5x1), berpasangan per run
for col, lab in [('rho1', 'ρ₁'), ('rf_oob_r2', 'R² surrogate'), ('fdc', 'FDC'), ('neutral_ratio', 'Rasio neutralitas')]:
    pv = F.pivot_table(index=['dataset', 'cond', 'run'], columns='scheme', values=col).reset_index()
    pv['d'] = pv.cv5x3 - pv.cv5x1; rows.append(summarize(pv, 'd', f'Landscape {lab}: cv5x3 − cv5x1'))
# winner's curse: cv5x1 − cv5x3 (RS & SMA)
for o in ['RS', 'HHO']:
    pv = O[O.optimizer == o].pivot_table(index=['dataset', 'cond', 'run'], columns='scheme', values='winners_curse').reset_index()
    pv['d'] = pv.cv5x1 - pv.cv5x3; rows.append(summarize(pv, 'd', f"Winner's curse {o}: cv5x1 − cv5x3"))
# ---- TPE (baseline Bayesian Optimization; ditambahkan di akhir agar baris sebelumnya tidak berubah) ----
TPE_LONG = os.path.join(REPO, 'results', 'baseline_tpe', 'tpe_long.csv')
if os.path.exists(TPE_LONG):
    TL = pd.read_csv(TPE_LONG)
    PT = TL.pivot_table(index=['unit', 'dataset', 'cond'], columns='metode', values='reeval_cv_wf1').reset_index()
    PT['cond'] = [u.split('_cv5x1')[0] for u in PT.unit]          # kondisi = dataset + level IR
    for a_, b_, lab in [('TPE', 'TPE_flip', 'Flip test TPE: asli − dicerminkan (wF1)'), ('TPE', 'RS', 'TPE − RS (wF1)'),
                        ('TPE', 'default', 'TPE − XGBoost default (wF1)'), ('TPE', 'SMA', 'TPE − SMA (wF1)')]:
        PT['d'] = PT[a_] - PT[b_]; rows.append(summarize(PT, 'd', lab))
T = pd.DataFrame(rows)
for c in ['p Wilcoxon (unit)', 'p Wilcoxon (kondisi)']:
    T[c] = T[c].map(lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}')
T.to_csv(f'{OUT}/TableS1_dependency_aware.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20); pd.set_option('display.max_colwidth', 50)
print(T.to_string(index=False))
