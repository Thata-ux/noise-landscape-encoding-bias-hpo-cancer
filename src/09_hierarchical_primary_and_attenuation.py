"""Hierarchical primary inference (Tables 6, 7, 10), attenuation analysis of landscape metrics, and
spatial correlation of partition noise. Reads stored results only (no model is trained).

Outputs
  results/hierarchical/Table6_primary.csv, Table7_primary.csv, Table10_primary.csv
  results/attenuation/attenuation_per_pair.csv, attenuation_summary.csv, noise_spatial_correlation.csv
  results/figures/Fig4_landscape_vs_noise.(png|pdf), Fig7_flip_test_boundary_bias.(png|pdf), Fig9_forest_vs_random_search.(png|pdf)
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RES = os.path.join(REPO, 'results')
OUT_H = os.path.join(RES, 'hierarchical'); OUT_A = os.path.join(RES, 'attenuation'); FIG = os.path.join(RES, 'figures')
for d in (OUT_H, OUT_A, FIG):
    os.makedirs(d, exist_ok=True)
T_ = os.path.join(RES, 'tidy')
O = pd.read_csv(f'{T_}/optimizers.csv'); F = pd.read_csv(f'{T_}/fla.csv'); BL = pd.read_csv(f'{T_}/baselines.csv')
NZ = pd.read_csv(f'{T_}/noise.csv')
for d in (O, F, BL, NZ):
    d['cond'] = d.dataset + '|' + d.ir_target.astype(str)
NB, SEED = 4000, 2026


# ------------------------------------------------------------------ statistics
def hboot(df, val, seed=SEED):
    """Hierarchical bootstrap of the median: datasets, then conditions within dataset, then runs within condition."""
    rng = np.random.default_rng(seed)
    g = {ds: {c: gg[val].values for c, gg in d.groupby('cond')} for ds, d in df.groupby('dataset')}
    dss = list(g); out = np.empty(NB)
    for b in range(NB):
        vals = []
        for ds in rng.choice(dss, len(dss)):
            conds = list(g[ds])
            for c in rng.choice(conds, len(conds)):
                v = g[ds][c]; vals.append(rng.choice(v, len(v)))
        out[b] = np.median(np.concatenate(vals))
    return np.percentile(out, [2.5, 97.5])


def summarize(df, val, label, **extra):
    d = df[[val, 'dataset', 'cond']].dropna()
    lo, hi = hboot(d, val)
    cm = d.groupby('cond')[val].mean()
    pc = stats.wilcoxon(cm).pvalue if (cm != 0).any() else 1.0
    nz = d[val][d[val] != 0]
    pu = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
    # unit-level complement (Table S1): percentile bootstrap CI of the median, rank-biserial r, leave-one-dataset-out medians
    rng = np.random.default_rng(SEED); v = d[val].values
    bs = np.median(rng.choice(v, (2000, len(v))), axis=1); ulo, uhi = np.percentile(bs, [2.5, 97.5])
    rk = stats.rankdata(np.abs(nz)); rpos = rk[nz.values > 0].sum(); rneg = rk[nz.values < 0].sum()
    r_rb = (rpos - rneg) / (rpos + rneg) if len(nz) else 0.0
    lodo = {ds: float(np.median(d.loc[d.dataset != ds, val])) for ds in sorted(d.dataset.unique())}
    return dict(label=label, n_units=len(d), median=float(np.median(d[val])), ci_lo=lo, ci_hi=hi,
                cond_pos=int((cm > 0).sum()), n_cond=len(cm), p_cond=pc, units_pos=int((d[val] > 0).sum()), p_unit=pu,
                unit_ci_lo=ulo, unit_ci_hi=uhi, r_rb=r_rb, **{f'lodo_without_{k}': x for k, x in lodo.items()}, **extra)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def add_holm(rows, families):
    T = pd.DataFrame(rows)
    T['p_holm_cond'] = np.nan; T['p_holm_unit'] = np.nan
    for fam in families:
        idx = T.index[T.family == fam]
        T.loc[idx, 'p_holm_cond'] = holm(T.loc[idx, 'p_cond']); T.loc[idx, 'p_holm_unit'] = holm(T.loc[idx, 'p_unit'])
    return T


# ------------------------------------------------------------------ Table 7 (flip test)
x = O[O.scheme == 'cv5x1']
P = x.pivot_table(index=['unit', 'dataset', 'cond'], columns='optimizer', values='reeval_cv_wf1').reset_index()
PM = x.pivot_table(index=['unit', 'dataset', 'cond'], columns='optimizer', values='reeval_cv_mf1').reset_index()
TL = pd.read_csv(os.path.join(RES, 'baseline_tpe', 'tpe_long.csv'))
PT = TL.pivot_table(index=['unit', 'dataset', 'cond'], columns='metode', values='reeval_cv_wf1').reset_index()
PTM = TL.pivot_table(index=['unit', 'dataset', 'cond'], columns='metode', values='reeval_cv_mf1').reset_index()
rows = []
for b in ['BFO', 'HHO', 'SMA']:
    P['d'] = P[b] - P[b + '_flip']; rows.append(summarize(P, 'd', b, family='flip_wf1', metric='wF1'))
    PM['d'] = PM[b] - PM[b + '_flip']; rows.append(summarize(PM, 'd', b, family='flip_mf1', metric='minority F1'))
PT['d'] = PT.TPE - PT.TPE_flip; rows.append(summarize(PT, 'd', 'TPE', family='flip_wf1_tpe', metric='wF1'))
PTM['d'] = PTM.TPE - PTM.TPE_flip; rows.append(summarize(PTM, 'd', 'TPE', family='flip_mf1_tpe', metric='minority F1'))
T7 = add_holm(rows, ['flip_wf1', 'flip_mf1', 'flip_wf1_tpe', 'flip_mf1_tpe'])
bb = O[O.base_optimizer.isin(['BFO', 'HHO', 'SMA']) & (O.scheme == 'cv5x1')]
bnd = bb.groupby(['base_optimizer', 'flipped'])[['final_real_frac_at_lower', 'final_real_frac_at_upper']].mean()
tb = TL[TL.metode.isin(['TPE', 'TPE_flip'])].groupby('metode')[['batas_bawah', 'batas_atas']].mean()
T7.to_csv(f'{OUT_H}/Table7_primary.csv', index=False)

# ------------------------------------------------------------------ Table 10 (vs RS and default)
dflt = BL[(BL.baseline == 'default') & (BL.scheme == 'cv5x1')].set_index('unit').reeval_cv_wf1
P['default'] = P.unit.map(dflt)
rows = []
for ref, fam in [('RS', 'rs'), ('default', 'default')]:
    for m in (['BFO', 'BFO_flip', 'HHO', 'HHO_flip', 'SMA', 'SMA_flip', 'SMAc'] + (['RS'] if ref == 'default' else [])):
        P['d'] = P[m] - P[ref]; rows.append(summarize(P, 'd', m, family=fam, reference=ref))
    for m in ['TPE', 'TPE_flip']:
        PT['d'] = PT[m] - PT[ref]; rows.append(summarize(PT, 'd', m, family=fam + '_tpe', reference=ref))
T10 = add_holm(rows, ['rs', 'rs_tpe', 'default', 'default_tpe'])
T10.to_csv(f'{OUT_H}/Table10_primary.csv', index=False)

# ------------------------------------------------------------------ Table 9 complement (winner's curse cv5x1 − cv5x3)
rows = []
for o in ['RS', 'BFO', 'HHO', 'SMA', 'SMAc']:
    pv = O[O.optimizer == o].pivot_table(index=['dataset', 'cond', 'run'], columns='scheme', values='winners_curse').reset_index()
    pv['d'] = pv.cv5x1 - pv.cv5x3; rows.append(summarize(pv, 'd', o, family='wc'))
T9 = add_holm(rows, ['wc']); T9.to_csv(f'{OUT_H}/Table9_primary.csv', index=False)

# ------------------------------------------------------------------ Table 6 (noise effect on landscape)
NE = pd.read_csv(os.path.join(RES, 'reanalysis', '2_neutrality_per_unit.csv'))
F = F.merge(NE[['unit', 'neutral_fixed_cv5x1']], on='unit', how='left')
rows = []
for col, lab in [('rho1', 'rho1'), ('rf_oob_r2', 'R2'), ('fdc', 'FDC'), ('neutral_fixed_cv5x1', 'neutrality_fixed'),
                 ('neutral_ratio', 'neutrality_own'), ('top_clusters', 'clusters'), ('dispersion', 'dispersion'), ('f_range', 'range')]:
    pv = F.pivot_table(index=['dataset', 'cond', 'run'], columns='scheme', values=col).reset_index()
    pv['d'] = pv.cv5x3 - pv.cv5x1
    rows.append(summarize(pv, 'd', lab, family='landscape' if lab != 'neutrality_own' else 'supp'))
T6 = add_holm(rows, ['landscape', 'supp'])

# ------------------------------------------------------------------ attenuation analysis
R1 = pd.read_csv(os.path.join(RES, 'reanalysis', '1a_snr_per_unit.csv'))
var = []
for u in F.unit:
    z = np.load(os.path.join(RES, 'units', f'{u}.npz'))
    var.append(dict(unit=u, V=z['fla_sample_wf1'].astype(float).var(ddof=1), Vw=z['fla_walk_wf1'].astype(float).var(ddof=1)))
D = F.merge(pd.DataFrame(var), on='unit').merge(R1[['unit', 's2_resid']], on='unit')
# condition-level reliability (median over 10 runs) of a single evaluation under the fixed search partition:
# rel = 1 − σ²_interaction / Var(observed fitness); the partition main effect shifts all points equally and cancels.
C = D.groupby(['cond', 'scheme'])[['V', 'Vw', 's2_resid']].median().reset_index()
C['rel'] = (1 - C.s2_resid / C.V).clip(0.05, 1); C['rel_walk'] = (1 - C.s2_resid / C.Vw).clip(0.05, 1)
Cr = C.pivot(index='cond', columns='scheme', values=['rel', 'rel_walk']); Cr.columns = ['_'.join(c) for c in Cr.columns]
PA = D.pivot_table(index=['dataset', 'cond', 'run'], columns='scheme', values=['rf_oob_r2', 'fdc', 'rho1']).reset_index()
PA.columns = ['_'.join(c).strip('_') for c in PA.columns]
PA = PA.merge(Cr.reset_index(), on='cond')
q, qw = PA.rel_cv5x3 / PA.rel_cv5x1, PA.rel_walk_cv5x3 / PA.rel_walk_cv5x1
# independent-noise predictions: R²_obs ≈ R²_true·rel ; FDC_obs ≈ FDC_true·√rel ; ρ1_obs ≈ ρ1_true·rel_walk
PA['pred_rf_oob_r2'] = PA.rf_oob_r2_cv5x1 * q
PA['pred_fdc'] = PA.fdc_cv5x1 * np.sqrt(q)
PA['pred_rho1'] = PA.rho1_cv5x1 * qw
att = []
for m, lab in [('rho1', 'rho1'), ('rf_oob_r2', 'R2'), ('fdc', 'FDC')]:
    PA[f'obs_d_{m}'] = PA[f'{m}_cv5x3'] - PA[f'{m}_cv5x1']; PA[f'pred_d_{m}'] = PA[f'pred_{m}'] - PA[f'{m}_cv5x1']
    PA['gap'] = PA[f'obs_d_{m}'] - PA[f'pred_d_{m}']
    cm = PA.groupby('cond')[[f'obs_d_{m}', f'pred_d_{m}']].mean()
    s = summarize(PA, 'gap', lab)
    att.append(dict(metric=lab, obs_median=float(PA[f'obs_d_{m}'].median()), pred_median=float(PA[f'pred_d_{m}'].median()),
                    ratio=float(PA[f'obs_d_{m}'].median() / PA[f'pred_d_{m}'].median()),
                    gap_median=s['median'], gap_ci_lo=s['ci_lo'], gap_ci_hi=s['ci_hi'], cond_obs_below_pred=int((cm.iloc[:, 0] < cm.iloc[:, 1]).sum()),
                    p_cond=s['p_cond'], spearman_cond=float(stats.spearmanr(cm.iloc[:, 0], cm.iloc[:, 1])[0])))
ATT = pd.DataFrame(att); ATT['p_holm_cond'] = holm(ATT.p_cond)
PA.to_csv(f'{OUT_A}/attenuation_per_pair.csv', index=False); ATT.to_csv(f'{OUT_A}/attenuation_summary.csv', index=False)
C.to_csv(f'{OUT_A}/reliability_per_condition.csv', index=False)
pm = {'rho1': 'rho1', 'R2': 'R2', 'FDC': 'FDC'}
T6['pred_median'] = T6.label.map(dict(zip(ATT.metric, ATT.pred_median)))
T6.to_csv(f'{OUT_H}/Table6_primary.csv', index=False)

# ------------------------------------------------------------------ spatial correlation of partition noise
rows, per_unit = [], []
for f in sorted(glob.glob(os.path.join(RES, 'units', '*.npz'))):
    u = os.path.basename(f)[:-4]
    z = np.load(f); Y = z['noise_wf1'].astype(float); X = z['noise_configs'].astype(float)
    Rr = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()      # config × partition interaction
    ok = Rr.std(1) > 1e-9
    Cc = np.corrcoef(Rr[ok]); Xo = X[ok]
    dist = np.sqrt(((Xo[:, None, :] - Xo[None, :, :]) ** 2).sum(-1)); iu = np.triu_indices(len(Xo), 1)
    r, dd = Cc[iu], dist[iu]
    rows.append(pd.DataFrame(dict(unit=u, d=dd, r=r)))
    near, far = r[dd < 0.8], r[dd > 1.2]
    per_unit.append(dict(unit=u, rho_dist=stats.spearmanr(dd, r)[0], near_minus_far=(near.mean() - far.mean()) if len(near) and len(far) else np.nan))
SP = pd.concat(rows); PU = pd.DataFrame(per_unit)
PU = PU.merge(NZ[['unit', 'dataset', 'cond', 'scheme']], on='unit')
SP['bin'] = pd.cut(SP.d, [0, 0.6, 0.8, 1.0, 1.2, 1.4, 3.0])
SP = SP.merge(NZ[['unit', 'scheme']], on='unit')
bins = SP.groupby(['scheme', 'bin'], observed=True).r.agg(['mean', 'count']).reset_index()
s_rho = summarize(PU, 'rho_dist', 'spearman_distance_vs_residual_corr')
s_nf = summarize(PU, 'near_minus_far', 'near(<0.8)_minus_far(>1.2)')
bins.to_csv(f'{OUT_A}/noise_spatial_correlation_bins.csv', index=False)
pd.DataFrame([s_rho, s_nf]).to_csv(f'{OUT_A}/noise_spatial_correlation_summary.csv', index=False)
PU.to_csv(f'{OUT_A}/noise_spatial_correlation_per_unit.csv', index=False)

# ------------------------------------------------------------------ figures
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
OPT_COLOR = {'RS': '#898781', 'BFO': '#2a78d6', 'HHO': '#eb6834', 'SMA': '#1baf7a', 'SMAc': '#eda100', 'TPE': '#7b3fb0'}
DS_ORDER = ['Lung', 'Thyroid', 'Breast', 'Cervical']
DS_COLOR = {'Lung': '#6250d6', 'Thyroid': '#008300', 'Breast': '#e87ba4', 'Cervical': '#52514e'}
DS_MARK = {'Lung': 'o', 'Thyroid': 's', 'Breast': '^', 'Cervical': 'D'}
plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5, 'axes.labelcolor': INK2,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 2, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
RNG = np.random.default_rng(SEED)


def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'= {p:.3f}'


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK, va='bottom')


# Fig 4: (a) learnability vs SNR; (b) paired Δ, neutrality with a fixed threshold
FS = F.merge(NZ[['unit', 'snr']], on='unit')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
for ds in DS_ORDER:
    g = FS[FS.dataset == ds]
    ax.scatter(g.snr.clip(lower=1e-2), g.rf_oob_r2, s=12, marker=DS_MARK[ds], color=DS_COLOR[ds], alpha=0.75, edgecolor='none', label=ds)
ax.set_xscale('log'); ax.set_xlabel('SNR (log scale; values < 0.01 set to 0.01)'); ax.set_ylabel('Surrogate R² (learnability)')
a_ = ax.xaxis; a_.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100])); a_.set_minor_locator(NullLocator())
a_.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); a_.set_minor_formatter(NullFormatter())
rr = stats.spearmanr(np.log(FS.snr.clip(lower=1e-2)), FS.rf_oob_r2)[0]
ax.text(0.03, 0.95, f'Spearman ρ = {rr:.2f}', transform=ax.transAxes, va='top', fontsize=8, color=INK)
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.3, 1.08), handletextpad=0.2, markerscale=1.6)
panel(ax, 'a')
ax = axes[1]
cols = [('rho1', 'ρ₁'), ('rf_oob_r2', 'R²'), ('fdc', 'FDC'), ('neutral_fixed_cv5x1', 'Neutrality\n(fixed thr.)')]
data = []
for col, _ in cols:
    pv = F.pivot_table(index=['dataset', 'cond', 'run'], columns='scheme', values=col).dropna()
    data.append((pv.cv5x3 - pv.cv5x1).values)
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
for p_ in bp['boxes']:
    p_.set(facecolor='#cde2fb', edgecolor='#2a78d6')
for i, d in enumerate(data, 1):
    ax.scatter(i + RNG.uniform(-0.15, 0.15, len(d)), d, s=4, color='#2a78d6', alpha=0.35, edgecolor='none')
for i, lab in enumerate(['rho1', 'R2', 'FDC'], 1):
    pr = ATT.set_index('metric').loc[lab, 'pred_median']
    ax.plot([i - 0.3, i + 0.3], [pr, pr], color='#eb6834', lw=1.6, ls='--', label='predicted Δ, independent noise' if i == 1 else None)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(1, len(cols) + 1)); ax.set_xticklabels([c[1] for c in cols])
ax.set_ylabel('Paired Δ (cv5x3 − cv5x1)'); ax.legend(loc='lower left', fontsize=6.8); panel(ax, 'b')
fig.tight_layout(); save(fig, 'Fig4_landscape_vs_noise')

# Fig 7: flip test with condition-level Holm p and hierarchical CI
FLIP = ['BFO', 'HHO', 'SMA', 'TPE']
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
ax = axes[0]
for i, b in enumerate(FLIP):
    d = (PT.TPE - PT.TPE_flip).values if b == 'TPE' else (P[b] - P[b + '_flip']).values
    ax.boxplot([d], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor=OPT_COLOR[b], lw=1.4), medianprops=dict(color=OPT_COLOR[b], lw=2),
               whiskerprops=dict(color=OPT_COLOR[b]), capprops=dict(color=OPT_COLOR[b]))
    ax.scatter(i + RNG.uniform(-0.18, 0.18, len(d)), d, s=6, color=OPT_COLOR[b], alpha=0.45, edgecolor='none')
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(len(FLIP))); ax.set_xticklabels(FLIP)
ax.set_ylabel('Δ re-evaluated wF1\n(original − mirrored)'); ax.grid(axis='x', visible=False)
yt = ax.get_ylim()[1]
for i, b in enumerate(FLIP):
    pr = T7[(T7.label == b) & (T7.metric == 'wF1')].iloc[0]
    ax.text(i, yt, f"{pr.cond_pos}/{pr.n_cond} cond.\np {fmt_p(pr.p_holm_cond)}", ha='center', va='bottom', fontsize=6.6, color=INK2)
panel(ax, 'a')
ax = axes[1]
x_ = np.arange(len(FLIP)); w = 0.2
for j, (fl_, col, tcol, lab, hatch) in enumerate([(False, 'final_real_frac_at_lower', 'batas_bawah', 'original: lower bound', None),
                                                 (False, 'final_real_frac_at_upper', 'batas_atas', 'original: upper bound', None),
                                                 (True, 'final_real_frac_at_lower', 'batas_bawah', 'mirrored: lower bound', '///'),
                                                 (True, 'final_real_frac_at_upper', 'batas_atas', 'mirrored: upper bound', '///')]):
    vals = [tb.loc['TPE_flip' if fl_ else 'TPE', tcol] if b == 'TPE' else bnd.loc[(b, fl_), col] for b in FLIP]
    shade = INK2 if col.endswith('lower') else '#b4b2a9'
    ax.bar(x_ + (j - 1.5) * w, vals, w * 0.9, color=shade if hatch is None else 'white', edgecolor=shade, hatch=hatch, lw=1.2, label=lab)
ax.set_xticks(x_); ax.set_xticklabels(FLIP); ax.set_ylabel('Fraction of dimensions at bound\n(final position, original space)')
ax.set_ylim(0, 1); ax.legend(fontsize=6.8, loc='upper left'); ax.grid(axis='x', visible=False); panel(ax, 'b')
fig.tight_layout(); save(fig, 'Fig7_flip_test_boundary_bias')

# Fig 9: forest plot vs RS with hierarchical 95% CI
sub = T10[T10.reference == 'RS'].set_index('label')
order = ['TPE', 'TPE_flip', 'BFO', 'BFO_flip', 'HHO', 'HHO_flip', 'SMA', 'SMA_flip', 'SMAc']
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for i, o in enumerate(order[::-1]):
    r = sub.loc[o]; b = o.replace('_flip', ''); hollow = o.endswith('_flip')
    ax.errorbar(r['median'], i, xerr=[[r['median'] - r.ci_lo], [r.ci_hi - r['median']]], fmt='none', ecolor=OPT_COLOR[b], lw=1.6, capsize=2)
    ax.scatter(r['median'], i, s=42, color='white' if hollow else OPT_COLOR[b], edgecolor=OPT_COLOR[b], lw=1.6, zorder=3)
    ax.text(0.0225, i, f"{r.cond_pos}/{r.n_cond}", va='center', ha='right', fontsize=6.8, color=INK2)
ax.axvline(0, color=MUTED, lw=0.9)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o.replace('_flip', ' (mirrored)') for o in order[::-1]])
ax.set_xlim(-0.011, 0.0230)
ax.set_xlabel('Median Δ re-evaluated wF1 vs Random Search\n[95% hierarchical bootstrap CI] (cv5x1)'); ax.grid(axis='y', visible=False)
ax.xaxis.set_major_locator(FixedLocator([-0.01, -0.005, 0, 0.005, 0.01, 0.015])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+.3f}' if v else '0'))
ax.text(0.0225, len(order) - 0.35, 'cond. Δ̄ > 0', ha='right', va='bottom', fontsize=6.8, color=INK2)
fig.tight_layout(); save(fig, 'Fig9_forest_vs_random_search')

pd.set_option('display.width', 250)
for name, T in [('Table 6', T6), ('Table 7', T7), ('Table 10', T10)]:
    print('\n' + name); print(T.round(4).to_string(index=False))
print('\nAttenuation'); print(ATT.round(4).to_string(index=False))
print('\nSpatial correlation of partition noise'); print(bins.round(3).to_string(index=False))
print(pd.DataFrame([s_rho, s_nf]).round(4).to_string(index=False))
