"""C. Uji bias struktural tanpa informasi fitness: optimizer dijalankan pada fitness acak murni (i.i.d. U(0,1)) dan pada
fitness konstan. Algoritma tanpa bias posisi seharusnya menyebar seragam di [0,1]^9 (rata-rata centroid ≈ 0,5; proporsi
dimensi di batas ≈ 0). Kode optimizer identik dengan eksperimen utama (v5.2) + implementasi mealpy 3.0.1."""
import os, math, importlib, numpy as np, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_SRC = open(os.path.join(REPO, 'src', '01_experiment_v5_2.py'), encoding='utf-8').read()
# ambil kode optimizer dari skrip eksperimen utama (identik dengan yang dipakai pada 280 unit)
_seg = lambda a, b: _SRC[_SRC.index(a):_SRC.index(b)]

D = 9
exec(_seg('class BudgetExhausted', "print('Evaluator siap"))
exec(_seg('class Tracker:', 'def _opt_params():'))

OUT = os.path.join(REPO, 'results', 'sensitivity'); os.makedirs(OUT, exist_ok=True)
BUDGET, POP, REPS = 300, 20, 200


class FakeEval:
    def __init__(self, kind, rng):
        self.kind, self.rng = kind, rng

    def __call__(self, x):
        f = float(self.rng.random()) if self.kind == 'acak' else 0.5
        return dict(f=f, wf1=-f, mf1=0.0, mcc=0.0)


OWN = {'BFO': run_bfo, 'HHO': run_hho, 'SMA': run_sma, 'SMAc': lambda o, r, n, t: run_sma(o, r, n, t, centered=True)}
rows = []
for kind in ('acak', 'konstan'):
    for name, fn in OWN.items():
        for rep in range(REPS):
            rng = np.random.default_rng([rep, 7]); obj = Objective(FakeEval(kind, np.random.default_rng([rep, 99])), BUDGET); tr = Tracker(obj)
            try:
                fn(obj, rng, POP, tr)
            except BudgetExhausted:
                pass
            c = tr.last_pop.mean(axis=0)
            rows.append(dict(fitness=kind, sumber='kode penelitian (v5.2)', metode=name, rep=rep, centroid=c.mean(),
                             bawah=np.mean(c < 0.02), atas=np.mean(c > 0.98), pusat=np.mean(np.abs(c - 0.5) < 0.02)))
# mealpy
from mealpy import FloatVar
for kind in ('acak', 'konstan'):
    for name, mod, cls, ep, dom in [('SMA', 'bio_based.SMA', 'OriginalSMA', 14, 'unit'), ('SMA [−1,1]', 'bio_based.SMA', 'OriginalSMA', 14, 'center'),
                                    ('HHO', 'swarm_based.HHO', 'OriginalHHO', 7, 'unit')]:
        Model = getattr(importlib.import_module(f'mealpy.{mod}'), cls)
        for rep in range(REPS):
            frng = np.random.default_rng([rep, 99]); obj = Objective(FakeEval(kind, frng), BUDGET)
            lb, ub = ((-1.0,) * D, (1.0,) * D) if dom == 'center' else ((0.0,) * D, (1.0,) * D)
            m = Model(epoch=ep, pop_size=POP, **({'p_t': 0.03} if 'SMA' in name else {}))
            try:
                m.solve(dict(bounds=FloatVar(lb=lb, ub=ub), minmax='min', log_to=None, obj_func=lambda u: obj(np.zeros(D))), mode='single', seed=rep)
            except BudgetExhausted:
                pass
            pu = np.array([a.solution for a in m.pop]); px = (pu + 1) / 2 if dom == 'center' else pu
            c = px.mean(axis=0)
            rows.append(dict(fitness=kind, sumber='mealpy 3.0.1', metode=name, rep=rep, centroid=c.mean(),
                             bawah=np.mean(c < 0.02), atas=np.mean(c > 0.98), pusat=np.mean(np.abs(c - 0.5) < 0.02)))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/structural_bias_raw.csv', index=False)
T = (R.groupby(['fitness', 'sumber', 'metode'], sort=False)
     .agg(n=('rep', 'count'), centroid=('centroid', 'mean'), bawah=('bawah', 'mean'), atas=('atas', 'mean'), pusat=('pusat', 'mean'))
     .round(3).reset_index())
T.to_csv(f'{OUT}/TableS3_structural_bias_no_fitness.csv', index=False)
print(T.to_string(index=False))
