"""core.py — fondasi bersama analisis lanjutan (06_reanalysis_stored_units, 07_subset_encoding_variants,
08_subset_lightgbm_mcc).

Prinsip: SEMUA komponen yang sudah dipakai pada 280 unit (data, preprocessing, subsampling, CVEvaluator,
Objective, Tracker, BFO/HHO/SMA, behavior_metrics, reevaluate, holdout) TIDAK ditulis ulang, tetapi diambil
langsung dari src/01_experiment_v5_2.py (seperti 04c_structural_bias_no_fitness.py). Dengan begitu hasil analisis lanjutan
memakai kode yang identik dengan eksperimen utama. Yang ditambahkan di sini hanya:
  - transformasi koordinat (flip, flip sebagian, encoding log) sebagai pembungkus evaluator;
  - optimizer baru: SMA dengan kontraksi ke centroid / solusi terbaik, DE, CMA-ES, TPE generik;
  - evaluator generik (keluarga model XGBoost/LightGBM, fitness wF1/MCC) untuk 08_subset_lightgbm_mcc;
  - runner per blok dengan checkpoint, dan fungsi statistik berpasangan.
"""
import os, sys, json, time, math, warnings, itertools, glob, shutil, platform, socket, datetime, inspect, hashlib

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
_trapz = getattr(np, 'trapezoid', None) or np.trapz

# ------------------------------------------------------------------ lokasi
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC_DIR)
# worker paralel (joblib/loky) perlu menemukan modul ini
if SRC_DIR not in os.environ.get('PYTHONPATH', '').split(os.pathsep):
    os.environ['PYTHONPATH'] = SRC_DIR + os.pathsep + os.environ.get('PYTHONPATH', '')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

_SRC = open(os.path.join(SRC_DIR, '01_experiment_v5_2.py'), encoding='utf-8').read()


def _seg(a, b):
    i = _SRC.index(a)
    return _SRC[i:_SRC.index(b, i)]


# ------------------------------------------------------------------ konfigurasi (identik dengan v5.2 'full')
RUN_MODE = 'full'
N_WORKERS = 2                     # hanya untuk XGB_N_JOBS di bawah; worker sebenarnya diatur di tiap skrip
DATA_DIR_CANDIDATES = ([os.environ['P2_DATA_DIR']] if 'P2_DATA_DIR' in os.environ else []) + [
    os.path.join(REPO, 'data'),
    '/content/drive/MyDrive/Colab Notebooks/Paper 2/data',
    '/content/drive/MyDrive/Paper2/data', '/content', '.', './data', '../data']
exec(_seg('SEED_BASE = 42', 'SETTINGS = {'), globals())
exec(_seg('SETTINGS = {', 'S = SETTINGS[RUN_MODE]'), globals())
S = SETTINGS['full']
exec(_seg('TEST_SIZE      = 0.15', '# Tag varian data'), globals())
XGB_N_JOBS = 1                    # sama dengan eksperimen utama yang dijalankan paralel (1 thread per model)
exec(_seg('DATASET_REGISTRY = {', 'DATA = {}'), globals())
exec(_seg('class HighMissingDropper', "print('Kelayakan level"), globals())
exec(_seg('LB = np.array', "print('Evaluator siap"), globals())
exec(_seg('def noise_analysis', 'def fla_analysis'), globals())
exec(_seg('def fla_analysis', 'class Tracker:'), globals())
exec(_seg('class Tracker:', 'def _opt_params():'), globals())
exec(_seg('def behavior_metrics', 'def run_unit('), globals())        # + holdout_eval, reevaluate, corner_eval, unit_id
OPTIMIZERS = {'RS': run_rs, 'BFO': run_bfo, 'HHO': run_hho, 'SMA': run_sma,
              'SMAc': lambda o, r, n, t: run_sma(o, r, n, t, centered=True)}   # urutan = indeks seed v5.2

HP_NAMES = [p[0] for p in PARAM_SPACE]
IMPORTANT_DIMS = [HP_NAMES.index(n) for n in ('min_child_weight', 'gamma', 'reg_alpha')]

_DATA = {}


def get_data(ds):
    if ds not in _DATA:
        _DATA[ds] = load_dataset(ds)
    return _DATA[ds]


# 12 unit yang sama dengan uji ketahanan mealpy (03_robustness_mealpy.py)
SUBSET_12 = [(ds, ir, r) for ds, ir in (('Cervical', None), ('Cervical', 6), ('Breast', None)) for r in range(4)]


ON_KAGGLE = os.path.isdir('/kaggle/working')


def default_output_base():
    if ON_KAGGLE:
        return '/kaggle/working/results'            # isi /kaggle/working tersimpan sebagai Output setelah commit
    for d in ('/content/drive/MyDrive/Colab Notebooks/Paper 2/outputs', '/content/drive/MyDrive/Paper2/outputs'):
        if os.path.isdir(d):
            return d
    return os.path.join(REPO, 'results')


MAIN_UNITS_DIR = os.environ.get('P2_MAIN_UNITS', os.path.join(REPO, 'results', 'units'))
TPE_UNITS_DIR = os.environ.get('P2_TPE_UNITS', os.path.join(REPO, 'results', 'baseline_tpe', 'units'))


def load_unit_json(uid, folder=None):
    p = os.path.join(folder or MAIN_UNITS_DIR, uid + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def load_unit_npz(uid, folder=None):
    with np.load(os.path.join(folder or MAIN_UNITS_DIR, uid + '.npz')) as z:
        return {k: z[k] for k in z.files}


# ================================================================== transformasi koordinat
# Optimizer selalu bekerja pada u ∈ [0,1]^9 (ruang internal). T(u) = x ∈ [0,1]^9 adalah koordinat ter-normalisasi
# LINIER yang dipakai decode() asli, sehingga parameter XGBoost (termasuk pembulatan integer) dihitung oleh kode v5.2.
def _log_decode(u):
    """Encoding log: learning_rate log10 antara 0,01–0,30; gamma, reg_alpha, reg_lambda log1p antara 0–batas atas.
    Rentang tiap hyperparameter TIDAK berubah (u=0 → batas bawah, u=1 → batas atas); hanya kerapatannya."""
    u = np.clip(np.asarray(u, float), 0, 1)
    x = u.copy()
    for name in ('learning_rate', 'gamma', 'reg_alpha', 'reg_lambda'):
        j = HP_NAMES.index(name)
        lo, hi = LB[j], UB[j]
        if lo > 0:
            v = lo * (hi / lo) ** u[..., j]
        else:
            v = np.expm1(u[..., j] * np.log1p(hi))
        x[..., j] = (v - lo) / (hi - lo)
    return x


def _flip(u):
    return 1.0 - np.clip(np.asarray(u, float), 0, 1)


def _partial_flip(dims):
    mask = np.zeros(D, bool); mask[list(dims)] = True

    def f(u):
        u = np.clip(np.asarray(u, float), 0, 1)
        return np.where(mask, 1.0 - u, u)
    return f


TRANSFORMS = {
    'lin': lambda u: np.clip(np.asarray(u, float), 0, 1),
    'flip': _flip,
    'log': _log_decode,
    'log_flip': lambda u: _log_decode(_flip(u)),                     # ruang log yang dicerminkan
    'flipImp': _partial_flip(IMPORTANT_DIMS),                        # hanya 3 dimensi berpengaruh dicerminkan
    'flipOth': _partial_flip([j for j in range(D) if j not in IMPORTANT_DIMS]),   # 6 dimensi lainnya
}


class TransformedEvaluator:
    def __init__(self, ev, T):
        self.ev, self.T = ev, T

    def __call__(self, u):
        return self.ev(self.T(u))

    def __getattr__(self, k):
        return getattr(self.ev, k)


# ================================================================== optimizer tambahan
def run_sma_anchor(obj, rng, pop_size, track, z=0.03, anchor='mean'):
    """SMA identik dengan run_sma v5.2, KECUALI cabang kontraksi: vc·(X − a) + a, dengan a = centroid populasi
    (anchor='mean') atau solusi terbaik (anchor='best'). Titik tarik kontraksi menjadi relatif terhadap populasi,
    sehingga (secara desain) tidak bergantung pada sistem koordinat."""
    P = rng.random((pop_size, D)); F = np.array([obj(x) for x in P]); track(P)
    best_x, best_f = P[np.argmin(F)].copy(), F.min()
    while True:
        prog = min(max(obj.progress, 1e-6), 1 - 1e-6)
        a, bcoef = math.atanh(1 - prog), 1 - prog
        order = np.argsort(F); bF, wF = F[order[0]], F[order[-1]]
        W = np.ones((pop_size, D))
        for rank, i in enumerate(order):
            ratio = (bF - F[i]) / (bF - wF - 1e-12)
            term = rng.random(D) * np.log10(ratio + 1)
            W[i] = 1 + term if rank < pop_size / 2 else 1 - term
        mean_pos = P.mean(axis=0)
        for i in range(pop_size):
            if rng.random() < z:
                P[i] = rng.random(D)
            else:
                p = math.tanh(abs(F[i] - best_f))
                vb, vc = rng.uniform(-a, a, D), rng.uniform(-bcoef, bcoef, D)
                A, B = rng.integers(pop_size, size=2)
                r = rng.random(D)
                ref = mean_pos if anchor == 'mean' else best_x
                contract = ref + vc * (P[i] - ref)
                P[i] = np.where(r < p, best_x + vb * (W[i] * P[A] - P[B]), contract)
            P[i] = np.clip(P[i], 0, 1); F[i] = obj(P[i])
            if F[i] < best_f:
                best_x, best_f = P[i].copy(), F[i]
        track(P)


def run_de(obj, rng, pop_size, track, F_=0.5, CR=0.9):
    """Differential evolution DE/rand/1/bin (Storn & Price, 1997), seleksi greedy, perbaikan batas dengan clipping."""
    P = rng.random((pop_size, D)); F = np.array([obj(x) for x in P]); track(P)
    while True:
        for i in range(pop_size):
            a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            v = np.clip(P[a] + F_ * (P[b] - P[c]), 0, 1)
            jr = rng.integers(D)
            mask = rng.random(D) < CR; mask[jr] = True
            trial = np.where(mask, v, P[i])
            ft = obj(trial)
            if ft <= F[i]:
                P[i], F[i] = trial, ft
        track(P)


def run_cmaes(obj, rng, pop_size, track, sigma0=0.3):
    """CMA-ES (Hansen, pustaka pycma) dengan batas [0,1], titik awal acak (bukan pusat ruang, agar tidak
    memasukkan bias pusat), popsize = 20. Restart otomatis bila konvergen sebelum anggaran habis."""
    import cma
    while True:
        es = cma.CMAEvolutionStrategy(rng.random(D), sigma0,
                                      {'popsize': pop_size, 'bounds': [0, 1], 'verbose': -9,
                                       'seed': int(rng.integers(1, 2 ** 31 - 1))})
        while not es.stop():
            X = es.ask()
            es.tell(X, [obj(np.asarray(x)) for x in X])
            track(np.clip(np.array(X), 0, 1))


ALL_METHODS = dict(OPTIMIZERS)
ALL_METHODS.update({'SMAm': lambda o, r, n, t: run_sma_anchor(o, r, n, t, anchor='mean'),
                    'SMAb': lambda o, r, n, t: run_sma_anchor(o, r, n, t, anchor='best'),
                    'DE': run_de, 'CMAES': run_cmaes})
# indeks seed: 0–4 sama dengan v5.2 (RS, BFO, HHO, SMA, SMAc) → arm ber-encoding lain memakai seed yang sama
RNG_INDEX = {k: i for i, k in enumerate(ALL_METHODS)}
T_STARTUP = 20


def run_tpe_obj(obj, seed):
    """TPE persis seperti 05_baseline_tpe.py (Optuna, 20 trial acak awal, multivariate=True), di ruang internal."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=T_STARTUP, multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    def f(trial):
        return obj(np.array([trial.suggest_float(n, 0.0, 1.0) for n in HP_NAMES]))
    try:
        study.optimize(f, n_trials=obj.budget, catch=())
    except BudgetExhausted:
        pass


def run_search(method, base_ev, transform, seed, budget=None, pop=None):
    """Satu pencarian. Return (obj, best_x_linear, beh). best_x selalu dalam koordinat linier ASLI."""
    budget = budget or S['opt_budget']; pop = pop or S['pop_size']
    T = TRANSFORMS[transform] if isinstance(transform, str) else transform
    obj = Objective(TransformedEvaluator(base_ev, T), budget)
    tr = Tracker(obj)
    if method == 'TPE':
        run_tpe_obj(obj, seed * 100 + 7)                         # seed sama dengan 05_baseline_tpe.py
    else:
        rng = np.random.default_rng([seed, RNG_INDEX[method]])
        try:
            ALL_METHODS[method](obj, rng, pop, tr)
        except BudgetExhausted:
            pass
    best_x = T(obj.best_x)
    if tr.last_pop is not None:
        beh = behavior_metrics(tr, obj, flipped=False)
        c_int = tr.last_pop.mean(axis=0)
    else:                                                        # TPE: 20 konfigurasi terbaik ≈ "populasi akhir"
        U = np.array(obj.X_log)
        c_int = U[np.argsort(obj.f_log)[:T_STARTUP]].mean(axis=0)
        tb = np.array(obj.trace_best); f0, fb = tb[0], tb[-1]
        beh = dict(n_evals=obj.n, evals_to_95pct=int(np.argmax(tb <= f0 - 0.95 * (f0 - fb)) + 1) if f0 > fb else 1)
    c_real = T(c_int)
    beh.update(final_centroid_internal=c_int.round(4).tolist(), final_centroid_real=c_real.round(4).tolist(),
               final_int_frac_at_lower=float(np.mean(c_int < 0.02)), final_int_frac_at_upper=float(np.mean(c_int > 0.98)),
               final_real_frac_at_lower=float(np.mean(c_real < 0.02)), final_real_frac_at_upper=float(np.mean(c_real > 0.98)),
               final_real_frac_at_center=float(np.mean(np.abs(c_real - 0.5) < 0.02)))
    beh.pop('final_centroid', None)
    obj._tracker = tr
    return obj, best_x, beh


# ================================================================== evaluator generik (model & metrik fitness)
PARAM_SPACE_LGBM = [
    ('learning_rate', 0.01, 0.30, 'float'), ('max_depth', 3, 10, 'int'), ('min_child_samples', 5, 50, 'int'),
    ('min_split_gain', 0.0, 5.0, 'float'), ('subsample', 0.5, 1.0, 'float'), ('colsample_bytree', 0.5, 1.0, 'float'),
    ('reg_alpha', 0.0, 10.0, 'float'), ('reg_lambda', 0.0, 10.0, 'float'), ('n_estimators', 50, 500, 'int')]


def decode_lgbm(x):
    if x is None:
        return {}
    x = np.clip(np.asarray(x, float), 0, 1)
    lo = np.array([p[1] for p in PARAM_SPACE_LGBM], float); hi = np.array([p[2] for p in PARAM_SPACE_LGBM], float)
    raw = lo + x * (hi - lo)
    return {n: (int(round(v)) if t == 'int' else float(v)) for (n, _, _, t), v in zip(PARAM_SPACE_LGBM, raw)}


def make_lgbm(params, spw, seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, scale_pos_weight=spw, subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1)


FAMILIES = {'xgb': (decode, make_xgb), 'lgbm': (decode_lgbm, make_lgbm)}


class GenCVEvaluator(CVEvaluator):
    """CVEvaluator v5.2 (fold & preprocessing identik) dengan keluarga model dan metrik fitness yang dapat dipilih.
    family='xgb', fitness='wf1' menghasilkan nilai yang sama persis dengan CVEvaluator asli."""
    def __init__(self, X_dev, y_dev, n_splits, n_repeats, partition_seed, model_seed, family='xgb', fitness='wf1'):
        super().__init__(X_dev, y_dev, n_splits, n_repeats, partition_seed, model_seed)
        self.family, self.fitness = family, fitness

    def __call__(self, x):
        dec, make = FAMILIES[self.family]
        params = dec(x) if x is not None else {}
        wf1, mf1, mcc = [], [], []
        for Xtr, ytr, Xva, yva, spw in self.folds:
            m = make(params, spw, self.model_seed).fit(Xtr, ytr)
            p = m.predict(Xva)
            wf1.append(f1_score(yva, p, average='weighted', zero_division=0))
            mf1.append(f1_score(yva, p, pos_label=1, zero_division=0))
            mcc.append(matthews_corrcoef(yva, p))
            self.n_fits += 1
        fit_v = {'wf1': wf1, 'mf1': mf1, 'mcc': mcc}[self.fitness]
        return dict(f=-float(np.mean(fit_v)), wf1=float(np.mean(wf1)), wf1_sd=float(np.std(wf1)),
                    mf1=float(np.mean(mf1)), mcc=float(np.mean(mcc)),
                    fold_wf1=wf1, fold_mf1=mf1, fold_mcc=mcc)


def reevaluate_g(x, X_dev, y_dev, scheme, n_rep, seed, family='xgb'):
    """Sama dengan reevaluate v5.2 (partisi baru seed*1000+500+r), untuk keluarga model apa pun."""
    k, rep = scheme
    rs = [GenCVEvaluator(X_dev, y_dev, k, rep, partition_seed=seed * 1000 + 500 + r, model_seed=seed,
                         family=family)(x) for r in range(n_rep)]
    reevaluate_g.last_raw = pack_evals(rs)
    return {m: float(np.mean([r[m] for r in rs])) for m in ('wf1', 'mf1', 'mcc')}


def holdout_g(x, X_dev, y_dev, X_te, y_te, seed, family='xgb'):
    dec, make = FAMILIES[family]
    pre = make_preprocessor().fit(X_dev, y_dev)
    m = make(dec(x) if x is not None else {}, scale_pos_weight_for(y_dev), seed)
    m.fit(pre.transform(X_dev), y_dev)
    Xt = pre.transform(X_te); p = m.predict(Xt); pr = m.predict_proba(Xt)[:, 1]
    return dict(ho_wf1=f1_score(y_te, p, average='weighted', zero_division=0),
                ho_mf1=f1_score(y_te, p, pos_label=1, zero_division=0),
                ho_mcc=matthews_corrcoef(y_te, p), ho_bacc=balanced_accuracy_score(y_te, p),
                ho_auc=roc_auc_score(y_te, pr) if len(np.unique(y_te)) > 1 else np.nan)


# ================================================================== runner per blok (checkpoint)
def setup_unit(ds, ir, run, scheme_name='cv5x1'):
    X, y, _ = get_data(ds)
    seed = SEED_BASE + run
    X_dev, X_te, y_dev, y_te, i_dev, i_te = make_unit_data(X, y, ir, seed)
    return dict(seed=seed, X_dev=X_dev, X_te=X_te, y_dev=y_dev, y_te=y_te, scheme=S['cv_schemes'][scheme_name],
                scheme_name=scheme_name, idx_dev=i_dev, idx_test=i_te)


def block_uid(block, ds, ir, run, scheme_name='cv5x1'):
    return f'{unit_id(ds, ir, scheme_name, run)}__{block}'


def run_block(block, arms, ds, ir, run, out_dir, family='xgb', fitness='wf1', include_default=False,
              budget=None, pop=None, verbose=True):
    """arms: daftar (nama_arm, metode, transformasi). Satu file JSON+NPZ per (unit, blok) → checkpoint."""
    os.makedirs(os.path.join(out_dir, 'units'), exist_ok=True)
    uid = block_uid(block, ds, ir, run)
    path = os.path.join(out_dir, 'units', uid + '.json')
    if os.path.exists(path):
        if verbose: print(f'⏭  {uid} sudah ada (checkpoint), dilewati')
        return json.load(open(path))
    U = setup_unit(ds, ir, run)
    seed, scheme = U['seed'], U['scheme']
    base_ev = GenCVEvaluator(U['X_dev'], U['y_dev'], *scheme, partition_seed=seed, model_seed=seed,
                             family=family, fitness=fitness)
    res = dict(unit=uid, base_unit=unit_id(ds, ir, 'cv5x1', run), block=block, dataset=ds, ir_target=ir, run=run,
               seed=seed, family=family, fitness=fitness, arms={})
    RAW = {}
    t0 = time.time()
    # cek reproduksi terhadap eksperimen utama (XGBoost default pada partisi pencarian yang sama)
    if family == 'xgb':
        main = load_unit_json(res['base_unit'])
        r0 = base_ev(None)
        if main is not None:
            res['repro_check_default_cv_wf1_diff'] = float(r0['wf1'] - main['default']['cv_wf1'])
    if include_default:
        r0 = base_ev(None)
        rv = reevaluate_g(None, U['X_dev'], U['y_dev'], scheme, S['reeval_repeats'], seed, family)
        res['arms']['default'] = dict(method='default', transform=None, best_cv_wf1=r0['wf1'], best_cv_mcc=r0['mcc'],
                                      reeval_cv_wf1=rv['wf1'], reeval_cv_mf1=rv['mf1'], reeval_cv_mcc=rv['mcc'],
                                      **holdout_g(None, U['X_dev'], U['y_dev'], U['X_te'], U['y_te'], seed, family))
    for name, method, transform in arms:
        _t = time.time()
        obj, best_x, beh = run_search(method, base_ev, transform, seed, budget, pop)
        rv = reevaluate_g(best_x, U['X_dev'], U['y_dev'], scheme, S['reeval_repeats'], seed, family)
        ho = holdout_g(best_x, U['X_dev'], U['y_dev'], U['X_te'], U['y_te'], seed, family)
        T = TRANSFORMS[transform]
        RAW[f'opt_{name}_Xreal'] = T(np.array(obj.X_log)).astype(np.float32)
        for m in ('wf1', 'mf1', 'mcc'):
            RAW[f'opt_{name}_{m}'] = np.array([r[m] for r in obj.r_log], np.float32)
        tr = obj._tracker
        if getattr(tr, 'pops', None):
            RAW[f'opt_{name}_pops_real'] = T(np.stack(tr.pops)).astype(np.float32)
        RAW.update({f'opt_{name}_reeval_{k}': v for k, v in reevaluate_g.last_raw.items()})
        best_info = obj.best_info
        res['arms'][name] = dict(method=method, transform=transform, behavior=beh, best_x=best_x.tolist(),
                                 best_params=FAMILIES[family][0](best_x),
                                 best_cv_fitness=-obj.best_f, best_cv_wf1=best_info['wf1'], best_cv_mf1=best_info['mf1'],
                                 best_cv_mcc=best_info['mcc'],
                                 reeval_cv_wf1=rv['wf1'], reeval_cv_mf1=rv['mf1'], reeval_cv_mcc=rv['mcc'],
                                 winners_curse=-obj.best_f - rv[fitness], **ho,
                                 trace_best=[-v for v in obj.trace_best], search_sec=time.time() - _t)
        if verbose:
            o = res['arms'][name]
            print(f"   {name:<14} bestCV={o['best_cv_fitness']:.4f} reeval_{fitness}={rv[fitness]:.4f} "
                  f"bawah(real)={beh['final_real_frac_at_lower']:.2f} atas(real)={beh['final_real_frac_at_upper']:.2f} "
                  f"({o['search_sec'] / 60:.1f} mnt)", flush=True)
    import sklearn, scipy, xgboost
    res['runtime_sec'] = time.time() - t0
    res['meta'] = dict(host=socket.gethostname(), finished_at=datetime.datetime.now().isoformat(timespec='seconds'),
                       libs=dict(numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                                 xgboost=xgboost.__version__))
    np.savez_compressed(os.path.join(out_dir, 'units', uid + '.npz'), **RAW)
    with open(path + '.tmp', 'w') as f:
        json.dump(res, f, default=float)
    os.replace(path + '.tmp', path)
    if verbose: print(f'✅ {uid}  {res["runtime_sec"] / 60:.1f} menit', flush=True)
    return res


def restore_checkpoints(out_dir, group_name, search_roots=('/kaggle/input',)):
    """Kaggle: salin checkpoint dari output versi sebelumnya (ditambahkan sebagai Input) ke out_dir/units,
    sehingga unit yang sudah selesai tidak dijalankan ulang. Mencari pola .../<group_name>/units/*.json."""
    os.makedirs(os.path.join(out_dir, 'units'), exist_ok=True)
    n = 0
    for root in search_roots:
        for f in glob.glob(os.path.join(root, '**', group_name, 'units', '*.json'), recursive=True):
            dst = os.path.join(out_dir, 'units', os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
                npz = f[:-5] + '.npz'
                if os.path.exists(npz):
                    shutil.copy(npz, dst[:-5] + '.npz')
                n += 1
    print(f'♻️  {n} checkpoint dipulihkan dari output sebelumnya' if n else 'Tidak ada checkpoint sebelumnya untuk dipulihkan')
    return n


def run_blocks_parallel(jobs, out_dir, n_workers=1, deadline_hours=None):
    """jobs: daftar dict(block, arms, ds, ir, run, family, fitness, include_default).
    deadline_hours: setelah batas ini TIDAK ada pekerjaan baru yang dimulai (pekerjaan yang sedang berjalan diselesaikan),
    agar sesi Kaggle/Colab berakhir normal dan output tersimpan. Sisanya dilanjutkan pada versi berikutnya."""
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(out_dir, 'units', block_uid(j['block'], j['ds'], j['ir'], j['run']) + '.json'))]
    print(f'{len(jobs)} pekerjaan, {len(jobs) - len(todo)} sudah ada checkpoint, {len(todo)} dijalankan '
          f'dengan {n_workers} worker' + (f'; tidak memulai pekerjaan baru setelah {deadline_hours} jam' if deadline_hours else ''),
          flush=True)
    t0 = time.time()
    state = {'stopped': 0}

    def _open():
        for j in todo:
            if deadline_hours and (time.time() - t0) / 3600 > deadline_hours:
                state['stopped'] += 1
                continue
            yield j

    if n_workers <= 1:
        for i, j in enumerate(_open(), 1):
            print(f"\n[{i}/{len(todo)}] {block_uid(j['block'], j['ds'], j['ir'], j['run'])}", flush=True)
            run_block(out_dir=out_dir, **j)
    else:
        from joblib import Parallel, delayed

        def _job(j):
            r = run_block(out_dir=out_dir, verbose=False, **j)
            return r['unit'], r['runtime_sec']
        done = 0
        # pre_dispatch='n_jobs': pekerjaan baru baru diambil saat ada worker kosong → cek batas waktu tepat waktu
        for uid, sec in Parallel(n_jobs=n_workers, backend='loky', return_as='generator_unordered',
                                 pre_dispatch='n_jobs')(delayed(_job)(j) for j in _open()):
            done += 1
            print(f'✅ [{done}/{len(todo)}] {uid}  {sec / 60:.1f} menit | total {(time.time() - t0) / 60:.1f} menit',
                  flush=True)
    left = len([j for j in jobs if not os.path.exists(
        os.path.join(out_dir, 'units', block_uid(j['block'], j['ds'], j['ir'], j['run']) + '.json'))])
    print(f'Selesai dalam {(time.time() - t0) / 60:.1f} menit. Sisa pekerjaan belum dijalankan: {left}'
          + (' → jalankan versi berikutnya dengan output ini sebagai Input' if left else ' → SEMUA SELESAI'))
    return left


def load_block_results(out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, 'units', '*.json'))):
        u = json.load(open(f))
        for name, o in u['arms'].items():
            rows.append(dict(unit=u['base_unit'], block=u['block'], dataset=u['dataset'],
                             cond=f"{u['dataset']}|{'asli' if u['ir_target'] is None else str(u['ir_target']) + ':1'}",
                             run=u['run'], family=u['family'], fitness=u['fitness'], arm=name,
                             **{k: v for k, v in o.items() if k not in ('behavior', 'best_x', 'best_params', 'trace_best')},
                             **{k: v for k, v in o.get('behavior', {}).items()
                                if not isinstance(v, (list, dict))}))
    return pd.DataFrame(rows)


def load_main_reference(units, metric_cols=('reeval_cv_wf1', 'reeval_cv_mf1', 'reeval_cv_mcc', 'ho_wf1')):
    """Hasil eksperimen utama (encoding linier) dan TPE untuk unit yang sama → format panjang (unit, arm, ...)."""
    rows = []
    for uid in units:
        m = load_unit_json(uid)
        if m is not None:
            for name, o in m['optimizers'].items():
                rows.append(dict(unit=uid, arm=f'{name}_lin' if '_flip' not in name else name.replace('_flip', '_lin_flip'),
                                 **{c: o.get(c) for c in metric_cols},
                                 final_real_frac_at_lower=o['behavior'].get('final_real_frac_at_lower'),
                                 final_real_frac_at_upper=o['behavior'].get('final_real_frac_at_upper')))
            rows.append(dict(unit=uid, arm='default_lin', **{c: m['default'].get(c) for c in metric_cols}))
        t = load_unit_json(uid, TPE_UNITS_DIR)
        if t is not None:
            for name, o in t['optimizers'].items():
                rows.append(dict(unit=uid, arm='TPE_lin' if name == 'TPE' else 'TPE_lin_flip',
                                 **{c: o.get(c) for c in metric_cols},
                                 final_real_frac_at_lower=o['behavior'].get('final_real_frac_at_lower'),
                                 final_real_frac_at_upper=o['behavior'].get('final_real_frac_at_upper')))
    return pd.DataFrame(rows)


# ================================================================== statistik berpasangan
def boot_ci_median(d, B=5000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(d[rng.integers(0, len(d), (B, len(d)))], axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


def paired_summary(wide, pairs, metric_label=''):
    """wide: DataFrame index=unit, kolom=arm. pairs: daftar (a, b, label) → Δ = a − b per unit."""
    out = []
    for a, b, label in pairs:
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).dropna()
        n = len(d)
        nz = d[d != 0]
        if len(nz) >= 1:
            w = stats.wilcoxon(d, zero_method='wilcox')
            p = float(w.pvalue)
            r_pos = stats.rankdata(np.abs(nz))
            r_rb = float((r_pos[nz > 0].sum() - r_pos[nz < 0].sum()) / r_pos.sum())
        else:
            p, r_rb = np.nan, 0.0
        lo, hi = boot_ci_median(d)
        out.append(dict(comparison=label, a=a, b=b, metric=metric_label, n=n, median_delta=float(np.median(d)) if n else np.nan,
                        mean_delta=float(d.mean()) if n else np.nan, ci_lo=lo, ci_hi=hi, n_pos=int((d > 0).sum()),
                        n_neg=int((d < 0).sum()), r_rank_biserial=r_rb, p=p))
    T = pd.DataFrame(out)
    if len(T):
        from statsmodels.stats.multitest import multipletests
        ok = T.p.notna()
        T['p_holm'] = np.nan
        if ok.any():
            T.loc[ok, 'p_holm'] = multipletests(T.loc[ok, 'p'], method='holm')[1]
    return T


print(f'core siap ✅  (kode inti diambil dari 01_experiment_v5_2.py; data dicari di {DATA_DIR_CANDIDATES[0]})')
