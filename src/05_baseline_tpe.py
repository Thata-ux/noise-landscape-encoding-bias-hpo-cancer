# %% [markdown]
# # Paper 2 — Baseline Bayesian Optimization (TPE)
# ## TPE dengan anggaran evaluasi yang sama, termasuk flip test sebagai kontrol positif
#
# **Tujuan.** Menambahkan satu baseline Bayesian Optimization, yaitu Tree-structured Parzen Estimator (TPE; Optuna 4.1),
# dengan anggaran 300 evaluasi objektif yang sama dengan RS, BFO, HHO, dan SMA. TPE juga dijalankan pada ruang yang
# dicerminkan (x → 1 − x). Karena TPE dengan prior seragam tidak memiliki arah istimewa, TPE seharusnya **lolos** flip test;
# hasil ini menjadi kontrol positif bahwa flip test memang membedakan optimizer yang bias dari yang tidak.
#
# **Identik dengan eksperimen utama (v5.2):** data & preprocessing, subsampling, seed per run, partisi CV pencarian
# (cv5x1), fungsi objektif (−wF1), ruang [0,1]⁹, partisi evaluasi ulang (5×), dan hold-out. TPE: 20 trial acak awal
# (= ukuran populasi metaheuristik), multivariate=True, seed sama untuk orientasi asli dan dicerminkan.
#
# **Unit bawaan:** 28 unit cv5x1 = 14 kondisi × run 0–1 (`P2_TPE_RUNS='0,1'`). Untuk 140 unit penuh: `P2_TPE_RUNS='0,1,2,3,4,5,6,7,8,9'`.
# Perkiraan waktu: ±6–10 menit per unit (2 pencarian TPE + evaluasi ulang) → 28 unit ±2–3 jam dengan 2 worker.
#
# **Cara pakai (Colab):** mount Google Drive, lalu jalankan semua sel. Hasil di `.../outputs/baseline_tpe/v5.2/full/<varian>/`
# dengan checkpoint per unit; bagian T4 otomatis memasangkan hasil dengan unit eksperimen utama.

# %%
# Colab: mount Google Drive dulu (hapus # pada dua baris berikut)
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# 0. INSTALL & IMPORT
# ============================================================
# Di Colab, jalankan sekali (hapus tanda # di baris berikut):
# !pip install -q xgboost optuna==4.1.0

import os, sys, json, time, math, warnings, itertools, glob, shutil, platform, socket, datetime, inspect, hashlib
from dataclasses import dataclass, field

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
from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score,
                             balanced_accuracy_score)
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
_trapz = getattr(np, 'trapezoid', None) or np.trapz
print('Library siap ✅')

# %%
# ============================================================
# 1. KONFIGURASI UTAMA — SESUAIKAN DI SINI
# ============================================================
RUN_MODE = 'full'           # 'full' (protokol lengkap, 280 unit) | 'pilot' (Cervical, budget kecil)
CODE_VERSION = 'v5.2'        # hasil disimpan per versi → checkpoint versi lama tidak terbawa

# Jumlah unit yang dijalankan paralel. Colab gratis (2 core): biarkan 1.
# Laptop/PC: isi ≈ jumlah core fisik − 1 (mis. i7 8-core → 6). Tiap worker memakai 1 thread XGBoost.
N_WORKERS = 2

# Potongan pekerjaan (untuk membagi eksperimen ke beberapa mesin/sesi). None = semua.
#   contoh: CHUNK = dict(datasets=['Cervical'], schemes=['cv5x1'], runs=[0, 1, 2, 3, 4], irs=None)
#   irs memakai label: 'asli', 3, 6, 12, 20
CHUNK = dict(datasets=None, schemes=None, runs=None, irs=None)

# Override via environment variable (berguna saat menjalankan file .py di laptop)
RUN_MODE  = os.environ.get('P2_RUN_MODE', RUN_MODE)
N_WORKERS = int(os.environ.get('P2_WORKERS', N_WORKERS))


def _env_list(key, cast=str):
    v = os.environ.get(key)
    if not v:
        return None
    out = []
    for tok in v.split(','):
        tok = tok.strip()
        if '-' in tok and cast is int:                      # rentang, mis. 0-4
            a, b = tok.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(cast(tok))
    return out


for _k, _key, _cast in [('datasets', 'P2_DATASETS', str), ('schemes', 'P2_SCHEMES', str),
                        ('runs', 'P2_RUNS', int), ('irs', 'P2_IRS', str)]:
    if _env_list(_key, _cast) is not None:
        CHUNK[_k] = _env_list(_key, _cast)

ON_KAGGLE = os.path.exists('/kaggle/working')

# Lokasi data & output. Notebook mencoba path pertama yang ada.
DATA_DIR_CANDIDATES = ([os.environ['P2_DATA_DIR']] if 'P2_DATA_DIR' in os.environ else []) + [
    '/content/drive/MyDrive/Colab Notebooks/Paper 2/data',   # Google Drive (Colab)
    '/content/drive/MyDrive/Paper2/data',
    '/content',                              # upload langsung ke Colab
    '.',                                     # lokal
    './data',
    '../data',                               # dijalankan dari folder notebooks/ repositori
    '/mnt/user-data/uploads',
] + sorted({os.path.dirname(p) for p in glob.glob('/kaggle/input/**/*.csv', recursive=True)})
# PENTING: di Colab, Google Drive selalu ada di '/content/drive/MyDrive' (bukan 'Drive Saya').
# Jalankan drive.mount('/content/drive') SEBELUM sel ini agar hasil tersimpan permanen.
OUTPUT_DIR = ('/content/drive/MyDrive/Colab Notebooks/Paper 2/outputs'
              if os.path.exists('/content/drive/MyDrive') else './paper2_outputs')
if ON_KAGGLE:
    OUTPUT_DIR = '/kaggle/working/outputs'
OUTPUT_DIR = os.environ.get('P2_OUTPUT_DIR', OUTPUT_DIR)
if os.path.exists('/content') and not OUTPUT_DIR.startswith('/content/drive'):
    print('⚠️  Drive belum ter-mount: hasil disimpan di penyimpanan SEMENTARA Colab dan akan hilang saat sesi berakhir!')

SEED_BASE = 42

# ---- Keputusan data (WAJIB disepakati dengan promotor, dicatat di Metodologi) ----
DROP_THYROID_RESPONSE   = True   # 'Response' adalah respons pasca-terapi; hampir menentukan label Recurred
DROP_CERVICAL_OTHER_TESTS = True   # Hinselmann/Schiller/Citology: di UCI ketiganya juga variabel target.
                                   # Pilot: jika dipertahankan, landscape Cervical hampir datar (rentang wF1 ~0,01).
                                   # False = konsisten dengan paper 1 & Tabel 3.2 proposal.
HIGH_MISSING_THRESHOLD  = 0.40   # fitur dengan missing > 40% (dihitung pada training fold) dibuang
USE_SCALE_POS_WEIGHT_IF_IR_GT = 3.0  # sesuai proposal: scale_pos_weight jika IR > 3:1 (training fold)

# ---- Ruang hyperparameter (Tabel 3.5 proposal), dioptimasi pada ruang ter-normalisasi [0,1]^9 ----
PARAM_SPACE = [
    # (nama, low, high, tipe)
    ('learning_rate',    0.01, 0.30, 'float'),
    ('max_depth',        3,    10,   'int'),
    ('min_child_weight', 1,    10,   'int'),
    ('gamma',            0.0,  5.0,  'float'),
    ('subsample',        0.5,  1.0,  'float'),
    ('colsample_bytree', 0.5,  1.0,  'float'),
    ('reg_alpha',        0.0,  10.0, 'float'),
    ('reg_lambda',       0.0,  10.0, 'float'),
    ('n_estimators',     50,   500,  'int'),
]
D = len(PARAM_SPACE)

SETTINGS = {
    'pilot': dict(
        datasets      = ['Cervical'],
        ir_targets    = [None, 6],          # None = rasio asli
        cv_schemes    = {'cv5x1': (5, 1), 'cv5x3': (5, 3)},
        n_runs        = 2,
        noise_configs = 12, noise_repeats = 5,
        fla_samples   = 60, walk_length = 40,
        opt_budget    = 150, pop_size = 10,   # 150: agar BFO sempat reproduksi & eliminasi-dispersal
        reeval_repeats= 3,
        optimizers    = ['RS', 'BFO', 'HHO', 'SMA', 'SMAc'],
        flip_optimizers = ['BFO', 'HHO', 'SMA'],   # flip test (RS invarian, tidak perlu)
        flip_schemes    = ['cv5x1'],               # flip hanya pada skema ini (hemat waktu)
    ),
    'full': dict(
        datasets      = ['Cervical', 'Breast', 'Lung', 'Thyroid'],
        ir_targets    = [None, 3, 6, 12, 20],
        exclude_levels = [['Lung', 6]],           # Lung 6:1 ≈ Lung asli (IR 6,26) → dikeluarkan
        cv_schemes    = {'cv5x1': (5, 1), 'cv5x3': (5, 3)},
        n_runs        = 10,
        noise_configs = 30, noise_repeats = 10,
        fla_samples   = 200, walk_length = 100,
        opt_budget    = 300, pop_size = 20,
        reeval_repeats= 5,
        optimizers    = ['RS', 'BFO', 'HHO', 'SMA', 'SMAc'],
        flip_optimizers = ['BFO', 'HHO', 'SMA'],
        flip_schemes    = ['cv5x1'],
    ),
}
S = SETTINGS[RUN_MODE]

TEST_SIZE      = 0.15     # 85:15 sesuai proposal
MIN_MINORITY   = 25       # minimal sampel minoritas (sebelum split) agar 5-fold masih bermakna
WALK_STEP_SD   = 0.05     # ukuran langkah random walk di ruang ter-normalisasi
TOP_FRACTION   = 0.10     # proporsi konfigurasi terbaik untuk dispersion & clustering
XGB_N_JOBS     = 1 if N_WORKERS > 1 else -1   # hindari oversubscription saat paralel

# Tag varian data → hasil tiap kombinasi keputusan data disimpan terpisah (tidak saling menimpa checkpoint)
VARIANT_TAG = (f"thyResp{'Drop' if DROP_THYROID_RESPONSE else 'Keep'}"
               f"_cervTests{'Drop' if DROP_CERVICAL_OTHER_TESTS else 'Keep'}")
# pilot & full disimpan di folder BERBEDA: nama unit sama, tetapi budget berbeda → tidak boleh tercampur
OUTPUT_ROOT = OUTPUT_DIR
OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'baseline_tpe', CODE_VERSION, RUN_MODE, VARIANT_TAG)
os.makedirs(os.path.join(OUTPUT_DIR, 'units'), exist_ok=True)

# Kaggle: lanjutkan dari output versi sebelumnya yang ditambahkan sebagai input notebook
if ON_KAGGLE:
    _n_copied = 0
    for _p in glob.glob(f'/kaggle/input/**/baseline_tpe/{CODE_VERSION}/{RUN_MODE}/{VARIANT_TAG}/units/*', recursive=True):
        _dst = os.path.join(OUTPUT_DIR, 'units', os.path.basename(_p))
        if not os.path.exists(_dst):
            shutil.copy(_p, _dst); _n_copied += 1
    if _n_copied:
        print(f'↩️  {_n_copied} file checkpoint disalin dari input Kaggle')


def _lib_versions():
    import sklearn, scipy, xgboost
    return dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__, xgboost=xgboost.__version__)


RUN_META = dict(code_version=CODE_VERSION, run_mode=RUN_MODE, variant=VARIANT_TAG,
                drop_thyroid_response=DROP_THYROID_RESPONSE, drop_cervical_other_tests=DROP_CERVICAL_OTHER_TESTS,
                settings={k: (dict(v) if isinstance(v, dict) else v) for k, v in S.items()},
                param_space=PARAM_SPACE, test_size=TEST_SIZE, min_minority=MIN_MINORITY,
                walk_step_sd=WALK_STEP_SD, top_fraction=TOP_FRACTION, seed_base=SEED_BASE,
                scale_pos_weight_if_ir_gt=USE_SCALE_POS_WEIGHT_IF_IR_GT,
                high_missing_threshold=HIGH_MISSING_THRESHOLD, libs=_lib_versions())
# ---- Pengaman: pengaturan desain tidak boleh berubah di tengah running (checkpoint hanya cek nama unit) ----
_DESIGN_KEYS = ['code_version', 'run_mode', 'variant', 'settings', 'param_space', 'test_size', 'min_minority',
                'walk_step_sd', 'top_fraction', 'seed_base', 'scale_pos_weight_if_ir_gt', 'high_missing_threshold']
_design = json.dumps({k: RUN_META[k] for k in _DESIGN_KEYS}, sort_keys=True, default=str)
RUN_META['design_fingerprint'] = hashlib.sha1(_design.encode()).hexdigest()[:12]
_cfg_path = os.path.join(OUTPUT_DIR, 'run_config.json')
if os.path.exists(_cfg_path):
    _old = json.load(open(_cfg_path)).get('design_fingerprint')
    if _old and _old != RUN_META['design_fingerprint'] and not os.environ.get('P2_FORCE'):
        raise RuntimeError(
            f'⛔ Pengaturan desain berbeda dari running sebelumnya di folder ini ({_old} ≠ '
            f"{RUN_META['design_fingerprint']}). Hasil lama dan baru akan tercampur. Kembalikan pengaturan, "
            'atau gunakan folder output lain. (Hanya N_WORKERS dan CHUNK yang boleh diubah.)')
with open(_cfg_path, 'w') as _f:
    json.dump(RUN_META, _f, indent=1, default=str)
print(f"🔒 Desain fingerprint: {RUN_META['design_fingerprint']}")
print(f"RUN_MODE = {RUN_MODE} | output baseline TPE → {OUTPUT_DIR}")
print(json.dumps({k: (v if not isinstance(v, dict) else list(v)) for k, v in S.items()}, indent=1))

# %%
# ============================================================
# 2. DATA LOADING (label minoritas selalu = 1)
# ============================================================
DATASET_REGISTRY = {
    'Cervical': dict(file='risk_factors_cervical_cancer.csv', target='Biopsy',
                     drop=(['Hinselmann', 'Schiller', 'Citology'] if DROP_CERVICAL_OTHER_TESTS else [])),
    'Breast':   dict(file='Breast_Cancer_Winconsin.csv', target='diagnosis', drop=['id', 'Unnamed: 32']),
    'Lung':     dict(file='survey_lung_cancer.csv', target='LUNG_CANCER', drop=[]),
    'Thyroid':  dict(file='Thyroid_Diff.csv', target='Recurred',
                     drop=(['Response'] if DROP_THYROID_RESPONSE else [])),
}


def resolve_file(fname):
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{fname} tidak ditemukan di {DATA_DIR_CANDIDATES}')


def load_dataset(name):
    cfg = DATASET_REGISTRY[name]
    df = pd.read_csv(resolve_file(cfg['file']), na_values=['?', 'NA', 'null', ''])
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)          # dedup sebelum split (proposal 3.4)
    df = df.drop(columns=[c for c in cfg['drop'] if c in df.columns])
    y_raw = df.pop(cfg['target'])
    counts = y_raw.value_counts()
    minority_label = counts.idxmin()
    y = (y_raw == minority_label).astype(int).values           # minoritas → 1
    # kolom non-numerik → object dengan np.nan (aman untuk SimpleImputer, juga di pandas 3)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    info = dict(n_raw=n_raw, n=len(df), n_features=df.shape[1], minority_label=str(minority_label),
                n_pos=int(y.sum()), n_neg=int((1 - y).sum()), ir=float((1 - y).sum() / y.sum()))
    return df, y, info


DATA = {}
for ds in S['datasets']:
    X, y, info = load_dataset(ds)
    DATA[ds] = (X, y, info)
    print(f"✅ {ds:<9} n={info['n']} (raw {info['n_raw']})  d={info['n_features']}  "
          f"minoritas='{info['minority_label']}' pos={info['n_pos']} neg={info['n_neg']}  IR={info['ir']:.2f}")

# %%
# ============================================================
# 3. PREPROCESSING FOLD-AWARE + IMBALANCE SUBSAMPLING
# ============================================================
class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Membuang kolom dengan proporsi missing > threshold, dihitung HANYA dari data fit."""
    def __init__(self, threshold=0.4):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.keep_ = [c for c in X.columns if X[c].isna().mean() <= self.threshold]
        return self

    def transform(self, X):
        return X[self.keep_]


def make_preprocessor():
    num = SimpleImputer(strategy='median')
    cat = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    ct = ColumnTransformer([
        ('num', num, make_column_selector(dtype_include='number')),
        ('cat', cat, make_column_selector(dtype_exclude='number')),
    ], remainder='drop')
    # Catatan: scaling tidak dipakai karena XGBoost (tree-based) invarian terhadap transformasi monoton.
    # Winsorization IQR juga tidak dipakai: pada fitur biner/ordinal (mis. STDs di Cervical) IQR = 0
    # sehingga capping akan menghapus seluruh nilai minoritas fitur tersebut.
    return Pipeline([('drop_missing', HighMissingDropper(HIGH_MISSING_THRESHOLD)),
                     ('ct', ct),
                     ('var', VarianceThreshold(0.0))])


def scale_pos_weight_for(y):
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    ir = n_neg / max(n_pos, 1)
    return float(ir) if ir > USE_SCALE_POS_WEIGHT_IF_IR_GT else 1.0


def level_allowed(ds, target_ir, n_pos, n_neg, settings=None):
    """Level IR dijalankan jika layak (minoritas ≥ MIN_MINORITY) dan tidak dikeluarkan dari desain."""
    settings = settings or S
    if [ds, target_ir] in [list(e) for e in settings.get('exclude_levels', [])]:
        return False
    return subsample_to_ir(n_pos, n_neg, target_ir) is not None


def subsample_to_ir(n_pos, n_neg, target_ir):
    """Hitung (n_pos_baru, n_neg_baru) untuk mencapai target IR dengan mengurangi SATU kelas saja.
    target > IR asli  → kurangi minoritas (kelangkaan minoritas);
    target < IR asli  → kurangi mayoritas.  Return None jika tidak layak."""
    if target_ir is None:
        return n_pos, n_neg
    cur = n_neg / n_pos
    if target_ir >= cur:
        new_pos = int(round(n_neg / target_ir))
        new_neg = n_neg
    else:
        new_pos = n_pos
        new_neg = int(round(n_pos * target_ir))
    if new_pos < MIN_MINORITY or new_neg < new_pos:
        return None
    return new_pos, new_neg


def make_unit_data(X, y, target_ir, seed):
    """Subsampling ke target IR, lalu split 85:15 stratified. Test set dikunci."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    plan = subsample_to_ir(len(pos), len(neg), target_ir)
    if plan is None:
        return None
    idx = np.concatenate([rng.choice(pos, plan[0], replace=False), rng.choice(neg, plan[1], replace=False)])
    i_dev, i_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y[idx], random_state=seed)
    return (X.iloc[i_dev].reset_index(drop=True), X.iloc[i_te].reset_index(drop=True),
            y[i_dev], y[i_te], i_dev, i_te)   # i_dev, i_te = indeks baris asli (setelah dedup)


print('Kelayakan level imbalance per dataset (✓ = dijalankan):')
for ds, (X, y, info) in DATA.items():
    row = []
    for t in S['ir_targets']:
        plan = subsample_to_ir(info['n_pos'], info['n_neg'], t)
        lab = 'asli' if t is None else f'{t}:1'
        if plan and not level_allowed(ds, t, info['n_pos'], info['n_neg']):
            row.append(f'{lab}→✗ dikeluarkan'); continue
        row.append(f"{lab}→{'✓ pos=%d neg=%d' % plan if plan else '✗'}")
    print(f'  {ds:<9}', ' | '.join(row))

# %%
# ============================================================
# 4. DECODING, EVALUATOR (fitness = −mean weighted F1), BUDGET
# ============================================================
LB = np.array([p[1] for p in PARAM_SPACE], float)
UB = np.array([p[2] for p in PARAM_SPACE], float)


def decode(x):
    if x is None:                       # None = XGBoost default (tanpa tuning)
        return {}
    x = np.clip(np.asarray(x, float), 0, 1)
    raw = LB + x * (UB - LB)
    return {name: (int(round(v)) if typ == 'int' else float(v))
            for (name, _, _, typ), v in zip(PARAM_SPACE, raw)}


def make_xgb(params, spw, seed):
    return XGBClassifier(**params, scale_pos_weight=spw, tree_method='hist', n_jobs=XGB_N_JOBS,
                         random_state=seed, verbosity=0, eval_metric='logloss')


class CVEvaluator:
    """Satu 'objective evaluation' = 1 konfigurasi dievaluasi dengan skema CV (k × repeats).
    Preprocessing di-fit per training fold SEKALI di awal (tidak bergantung hyperparameter),
    jadi tetap bebas kebocoran tetapi jauh lebih cepat."""
    def __init__(self, X_dev, y_dev, n_splits, n_repeats, partition_seed, model_seed):
        self.model_seed = model_seed
        self.folds = []
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=partition_seed)
        for tr, va in rskf.split(X_dev, y_dev):
            pre = make_preprocessor().fit(X_dev.iloc[tr], y_dev[tr])
            self.folds.append((pre.transform(X_dev.iloc[tr]), y_dev[tr],
                               pre.transform(X_dev.iloc[va]), y_dev[va], scale_pos_weight_for(y_dev[tr])))
        self.n_fits = 0

    def __call__(self, x):
        params = decode(x)
        wf1, mf1, mcc = [], [], []
        for Xtr, ytr, Xva, yva, spw in self.folds:
            m = make_xgb(params, spw, self.model_seed).fit(Xtr, ytr)
            p = m.predict(Xva)
            wf1.append(f1_score(yva, p, average='weighted', zero_division=0))
            mf1.append(f1_score(yva, p, pos_label=1, zero_division=0))
            mcc.append(matthews_corrcoef(yva, p))
            self.n_fits += 1
        return dict(f=-float(np.mean(wf1)), wf1=float(np.mean(wf1)), wf1_sd=float(np.std(wf1)),
                    mf1=float(np.mean(mf1)), mcc=float(np.mean(mcc)),
                    fold_wf1=wf1, fold_mf1=mf1, fold_mcc=mcc)


def pack_evals(rs, Xpos=None):
    """Kumpulkan daftar hasil evaluator menjadi array mentah (float32) untuk disimpan di .npz."""
    out = {m: np.array([r[m] for r in rs], np.float32) for m in ('wf1', 'mf1', 'mcc')}
    for m in ('fold_wf1', 'fold_mf1', 'fold_mcc'):
        out[m] = np.array([r[m] for r in rs], np.float32)
    if Xpos is not None:
        out['X'] = np.asarray(Xpos, np.float32)
    return out


class FlippedEvaluator:
    """Ruang pencarian dicerminkan per dimensi: optimizer melihat u, model memakai x = 1 − u.
    Landscape identik, hanya 'arah' batas yang dibalik → menguji bias optimizer terhadap batas ruang."""
    def __init__(self, ev):
        self.ev = ev

    def __call__(self, u):
        return self.ev(1.0 - np.clip(np.asarray(u, float), 0, 1))

    def __getattr__(self, k):
        return getattr(self.ev, k)


class BudgetExhausted(Exception):
    pass


class Objective:
    """Membungkus evaluator: menghitung jumlah evaluasi, menghentikan pada budget, mencatat jejak."""
    def __init__(self, evaluator, budget):
        self.ev, self.budget, self.n = evaluator, budget, 0
        self.best_f, self.best_x = np.inf, None
        self.trace_best, self.X_log, self.f_log, self.r_log = [], [], [], []

    @property
    def progress(self):
        return min(self.n / self.budget, 1.0)

    def __call__(self, x):
        if self.n >= self.budget:
            raise BudgetExhausted
        x = np.clip(np.asarray(x, float), 0, 1)
        r = self.ev(x)
        self.n += 1
        if r['f'] < self.best_f:
            self.best_f, self.best_x = r['f'], x.copy()
            self.best_info = r
        self.trace_best.append(self.best_f)
        self.X_log.append(x.copy())
        self.f_log.append(r['f'])
        self.r_log.append(r)
        return r['f']


print('Evaluator siap ✅  contoh decode(0.5):', decode(np.full(D, 0.5)))

# %%

# %%
# ============================================================
# 4b. EVALUASI ULANG & HOLD-OUT (sama dengan v5.2) & ID UNIT
# ============================================================
def holdout_eval(x, X_dev, y_dev, X_te, y_te, seed):
    pre = make_preprocessor().fit(X_dev, y_dev)
    m = make_xgb(decode(x) if x is not None else {}, scale_pos_weight_for(y_dev), seed)
    m.fit(pre.transform(X_dev), y_dev)
    Xt = pre.transform(X_te); p = m.predict(Xt); pr = m.predict_proba(Xt)[:, 1]
    holdout_eval.last_proba = pr.astype(np.float32)        # disimpan ke .npz oleh pemanggil
    return dict(ho_wf1=f1_score(y_te, p, average='weighted', zero_division=0),
                ho_mf1=f1_score(y_te, p, pos_label=1, zero_division=0),
                ho_mcc=matthews_corrcoef(y_te, p), ho_bacc=balanced_accuracy_score(y_te, p),
                ho_auc=roc_auc_score(y_te, pr) if len(np.unique(y_te)) > 1 else np.nan)


def reevaluate(x, X_dev, y_dev, scheme, n_rep, seed):
    """Evaluasi ulang konfigurasi terpilih dengan partisi CV BARU → mengukur winner's curse."""
    k, rep = scheme
    rs = [CVEvaluator(X_dev, y_dev, k, rep, partition_seed=seed * 1000 + 500 + r, model_seed=seed)(x)
          for r in range(n_rep)]
    reevaluate.last_raw = pack_evals(rs)                  # skor per pengulangan & per fold
    return {m: float(np.mean([r[m] for r in rs])) for m in ('wf1', 'mf1', 'mcc')}




def unit_id(ds, ir, scheme_name, run):
    return f"{ds}_IR-{'orig' if ir is None else ir}_{scheme_name}_run{run:02d}"


# %%
# ============================================================
# T1. PENGATURAN BASELINE BAYESIAN OPTIMIZATION (TPE, Optuna)
# ============================================================
# Unit = unit eksperimen utama v5.2 (seed, subsampling, partisi CV pencarian, dan partisi evaluasi ulang identik),
# sehingga TPE dapat dipasangkan langsung dengan RS, BFO, HHO, SMA, SMAc, dan XGBoost default.
# Desain bawaan: 2 run per kondisi × 14 kondisi = 28 unit cv5x1 (ubah T_RUNS untuk memperluas, mis. range(10) = 140 unit).
T_SCHEME = 'cv5x1'
T_RUNS = [int(r) for r in os.environ.get('P2_TPE_RUNS', '0,1,2,3,4,5,6,7,8,9').split(',')]   # 140 unit (seperti di artikel)
T_BUDGET = S['opt_budget']                 # 300 evaluasi objektif, sama dengan optimizer lain
T_STARTUP = 20                             # trial acak awal TPE (= ukuran populasi metaheuristik)
T_VARIANTS = {'TPE': False, 'TPE_flip': True}   # TPE_flip: ruang dicerminkan (x → 1 − x) = kontrol positif flip test
T_UNITS = [(ds, ir, r) for ds in S['datasets'] for ir in S['ir_targets'] for r in T_RUNS
           if level_allowed(ds, ir, DATA[ds][2]['n_pos'], DATA[ds][2]['n_neg'])]
if os.environ.get('P2_TPE_N'):                                # batasi jumlah unit (cek runtime)
    T_UNITS = T_UNITS[:int(os.environ['P2_TPE_N'])]
if os.environ.get('P2_TPE_TEST'):                             # uji cepat kode (BUKAN untuk naskah)
    T_UNITS, T_BUDGET, T_STARTUP = T_UNITS[:1], 30, 10
    OUTPUT_DIR = OUTPUT_DIR + '_TEST'; os.makedirs(os.path.join(OUTPUT_DIR, 'units'), exist_ok=True)
MAIN_UNITS_DIR = os.environ.get('P2_MAIN_UNITS', os.path.join(OUTPUT_ROOT, CODE_VERSION, RUN_MODE, VARIANT_TAG, 'units'))
for _cand in ('../results/units', './results/units'):        # struktur repositori GitHub
    if not os.path.isdir(MAIN_UNITS_DIR) and os.path.isdir(_cand):
        MAIN_UNITS_DIR = _cand
print(f'Unit TPE: {len(T_UNITS)} | anggaran {T_BUDGET} | varian {list(T_VARIANTS)}')
print('Unit eksperimen utama (pembanding):', MAIN_UNITS_DIR, '→', 'ADA' if os.path.isdir(MAIN_UNITS_DIR) else 'TIDAK DITEMUKAN')

# %%
# ============================================================
# T2. TPE DENGAN EVALUATOR YANG SAMA
# ============================================================
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
HP_NAMES = [p[0] for p in PARAM_SPACE]


def run_tpe(search_ev, flipped, seed):
    """TPE (Bergstra et al., 2011) di ruang ter-normalisasi [0,1]^9, seperti optimizer lain.
    flipped=True: TPE melihat u, model memakai x = 1 − u (flip test). TPE dengan prior seragam seharusnya invarian."""
    obj = Objective(FlippedEvaluator(search_ev) if flipped else search_ev, T_BUDGET)
    sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=T_STARTUP, multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    def f(trial):
        u = np.array([trial.suggest_float(n, 0.0, 1.0) for n in HP_NAMES])
        return obj(u)
    try:
        study.optimize(f, n_trials=T_BUDGET, catch=())
    except BudgetExhausted:
        pass
    U = np.array(obj.X_log)                                  # posisi yang dilihat TPE (ruang internal)
    X = 1 - U if flipped else U                              # ruang asli
    fl = np.array(obj.f_log)
    top = X[np.argsort(fl)[:T_STARTUP]]                      # 20 konfigurasi terbaik ≈ "populasi akhir"
    c = top.mean(axis=0)
    best_x = 1 - obj.best_x if flipped else obj.best_x
    beh = dict(n_evals=obj.n, final_centroid=c.round(4).tolist(),
               final_real_frac_at_lower=float(np.mean(c < 0.02)), final_real_frac_at_upper=float(np.mean(c > 0.98)),
               last100_centroid=X[-100:].mean(axis=0).round(4).tolist())
    tb = np.array(obj.trace_best); f0, fb = tb[0], tb[-1]
    beh['evals_to_95pct'] = int(np.argmax(tb <= f0 - 0.95 * (f0 - fb)) + 1) if f0 > fb else 1
    return obj, best_x, X, beh


def t_unit_id(ds, ir, run):
    return unit_id(ds, ir, T_SCHEME, run)


def run_tpe_unit(ds, ir, run, verbose=True):
    uid = t_unit_id(ds, ir, run)
    path = os.path.join(OUTPUT_DIR, 'units', uid + '.json')
    if os.path.exists(path):
        if verbose: print(f'⏭  {uid} sudah ada (checkpoint), dilewati')
        return json.load(open(path))
    X, y, _ = DATA[ds]
    seed = SEED_BASE + run
    X_dev, X_te, y_dev, y_te, i_dev, i_te = make_unit_data(X, y, ir, seed)
    scheme = S['cv_schemes'][T_SCHEME]
    search_ev = CVEvaluator(X_dev, y_dev, *scheme, partition_seed=seed, model_seed=seed)    # sama dengan v5.2
    res = dict(unit=uid, dataset=ds, ir_target=ir, scheme=T_SCHEME, run=run, seed=seed,
               optuna_version=optuna.__version__, optimizers={})
    RAW = {}
    t0 = time.time()
    for k, (variant, flipped) in enumerate(T_VARIANTS.items()):
        _t = time.time()
        obj, best_x, Xreal, beh = run_tpe(search_ev, flipped, seed * 100 + 7)   # seed SAMA untuk asli & dicerminkan
        reev = reevaluate(best_x, X_dev, y_dev, scheme, S['reeval_repeats'], seed)  # partisi sama dengan v5.2
        ho = holdout_eval(best_x, X_dev, y_dev, X_te, y_te, seed)
        RAW[f'opt_{variant}_X'] = Xreal.astype(np.float32)
        RAW[f'opt_{variant}_wf1'] = np.array([r['wf1'] for r in obj.r_log], np.float32)
        RAW[f'opt_{variant}_mf1'] = np.array([r['mf1'] for r in obj.r_log], np.float32)
        res['optimizers'][variant] = dict(flipped=flipped, behavior=beh, best_x=best_x.tolist(), best_params=decode(best_x),
                                          best_cv_wf1=-obj.best_f, reeval_cv_wf1=reev['wf1'], reeval_cv_mf1=reev['mf1'],
                                          reeval_cv_mcc=reev['mcc'], winners_curse=-obj.best_f - reev['wf1'], **ho,
                                          trace_best=[-v for v in obj.trace_best], search_sec=time.time() - _t)
        if verbose:
            o = res['optimizers'][variant]
            print(f"   {variant:<9} eval={beh['n_evals']} bestCV={o['best_cv_wf1']:.4f} reeval={o['reeval_cv_wf1']:.4f} "
                  f"WC={o['winners_curse']:.4f} batas bawah={beh['final_real_frac_at_lower']:.2f} atas={beh['final_real_frac_at_upper']:.2f} "
                  f"({o['search_sec']/60:.1f} mnt)")
    res['runtime_sec'] = time.time() - t0
    res['meta'] = dict(host=socket.gethostname(), finished_at=datetime.datetime.now().isoformat(timespec='seconds'),
                       libs={**RUN_META['libs'], 'optuna': optuna.__version__})
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'units', uid + '.npz'), **RAW)
    with open(path + '.tmp', 'w') as f:
        json.dump(res, f, default=float)
    os.replace(path + '.tmp', path)
    if verbose: print(f'✅ {uid}  {res["runtime_sec"]/60:.1f} menit')
    return res

# %%
# ============================================================
# T3. JALANKAN (checkpoint per unit; aman dijalankan ulang setelah Colab terputus)
# ============================================================
T_START = time.time()
if N_WORKERS <= 1:
    for i, (ds, ir, r) in enumerate(T_UNITS, 1):
        print(f'\n[{i}/{len(T_UNITS)}] {t_unit_id(ds, ir, r)}')
        run_tpe_unit(ds, ir, r)
else:
    from joblib import Parallel, delayed
    todo = [t for t in T_UNITS if not os.path.exists(os.path.join(OUTPUT_DIR, 'units', t_unit_id(*t) + '.json'))]
    print(f'Paralel: {N_WORKERS} worker, {len(todo)} unit belum selesai')

    def _job(t):
        r = run_tpe_unit(*t, verbose=False)
        return t_unit_id(*t), r['runtime_sec']

    for uid, sec in Parallel(n_jobs=N_WORKERS, backend='loky', return_as='generator_unordered')(
            delayed(_job)(t) for t in todo):
        print(f'✅ {uid}  {sec/60:.1f} menit | total {(time.time() - T_START)/60:.1f} menit', flush=True)
print(f'Selesai dalam {(time.time() - T_START)/60:.1f} menit')

# %%
# ============================================================
# T4. ANALISIS: TPE vs RS / default / metaheuristik, flip test TPE (kontrol positif), winner's curse
# ============================================================
from scipy import stats as _st
LANG = os.environ.get('P2_LANG', 'en')
L = lambda i, e: e if LANG == 'en' else i
rows = []
for ds, ir, r in T_UNITS:
    uid = t_unit_id(ds, ir, r)
    p = os.path.join(OUTPUT_DIR, 'units', uid + '.json')
    if not os.path.exists(p):
        continue
    u = json.load(open(p))
    base = dict(unit=uid, dataset=ds, cond=f"{ds}|{'asli' if ir is None else ir}", run=r)
    for name, o in u['optimizers'].items():
        rows.append(dict(**base, metode=name, reeval_cv_wf1=o['reeval_cv_wf1'], reeval_cv_mf1=o['reeval_cv_mf1'],
                         winners_curse=o['winners_curse'], ho_wf1=o['ho_wf1'],
                         batas_bawah=o['behavior']['final_real_frac_at_lower'], batas_atas=o['behavior']['final_real_frac_at_upper'],
                         evals_to_95pct=o['behavior']['evals_to_95pct']))
    mp = os.path.join(MAIN_UNITS_DIR, uid + '.json')
    if os.path.exists(mp):
        m = json.load(open(mp))
        for name in ['RS', 'BFO', 'BFO_flip', 'HHO', 'HHO_flip', 'SMA', 'SMA_flip', 'SMAc']:
            o = m['optimizers'].get(name)
            if o:
                b = o['behavior']
                rows.append(dict(**base, metode=name, reeval_cv_wf1=o['reeval_cv_wf1'], reeval_cv_mf1=o['reeval_cv_mf1'],
                                 winners_curse=o['winners_curse'], ho_wf1=o['ho_wf1'],
                                 batas_bawah=b.get('final_real_frac_at_lower'), batas_atas=b.get('final_real_frac_at_upper'),
                                 evals_to_95pct=b.get('evals_to_95pct')))
        d = m['default']
        rows.append(dict(**base, metode='default', reeval_cv_wf1=d['reeval_cv_wf1'], reeval_cv_mf1=d['reeval_cv_mf1'],
                         winners_curse=np.nan, ho_wf1=d['ho_wf1'], batas_bawah=np.nan, batas_atas=np.nan, evals_to_95pct=np.nan))
TL = pd.DataFrame(rows)
TL.to_csv(os.path.join(OUTPUT_DIR, 'tpe_long.csv'), index=False)
print(f'{TL.unit.nunique()} unit dimuat; metode: {sorted(TL.metode.unique())}')

W = TL.pivot_table(index=['unit', 'dataset', 'cond'], columns='metode', values='reeval_cv_wf1').reset_index()


def paired(a, b, label):
    if a not in W or b not in W:
        return None
    g = W[['cond', a, b]].dropna(); d = g[a] - g[b]
    if len(d) < 3:
        return None
    nz = d[d != 0]
    rk = _st.rankdata(np.abs(nz)); rrb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if len(nz) else np.nan
    cm = d.groupby(g['cond']).mean()
    return {L('Perbandingan', 'Comparison'): label, 'n': len(d), L('Median Δ wF1', 'Median Δ wF1'): round(float(d.median()), 4),
            'Δ > 0': f'{int((d > 0).sum())}/{len(d)}', 'r': round(float(rrb), 2),
            'p Wilcoxon': float(_st.wilcoxon(nz).pvalue) if len(nz) > 1 else np.nan,
            L('Kondisi Δ̄ > 0', 'Conditions mean Δ > 0'): f'{int((cm > 0).sum())}/{len(cm)}'}


tests = [paired('TPE', 'TPE_flip', L('TPE: asli − dicerminkan (kontrol positif flip test)', 'TPE: original − mirrored (positive control)')),
         paired('SMA', 'SMA_flip', L('SMA: asli − dicerminkan (unit yang sama)', 'SMA: original − mirrored (same units)')),
         paired('HHO', 'HHO_flip', L('HHO: asli − dicerminkan (unit yang sama)', 'HHO: original − mirrored (same units)')),
         paired('TPE', 'RS', 'TPE − RS'), paired('TPE', 'default', L('TPE − XGBoost default', 'TPE − default XGBoost')),
         paired('TPE', 'SMA', 'TPE − SMA'), paired('TPE', 'SMA_flip', L('TPE − SMA (dicerminkan)', 'TPE − SMA (mirrored)')),
         paired('TPE', 'HHO', 'TPE − HHO'), paired('TPE', 'BFO', 'TPE − BFO'), paired('TPE', 'SMAc', 'TPE − SMAc')]
T_T = pd.DataFrame([t for t in tests if t])
if len(T_T):
    T_T['p Holm'] = np.nan
    ps = T_T['p Wilcoxon'].values; order = np.argsort(ps); m_ = len(ps); run_ = 0; adj = np.empty(m_)
    for k, i in enumerate(order):
        run_ = max(run_, (m_ - k) * ps[i]); adj[i] = min(1.0, run_)
    T_T['p Holm'] = [('< 0.001' if p < 0.001 else f'{p:.3f}') for p in adj]
    T_T['p Wilcoxon'] = [('< 0.001' if p < 0.001 else f'{p:.3f}') for p in T_T['p Wilcoxon']]
T_T.to_csv(os.path.join(OUTPUT_DIR, L('tpe_uji_berpasangan.csv', 'TableS4_tpe_baseline.csv')), index=False)
display(T_T) if 'display' in globals() else print(T_T.to_string(index=False))

summ = TL.groupby('metode').agg(n=('unit', 'nunique'), reeval_wf1=('reeval_cv_wf1', 'mean'), winners_curse=('winners_curse', 'mean'),
                                batas_bawah=('batas_bawah', 'mean'), batas_atas=('batas_atas', 'mean'),
                                evals_to_95pct=('evals_to_95pct', 'median')).round(4)
summ = summ.reindex([m for m in ['default', 'RS', 'TPE', 'TPE_flip', 'BFO', 'BFO_flip', 'HHO', 'HHO_flip', 'SMA', 'SMA_flip', 'SMAc'] if m in summ.index])
summ.to_csv(os.path.join(OUTPUT_DIR, L('tpe_ringkasan.csv', 'tpe_summary.csv')))
display(summ) if 'display' in globals() else print(summ.to_string())

# %%
# ============================================================
# T5. GAMBAR SUPLEMEN: selisih wF1 evaluasi ulang terhadap RS (asli vs dicerminkan), termasuk TPE
# ============================================================
import matplotlib.pyplot as plt
COL = {'TPE': '#6250d6', 'BFO': '#2a78d6', 'HHO': '#eb6834', 'SMA': '#1baf7a', 'SMAc': '#eda100'}
order = [m for m in ['TPE', 'TPE_flip', 'BFO', 'BFO_flip', 'HHO', 'HHO_flip', 'SMA', 'SMA_flip', 'SMAc'] if m in W]
rng_ = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for i, mth in enumerate(order[::-1]):
    d = (W[mth] - W['RS']).dropna().values
    if not len(d):
        continue
    bs = [np.median(rng_.choice(d, len(d))) for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5]); md = np.median(d)
    b = mth.replace('_flip', ''); hollow = mth.endswith('_flip')
    ax.errorbar(md, i, xerr=[[md - lo], [hi - md]], fmt='none', ecolor=COL[b], lw=1.6, capsize=2)
    ax.scatter(md, i, s=40, color='white' if hollow else COL[b], edgecolor=COL[b], lw=1.6, zorder=3)
ax.axvline(0, color='#898781', lw=0.9)
ax.set_yticks(range(len(order))); ax.set_yticklabels([m.replace('_flip', L(' (dicerminkan)', ' (mirrored)')) for m in order[::-1]])
ax.set_xlabel(L('Δ median wF1 evaluasi ulang vs Random Search [95% CI]', 'Median Δ re-evaluated wF1 vs Random Search [95% CI]') +
              f"\n(cv5x1, n = {W.dropna(subset=['TPE', 'RS']).shape[0]} {L('unit berpasangan', 'paired units')})")
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUTPUT_DIR, L('GambarS_tpe_vs_rs', 'FigS2_tpe_vs_random_search') + '.' + ext), dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.show()
print('✅ Hasil →', OUTPUT_DIR)
