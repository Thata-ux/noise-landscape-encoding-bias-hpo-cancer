# %% [markdown]
# # Paper 2 — Noise-Aware Fitness Landscape & Search Behavior Analysis
# ## XGBoost Hyperparameter Optimization on Small Imbalanced Cancer Data
#
# **Rantai yang diuji:** `noise evaluasi + imbalance → fitness landscape → search behavior → optimization reliability`
#
# | RQ | Mata rantai | Modul |
# |---|---|---|
# | RQ1 | noise CV & imbalance → noise/SNR landscape | Bagian 5 (`noise_analysis`) |
# | RQ2 | landscape → karakteristik (FDC, ruggedness, neutralitas, importance) | Bagian 6 (`fla_analysis`) |
# | RQ2–3 | landscape → perilaku BFO / HHO / SMA (eksplorasi–eksploitasi) | Bagian 7–8 |
# | RQ3 | perilaku → reliabilitas (winner's curse, gap CV–holdout, konsistensi) | Bagian 9 |
#
# **Faktor eksperimen (dimanipulasi):** dataset (4) × target imbalance ratio × skema CV (noise) × run.
# BFO, HHO, dan SMA dipakai sebagai *probe* perilaku pencarian — **bukan** untuk mencari optimizer terbaik.
# Random Search dipakai sebagai referensi tanpa struktur pencarian.
#
# **Cara pakai:** atur `RUN_MODE` di Bagian 1. `'pilot'` = Cervical saja, budget kecil, untuk cek runtime &
# sinyal efek. `'full'` = semua dataset sesuai protokol. Setiap unit eksperimen disimpan sebagai JSON
# (checkpoint), jadi notebook bisa dijalankan ulang setelah Colab terputus tanpa mengulang unit yang sudah selesai.
#
# **Keputusan desain final (disepakati, v5.2):** fitness = weighted F1 (sesuai proposal; F1 minoritas & MCC
# sebagai metrik sekunder, juga untuk analisis landscape); manipulasi imbalance dibingkai sebagai dua seri
# komposisi kelas (minoritas tetap vs mayoritas tetap); level Lung 6:1 dikeluarkan (≈ duplikat Lung asli,
# 38/228 vs 38/238); Bayesian Optimization tidak termasuk paper 2 (ditempatkan di paper 3).
#
# **Baru di v5.1:** snapshot populasi per iterasi, waktu komputasi per komponen, parameter optimizer
# tercatat, folder terpisah untuk pilot vs full, dan pengaman agar pengaturan tidak berubah di tengah running.
#
# **Baru di v5 (tanpa mengubah desain eksperimen v4):**
# - **Seluruh data mentah disimpan** per unit dalam file `.npz` (terkompresi): matriks noise, semua sampel
#   landscape & random walk, SETIAP evaluasi optimizer (posisi + skor per fold + wF1/F1-minoritas/MCC),
#   skor evaluasi ulang per pengulangan, probabilitas prediksi holdout, dan indeks baris dev/test.
#   → Analisis baru untuk artikel/revisi reviewer cukup dihitung ulang dari file, TANPA running ulang.
# - Baseline XGBoost default kini juga dievaluasi dengan CV & evaluasi ulang (bukan hanya holdout).
# - Metadata reproduktibilitas (versi library, mesin, waktu, konfigurasi lengkap) di setiap unit.
# - Pemotongan pekerjaan (`CHUNK`) & dukungan Kaggle, agar eksperimen bisa dibagi ke beberapa mesin/sesi.
#
# **Baru di v4:**
# 1. **Uji invariansi encoding (flip test):** BFO/HHO/SMA dijalankan ulang pada ruang yang dicerminkan
#    (x → 1 − x) dengan seed yang sama. Jika kinerja turun setelah dicerminkan, keunggulan optimizer berasal
#    dari bias terhadap batas ruang, bukan dari kemampuan mencari. Hasil muncul sebagai optimizer `*_flip`.
# 2. **F1 kelas minoritas & MCC** pada evaluasi CV dan evaluasi ulang (bukan hanya weighted F1).
# 3. **Baseline pojok atas** (semua hyperparameter = batas atas) sebagai pasangan baseline pojok bawah.
# 4. **Eksekusi paralel** antar-unit (`N_WORKERS`), untuk laptop/PC multi-core.
#
# **Menjalankan di laptop (file .py):**
# `P2_RUN_MODE=full P2_WORKERS=6 P2_DATA_DIR=./data P2_OUTPUT_DIR=./outputs python paper2_noisefla_searchbehavior_v4.py`
# (Windows PowerShell: `$env:P2_RUN_MODE="full"; $env:P2_WORKERS="6"; python paper2_noisefla_searchbehavior_v4.py`)

# %%
# ============================================================
# 0. INSTALL & IMPORT
# ============================================================
# Di Colab, jalankan sekali (hapus tanda # di baris berikut):
# !pip install -q xgboost

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
OUTPUT_DIR = os.path.join(OUTPUT_DIR, CODE_VERSION, RUN_MODE, VARIANT_TAG)
os.makedirs(os.path.join(OUTPUT_DIR, 'units'), exist_ok=True)

# Kaggle: lanjutkan dari output versi sebelumnya yang ditambahkan sebagai input notebook
if ON_KAGGLE:
    _n_copied = 0
    for _p in glob.glob(f'/kaggle/input/**/{CODE_VERSION}/{RUN_MODE}/{VARIANT_TAG}/units/*', recursive=True):
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
print(f"RUN_MODE = {RUN_MODE} | output → {OUTPUT_DIR}")
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
# ============================================================
# 5. RQ1 — ANALISIS NOISE EVALUASI
# ============================================================
def noise_analysis(X_dev, y_dev, scheme, n_configs, n_repeats, seed):
    """Setiap konfigurasi dievaluasi dengan n_repeats partisi CV berbeda.
    var_within  = noise satu evaluasi (variasi karena partisi)
    var_between = variasi 'sinyal' antar konfigurasi (dikoreksi noise)."""
    k, rep = scheme
    rng = np.random.default_rng(seed)
    configs = rng.random((n_configs, D))
    F = np.zeros((n_configs, n_repeats))
    FM = {m: np.zeros((n_configs, n_repeats)) for m in ('mf1', 'mcc')}
    evals = [CVEvaluator(X_dev, y_dev, k, rep, partition_seed=seed * 1000 + r, model_seed=seed)
             for r in range(n_repeats)]
    for i, x in enumerate(configs):
        for r, ev in enumerate(evals):
            out_r = ev(x)
            F[i, r] = out_r['wf1']
            for m in FM:
                FM[m][i, r] = out_r[m]
    raw = dict(noise_configs=configs.astype(np.float32), noise_wf1=F.astype(np.float32),
               noise_mf1=FM['mf1'].astype(np.float32), noise_mcc=FM['mcc'].astype(np.float32))
    var_within = float(F.var(axis=1, ddof=1).mean())
    var_means = float(F.mean(axis=1).var(ddof=1))
    var_signal = max(var_means - var_within / n_repeats, 0.0)
    icc = var_signal / (var_signal + var_within) if (var_signal + var_within) > 0 else np.nan
    taus = [stats.kendalltau(F[:, a], F[:, b])[0] for a, b in itertools.combinations(range(n_repeats), 2)]
    return dict(sigma_noise=math.sqrt(var_within), sigma_signal=math.sqrt(var_signal),
                snr=(var_signal / var_within if var_within > 0 else np.inf), icc=icc,
                rank_stability_tau=float(np.nanmean(taus)), mean_wf1=float(F.mean()),
                n_fits=sum(e.n_fits for e in evals)), raw

# %%
# ============================================================
# 6. RQ2 — FITNESS LANDSCAPE ANALYSIS
# ============================================================
def fla_analysis(evaluator, n_samples, walk_length, seed, sigma_noise):
    rng = np.random.default_rng(seed + 7)
    # (a) sampel acak global
    Xs = rng.random((n_samples, D))
    rs_samp = [evaluator(x) for x in Xs]
    fs = np.array([r['f'] for r in rs_samp])                  # minimisasi
    best = Xs[np.argmin(fs)]
    dist = np.linalg.norm(Xs - best, axis=1)
    fdc = float(stats.pearsonr(fs, dist)[0]) if np.std(fs) > 0 else np.nan
    # dispersion (Lunacek & Whitley): top-set vs semua; negatif → struktur 'funnel'
    n_top = max(3, int(TOP_FRACTION * n_samples))
    top = Xs[np.argsort(fs)[:n_top]]
    dispersion = float(pdist(top).mean() - pdist(Xs).mean())
    # proxy multimodalitas: jumlah klaster di antara konfigurasi terbaik
    n_clusters = int(fcluster(linkage(top, 'average'), t=0.35 * math.sqrt(D), criterion='distance').max())
    # struktur yang dapat dipelajari: R² OOB surrogate RF (rendah → landscape didominasi noise)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, oob_score=True, random_state=seed).fit(Xs, fs)
    imp = permutation_importance(rf, Xs, fs, n_repeats=5, random_state=seed).importances_mean
    imp = np.clip(imp, 0, None); imp = imp / imp.sum() if imp.sum() > 0 else imp
    # (b) random walk → ruggedness & neutralitas
    x = rng.random(D); walk_X = [x.copy()]; rs_walk = [evaluator(x)]
    for _ in range(walk_length - 1):
        x = np.clip(x + rng.normal(0, WALK_STEP_SD, D), 0, 1)
        walk_X.append(x.copy()); rs_walk.append(evaluator(x))
    walk = np.array([r['f'] for r in rs_walk])
    raw = {**{f'fla_sample_{k}': v for k, v in pack_evals(rs_samp, Xs).items()},
           **{f'fla_walk_{k}': v for k, v in pack_evals(rs_walk, walk_X).items()}}
    # Jika seluruh walk bernilai sama (plateau total), autokorelasi tidak terdefinisi → dicatat walk_flat=True.
    # Koreksi atenuasi noise TIDAK dipakai: dengan partisi CV tetap, 'noise' bersifat deterministik per titik
    # dan tidak independen, sehingga koreksi klasik menghasilkan nilai > 1 (terlihat di pilot).
    walk_flat = bool(np.std(walk) == 0)
    rho1 = np.nan if walk_flat else float(pd.Series(walk).autocorr(1))
    corr_len = float(-1 / math.log(abs(rho1))) if (not walk_flat and 0 < abs(rho1) < 1) else np.nan
    steps = np.abs(np.diff(walk))
    neutral_ratio = float(np.mean(steps < sigma_noise)) if sigma_noise > 0 else float(np.mean(steps == 0))
    flat_ratio = float(np.mean(steps < 1e-12))
    out = dict(fdc=fdc, dispersion=dispersion, top_clusters=n_clusters, rf_oob_r2=float(rf.oob_score_),
               rho1=rho1, walk_flat=walk_flat, corr_length=corr_len,
               neutral_ratio=neutral_ratio, flat_ratio=flat_ratio,
               f_range=float(fs.max() - fs.min()), f_sd=float(fs.std()))
    out.update({f'imp_{p[0]}': float(v) for p, v in zip(PARAM_SPACE, imp)})
    return out, raw

# %%
# ============================================================
# 7. OPTIMIZER (probe perilaku) — semua di ruang [0,1]^D, berbasis budget evaluasi
# ============================================================
class Tracker:
    """Mencatat diversity populasi tiap iterasi (Hussain et al., 2019: dimension-wise diversity)."""
    def __init__(self, obj):
        self.obj, self.evals, self.div = obj, [], []
        self.last_pop = None
        self.counters = {}

    def __call__(self, pop):
        self.last_pop = np.array(pop, copy=True)
        self.pops = getattr(self, 'pops', [])
        self.pops.append(self.last_pop.astype(np.float32))      # snapshot populasi per iterasi
        med = np.median(pop, axis=0)
        self.evals.append(self.obj.n)
        self.div.append(float(np.mean(np.abs(pop - med))))


def run_rs(obj, rng, pop_size, track):
    while True:
        obj(rng.random(D))


def run_bfo(obj, rng, pop_size, track, C=0.10, Ns=4, Nre=2, Ped=0.25, ed_cycles_target=2):
    """BFO (Passino, 2002) tanpa swarming: tumble+swim, reproduksi berbasis health, eliminasi-dispersal.
    Nc diskalakan terhadap budget agar siklus hidup lengkap (chemotaxis → reproduksi → eliminasi-dispersal)
    terjadi minimal `ed_cycles_target` kali; jika tidak, yang diukur bukan BFO sesungguhnya.
    Asumsi biaya satu langkah chemotaxis ≈ 1,5 × pop (tumble + rata-rata swim)."""
    Nc = max(2, int(obj.budget / (ed_cycles_target * Nre * pop_size * 1.5)))
    track.counters.update(bfo_Nc=Nc, bfo_reproductions=0, bfo_dispersals=0)
    P = rng.random((pop_size, D)); J = np.array([obj(x) for x in P]); track(P)
    while True:
        for _ in range(Nre):
            health = np.zeros(pop_size)
            for _ in range(Nc):
                for i in range(pop_size):
                    J_last = J[i]
                    phi = rng.standard_normal(D); phi /= np.linalg.norm(phi) + 1e-12
                    P[i] = np.clip(P[i] + C * phi, 0, 1); J[i] = obj(P[i])
                    m = 0
                    while m < Ns and J[i] < J_last:
                        J_last = J[i]
                        P[i] = np.clip(P[i] + C * phi, 0, 1); J[i] = obj(P[i]); m += 1
                    health[i] += J[i]
                track(P)
            order = np.argsort(health)[: pop_size // 2]                 # reproduksi
            P = np.vstack([P[order], P[order]]); J = np.concatenate([J[order], J[order]])
            track.counters['bfo_reproductions'] += 1
        for i in range(pop_size):                                       # eliminasi-dispersal
            if rng.random() < Ped:
                P[i] = rng.random(D); J[i] = obj(P[i])
        track.counters['bfo_dispersals'] += 1
        track(P)


def _levy(rng, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return 0.01 * rng.standard_normal(D) * sig / np.abs(rng.standard_normal(D)) ** (1 / beta)


def run_hho(obj, rng, pop_size, track):
    """HHO (Heidari et al., 2019). Energi melarikan diri meluruh terhadap fraksi budget terpakai."""
    P = rng.random((pop_size, D)); F = np.array([obj(x) for x in P]); track(P)
    while True:
        b = np.argmin(F); Xr, Fr = P[b].copy(), F[b]
        for i in range(pop_size):
            E = 2 * (2 * rng.random() - 1) * (1 - obj.progress)
            if abs(E) >= 1:                                              # eksplorasi
                if rng.random() >= 0.5:
                    Xk = P[rng.integers(pop_size)]
                    P[i] = Xk - rng.random() * np.abs(Xk - 2 * rng.random() * P[i])
                else:
                    P[i] = (Xr - P.mean(axis=0)) - rng.random() * rng.random(D)
                P[i] = np.clip(P[i], 0, 1); F[i] = obj(P[i])
            else:                                                        # eksploitasi
                r, Jp = rng.random(), 2 * (1 - rng.random())
                if r >= 0.5 and abs(E) >= 0.5:                           # soft besiege
                    P[i] = np.clip((Xr - P[i]) - E * np.abs(Jp * Xr - P[i]), 0, 1); F[i] = obj(P[i])
                elif r >= 0.5:                                           # hard besiege
                    P[i] = np.clip(Xr - E * np.abs(Xr - P[i]), 0, 1); F[i] = obj(P[i])
                else:                                                    # rapid dives (greedy)
                    ref = P[i] if abs(E) >= 0.5 else P.mean(axis=0)
                    Y = np.clip(Xr - E * np.abs(Jp * Xr - ref), 0, 1); fY = obj(Y)
                    if fY < F[i]:
                        P[i], F[i] = Y, fY
                    else:
                        Z = np.clip(Y + rng.random(D) * _levy(rng), 0, 1); fZ = obj(Z)
                        if fZ < F[i]:
                            P[i], F[i] = Z, fZ
            if F[i] < Fr:
                Xr, Fr = P[i].copy(), F[i]
        track(P)


def run_sma(obj, rng, pop_size, track, z=0.03, centered=False):
    """SMA (Li et al., 2020), implementasi sesuai rumus asli.
    Catatan: p = tanh|S_i − DF| bergantung pada skala fitness; pada skala F1 (selisih ~0.01)
    p kecil sehingga cabang vc·X mendominasi — perilaku ini justru salah satu yang diukur."""
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
        for i in range(pop_size):
            if rng.random() < z:
                P[i] = rng.random(D)
            else:
                p = math.tanh(abs(F[i] - best_f))
                vb, vc = rng.uniform(-a, a, D), rng.uniform(-bcoef, bcoef, D)
                A, B = rng.integers(pop_size, size=2)
                r = rng.random(D)
                # centered=True: kontraksi vc·X dilakukan terhadap PUSAT ruang (u = 2x − 1), bukan titik 0.
                # Varian diagnostik untuk memisahkan perilaku SMA dari artefak normalisasi ruang.
                contract = (vc * (2 * P[i] - 1) + 1) / 2 if centered else vc * P[i]
                P[i] = np.where(r < p, best_x + vb * (W[i] * P[A] - P[B]), contract)
            P[i] = np.clip(P[i], 0, 1); F[i] = obj(P[i])
            if F[i] < best_f:
                best_x, best_f = P[i].copy(), F[i]
        track(P)


def _opt_params():
    out = {}
    for n, f in [('BFO', run_bfo), ('HHO', run_hho), ('SMA', run_sma)]:
        out[n] = {k: v.default for k, v in inspect.signature(f).parameters.items()
                  if v.default is not inspect.Parameter.empty}
    return out


RUN_META['optimizer_params'] = _opt_params()
with open(_cfg_path, 'w') as _f:
    json.dump(RUN_META, _f, indent=1, default=str)

OPTIMIZERS = {'RS': run_rs, 'BFO': run_bfo, 'HHO': run_hho, 'SMA': run_sma,
              'SMAc': lambda o, r, n, t: run_sma(o, r, n, t, centered=True)}   # opsional: tambahkan 'SMAc' ke S['optimizers']

# %%
# ============================================================
# 8. METRIK PERILAKU (RQ2) & RELIABILITAS (RQ3)
# ============================================================
def behavior_metrics(tracker, obj, flipped=False):
    out = dict(n_evals=obj.n)
    tb = np.array(obj.trace_best)
    f0, fb = tb[0], tb[-1]
    if f0 > fb:
        thr = f0 - 0.95 * (f0 - fb)
        out['evals_to_95pct'] = int(np.argmax(tb <= thr) + 1)
    else:
        out['evals_to_95pct'] = 1
    impr = np.r_[True, np.diff(tb) < 0]
    gaps = np.diff(np.where(impr)[0].tolist() + [len(tb)])
    out['n_improvements'] = int(impr.sum() - 1)
    out['longest_stagnation_frac'] = float(gaps.max() / len(tb))
    if tracker.div:
        div = np.array(tracker.div); dmax = div.max() if div.max() > 0 else 1.0
        xpl = 100 * div / dmax
        out.update(xpl_mean=float(xpl.mean()), xpl_final=float(xpl[-1]),
                   xpl_auc=float(_trapz(xpl, tracker.evals) / max(tracker.evals[-1], 1) if len(xpl) > 1 else xpl[0]),
                   evals_xpl_below_20=int(tracker.evals[int(np.argmax(xpl < 20))]) if (xpl < 20).any() else None,
                   div_initial=float(div[0]), div_final=float(div[-1]))
    if tracker.last_pop is not None:
        c = tracker.last_pop.mean(axis=0)
        out['final_centroid_mean'] = float(c.mean())
        out['final_frac_dims_at_lower'] = float(np.mean(c < 0.02))   # deteksi kolaps ke batas bawah
        out['final_frac_dims_at_upper'] = float(np.mean(c > 0.98))
        out['final_centroid'] = c.round(4).tolist()
        # posisi di ruang hyperparameter SEBENARNYA (untuk optimizer *_flip, batas internal 0 = batas atas asli)
        cr = 1 - c if flipped else c
        out['final_real_frac_at_lower'] = float(np.mean(cr < 0.02))
        out['final_real_frac_at_upper'] = float(np.mean(cr > 0.98))
    out.update(tracker.counters)
    return out


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


def corner_eval(x, search_ev, X_dev, y_dev, X_te, y_te, scheme, seed, raw=None, tag=None):
    r, rv = search_ev(x), reevaluate(x, X_dev, y_dev, scheme, S['reeval_repeats'], seed)
    if raw is not None:
        raw.update({f'{tag}_search_{k}': v for k, v in pack_evals([r]).items()})
        raw.update({f'{tag}_reeval_{k}': v for k, v in reevaluate.last_raw.items()})
    out = dict(cv_wf1=r['wf1'], cv_mf1=r['mf1'], cv_mcc=r['mcc'],
                reeval_cv_wf1=rv['wf1'], reeval_cv_mf1=rv['mf1'], reeval_cv_mcc=rv['mcc'],
                **holdout_eval(x, X_dev, y_dev, X_te, y_te, seed))
    if raw is not None:
        raw[f'{tag}_holdout_proba'] = holdout_eval.last_proba
    return out

# %%
# ============================================================
# 9. RUNNER SATU UNIT EKSPERIMEN (dataset × IR × skema CV × run) + CHECKPOINT
# ============================================================
def unit_id(ds, ir, scheme_name, run):
    return f"{ds}_IR-{'orig' if ir is None else ir}_{scheme_name}_run{run:02d}"


def run_unit(ds, ir, scheme_name, run, verbose=True):
    uid = unit_id(ds, ir, scheme_name, run)
    path = os.path.join(OUTPUT_DIR, 'units', uid + '.json')
    if os.path.exists(path):
        if verbose: print(f'⏭  {uid} sudah ada (checkpoint), dilewati')
        return json.load(open(path))
    X, y, _ = DATA[ds]
    seed = SEED_BASE + run
    split = make_unit_data(X, y, ir, seed)
    if split is None:
        return None
    X_dev, X_te, y_dev, y_te, i_dev, i_te = split
    scheme = S['cv_schemes'][scheme_name]
    t0 = time.time()
    RAW = dict(idx_dev=np.asarray(i_dev, np.int32), idx_test=np.asarray(i_te, np.int32),
               y_dev=np.asarray(y_dev, np.int8), y_test=np.asarray(y_te, np.int8))
    res = dict(unit=uid, dataset=ds, ir_target=ir, scheme=scheme_name, run=run, seed=seed,
               n_dev=len(y_dev), n_test=len(y_te), dev_pos=int(y_dev.sum()), test_pos=int(y_te.sum()),
               ir_actual=float((len(y_dev) - y_dev.sum()) / y_dev.sum()))
    # RQ1
    TIM = {}
    _t = time.time()
    res['noise'], raw_n = noise_analysis(X_dev, y_dev, scheme, S['noise_configs'], S['noise_repeats'], seed)
    RAW.update(raw_n); TIM['noise_sec'] = time.time() - _t
    # evaluator 'yang dilihat optimizer': satu partisi tetap per run
    search_ev = CVEvaluator(X_dev, y_dev, *scheme, partition_seed=seed, model_seed=seed)
    _t = time.time()
    # RQ2 — landscape
    res['fla'], raw_f = fla_analysis(search_ev, S['fla_samples'], S['walk_length'], seed,
                                     res['noise']['sigma_noise'])
    RAW.update(raw_f); TIM['fla_sec'] = time.time() - _t
    _t = time.time()
    # baseline default XGBoost (tanpa tuning): CV pencarian + evaluasi ulang + holdout
    res['default'] = corner_eval(None, search_ev, X_dev, y_dev, X_te, y_te, scheme, seed, RAW, 'default')
    # baseline 'pojok bawah' (semua hyperparameter di batas bawah): mendeteksi apakah keunggulan
    # optimizer yang kolaps ke batas (mis. SMA) hanya berasal dari bias ruang, bukan pencarian
    res['corner'] = corner_eval(np.zeros(D), search_ev, X_dev, y_dev, X_te, y_te, scheme, seed, RAW, 'corner')
    res['corner_upper'] = corner_eval(np.ones(D), search_ev, X_dev, y_dev, X_te, y_te, scheme, seed,
                                      RAW, 'corner_upper')
    TIM['baselines_sec'] = time.time() - _t
    # RQ2–3 — perilaku & reliabilitas
    res['optimizers'] = {}
    jobs = [(n, False) for n in S['optimizers']]
    if scheme_name in S.get('flip_schemes', []):
        jobs += [(n, True) for n in S.get('flip_optimizers', [])]
    for base, flipped in jobs:
        name = base + ('_flip' if flipped else '')
        obj = Objective(FlippedEvaluator(search_ev) if flipped else search_ev, S['opt_budget'])
        tr = Tracker(obj)
        _t = time.time()
        # seed SAMA dengan versi asli → satu-satunya perbedaan adalah arah encoding
        rng = np.random.default_rng([seed, list(OPTIMIZERS).index(base)])
        try:
            OPTIMIZERS[base](obj, rng, S['pop_size'], tr)
        except BudgetExhausted:
            pass
        TIM[f'{name}_search_sec'] = time.time() - _t
        beh = behavior_metrics(tr, obj, flipped=flipped)
        _t = time.time()
        best_x = 1 - obj.best_x if flipped else obj.best_x          # selalu dalam ruang asli
        best_cv = -obj.best_f
        reev = reevaluate(best_x, X_dev, y_dev, scheme, S['reeval_repeats'], seed)
        ho = holdout_eval(best_x, X_dev, y_dev, X_te, y_te, seed)
        TIM[f'{name}_reeval_holdout_sec'] = time.time() - _t
        # --- data mentah: SEMUA evaluasi optimizer (posisi dalam ruang ASLI) ---
        Xl = np.array(obj.X_log)
        Xl_real = 1 - Xl if flipped else Xl
        RAW.update({f'opt_{name}_{k}': v for k, v in pack_evals(obj.r_log, Xl_real).items()})
        RAW[f'opt_{name}_div'] = np.asarray(tr.div, np.float32)
        RAW[f'opt_{name}_div_evals'] = np.asarray(tr.evals, np.int32)
        if getattr(tr, 'pops', None):
            _P = np.stack(tr.pops)                                   # (iterasi, populasi, D)
            RAW[f'opt_{name}_pops'] = (1 - _P) if flipped else _P   # selalu ruang asli
        RAW.update({f'opt_{name}_reeval_{k}': v for k, v in reevaluate.last_raw.items()})
        RAW[f'opt_{name}_holdout_proba'] = holdout_eval.last_proba
        res['optimizers'][name] = dict(
            base_optimizer=base, flipped=flipped,
            behavior=beh, best_x=best_x.tolist(), best_params=decode(best_x),
            best_cv_wf1=best_cv, best_cv_mf1=obj.best_info['mf1'], best_cv_mcc=obj.best_info['mcc'],
            reeval_cv_wf1=reev['wf1'], reeval_cv_mf1=reev['mf1'], reeval_cv_mcc=reev['mcc'],
            winners_curse=best_cv - reev['wf1'],
            gap_cv_holdout=best_cv - ho['ho_wf1'], **ho,
            trace_best=[-v for v in obj.trace_best],
            div_evals=tr.evals, div=tr.div)
        if verbose:
            print(f"   {name:<9} bestCV={best_cv:.4f} reeval={reev['wf1']:.4f} (mF1={reev['mf1']:.3f}) "
                  f"holdout={ho['ho_wf1']:.4f} XPL̄={beh.get('xpl_mean', float('nan')):.1f}%")
    res['runtime_sec'] = time.time() - t0
    res['timing'] = TIM
    res['n_model_fits_search_ev'] = search_ev.n_fits
    res['meta'] = dict(code_version=CODE_VERSION, variant=VARIANT_TAG, host=socket.gethostname(),
                       platform=platform.platform(), on_kaggle=ON_KAGGLE, n_workers=N_WORKERS,
                       finished_at=datetime.datetime.now().isoformat(timespec='seconds'),
                       libs=RUN_META['libs'])
    # .npz ditulis DULU, lalu .json: .json berfungsi sebagai tanda 'unit selesai' untuk checkpoint
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'units', uid + '.npz'), **RAW)
    with open(path + '.tmp', 'w') as f:
        json.dump(res, f, default=float)
    os.replace(path + '.tmp', path)
    if verbose:
        n = res['noise']
        print(f"✅ {uid}  {res['runtime_sec']/60:.1f} menit | σ_noise={n['sigma_noise']:.4f} SNR={n['snr']:.2f} "
              f"ICC={n['icc']:.2f} | FDC={res['fla']['fdc']:.2f} ρ1={res['fla']['rho1']:.2f} "
              f"R²oob={res['fla']['rf_oob_r2']:.2f}")
    return res

# %%
# ============================================================
# 10. JALANKAN EKSPERIMEN
# ============================================================
def _ir_label(ir):
    return 'asli' if ir is None else str(ir)


schedule = [(ds, ir, sc, r) for ds in S['datasets'] for ir in S['ir_targets']
            for sc in S['cv_schemes'] for r in range(S['n_runs'])
            if level_allowed(ds, ir, DATA[ds][2]['n_pos'], DATA[ds][2]['n_neg'])]
n_all = len(schedule)
schedule = [t for t in schedule
            if (CHUNK['datasets'] is None or t[0] in CHUNK['datasets'])
            and (CHUNK['irs'] is None or _ir_label(t[1]) in [str(v) for v in CHUNK['irs']])
            and (CHUNK['schemes'] is None or t[2] in CHUNK['schemes'])
            and (CHUNK['runs'] is None or t[3] in CHUNK['runs'])]
print(f'Total unit dalam desain: {n_all} | dalam potongan ini (CHUNK): {len(schedule)}')
T_START = time.time()
if N_WORKERS <= 1:
    for i, (ds, ir, sc, r) in enumerate(schedule, 1):
        print(f'\n[{i}/{len(schedule)}] {unit_id(ds, ir, sc, r)}')
        run_unit(ds, ir, sc, r)
else:
    from joblib import Parallel, delayed
    todo = [t for t in schedule
            if not os.path.exists(os.path.join(OUTPUT_DIR, 'units', unit_id(*t) + '.json'))]
    print(f'Paralel: {N_WORKERS} worker, {len(todo)} unit belum selesai '
          f'({len(schedule) - len(todo)} sudah ada checkpoint)')

    def _job(t):
        r = run_unit(*t, verbose=False)
        return unit_id(*t), (r or {}).get('runtime_sec', 0)

    for uid, sec in Parallel(n_jobs=N_WORKERS, backend='loky', return_as='generator_unordered')(
            delayed(_job)(t) for t in todo):
        print(f'✅ {uid}  {sec/60:.1f} menit  | total berjalan {(time.time() - T_START)/60:.1f} menit', flush=True)
print(f'\nSelesai dalam {(time.time() - T_START)/60:.1f} menit')

# %%
# ============================================================
# 11. RINGKASAN & ESTIMASI RUNTIME UNTUK MODE FULL
# ============================================================
def load_units():
    rows = []
    for f in sorted(os.listdir(os.path.join(OUTPUT_DIR, 'units'))):
        if f.endswith('.json'):
            rows.append(json.load(open(os.path.join(OUTPUT_DIR, 'units', f))))
    return rows


UNITS = load_units()


def load_raw(uid):
    """Muat data mentah satu unit (.npz) untuk analisis lanjutan tanpa running ulang.
    Kunci penting: noise_*, fla_sample_*, fla_walk_*, opt_<NAMA>_X / _wf1 / _mf1 / _mcc / _fold_*,
    opt_<NAMA>_reeval_*, opt_<NAMA>_holdout_proba, default_*/corner_*/corner_upper_*, idx_*, y_*."""
    with np.load(os.path.join(OUTPUT_DIR, 'units', uid + '.npz')) as z:
        return {k: z[k] for k in z.files}
for u in UNITS:   # label IR sebagai teks agar 'asli' dan angka bisa dikelompokkan bersama
    u['ir_target'] = 'asli' if u['ir_target'] is None else f"{u['ir_target']}:1"
base_cols = ['dataset', 'ir_target', 'ir_actual', 'scheme', 'run', 'n_dev', 'dev_pos', 'test_pos']
noise_df = pd.DataFrame([{**{k: u[k] for k in base_cols}, **u['noise']} for u in UNITS])
fla_df = pd.DataFrame([{**{k: u[k] for k in base_cols}, **u['fla']} for u in UNITS])
corner_df = pd.DataFrame([{**{k: u[k] for k in base_cols}, 'corner': lab, **u[key]}
                          for u in UNITS for key, lab in (('default', 'default'), ('corner', 'bawah'),
                                                          ('corner_upper', 'atas'))
                          if key in u])
opt_df = pd.DataFrame([{**{k: u[k] for k in base_cols}, 'optimizer': n,
                        **{k: v for k, v in o.items() if k not in ('trace_best', 'div', 'div_evals', 'best_x', 'best_params', 'behavior')},
                        **{k: v for k, v in o['behavior'].items() if k != 'final_centroid'}}
                       for u in UNITS for n, o in u['optimizers'].items()])
for name, df in [('noise', noise_df), ('fla', fla_df), ('optimizers', opt_df), ('corner', corner_df)]:
    df.to_csv(os.path.join(OUTPUT_DIR, f'summary_{name}.csv'), index=False)

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 50)
print('=== RQ1: Noise ===')
print(noise_df.groupby(['dataset', 'ir_target', 'scheme'], dropna=False)[
    ['ir_actual', 'sigma_noise', 'sigma_signal', 'snr', 'icc', 'rank_stability_tau']].mean().round(4))
print('\n=== RQ2: Landscape ===')
print(fla_df.groupby(['dataset', 'ir_target', 'scheme'], dropna=False)[
    ['fdc', 'dispersion', 'top_clusters', 'rf_oob_r2', 'rho1', 'neutral_ratio', 'f_range']].mean().round(3))
print('\n=== RQ2–3: Perilaku & reliabilitas per optimizer ===')
print(opt_df.groupby(['ir_target', 'scheme', 'optimizer'], dropna=False)[
    ['best_cv_wf1', 'reeval_cv_wf1', 'reeval_cv_mf1', 'reeval_cv_mcc', 'winners_curse', 'ho_wf1', 'xpl_mean',
     'final_real_frac_at_lower', 'final_real_frac_at_upper']].mean().round(4))
if len(corner_df):
    print('\n=== Baseline pojok (semua hyperparameter di batas bawah / atas) ===')
    print(corner_df.groupby(['ir_target', 'scheme', 'corner'])[
        ['cv_wf1', 'reeval_cv_wf1', 'reeval_cv_mf1', 'reeval_cv_mcc', 'ho_wf1']].mean().round(4))

# ---- Flip test: asli vs dicerminkan (berpasangan per unit, seed sama) ----
if 'flipped' in opt_df and opt_df['flipped'].any():
    fl = opt_df[opt_df['flipped'] == True]
    keys = ['dataset', 'ir_target', 'scheme', 'run', 'base_optimizer']
    pair = fl.merge(opt_df[opt_df['flipped'] == False], on=keys, suffixes=('_flip', '_asli'))
    pair['delta_reeval_wf1'] = pair['reeval_cv_wf1_asli'] - pair['reeval_cv_wf1_flip']
    pair['delta_reeval_mf1'] = pair['reeval_cv_mf1_asli'] - pair['reeval_cv_mf1_flip']
    pair.to_csv(os.path.join(OUTPUT_DIR, 'summary_flip_pairs.csv'), index=False)
    print('\n=== Flip test (Δ = asli − dicerminkan; Δ > 0 → optimizer diuntungkan oleh bias batas bawah) ===')
    agg = pair.groupby(['ir_target', 'base_optimizer']).agg(
        reeval_asli=('reeval_cv_wf1_asli', 'mean'), reeval_flip=('reeval_cv_wf1_flip', 'mean'),
        delta_wf1=('delta_reeval_wf1', 'mean'), delta_mf1=('delta_reeval_mf1', 'mean'),
        bawah_asli=('final_real_frac_at_lower_asli', 'mean'), atas_flip=('final_real_frac_at_upper_flip', 'mean'),
        n=('run', 'count'))
    print(agg.round(4))
    for b, g in pair.groupby('base_optimizer'):
        if len(g) >= 6 and g['delta_reeval_wf1'].abs().sum() > 0:
            p = stats.wilcoxon(g['delta_reeval_wf1']).pvalue
            print(f'   Wilcoxon {b}: median Δ={g.delta_reeval_wf1.median():.4f}, p={p:.4f} (n={len(g)})')

# Estimasi runtime full berdasarkan waktu per evaluasi pilot
if UNITS:
    rt = np.mean([u['runtime_sec'] for u in UNITS])
    P, F = SETTINGS['pilot'], SETTINGS['full']
    def evals_per_unit(s):
        return (s['noise_configs'] * s['noise_repeats'] + s['fla_samples'] + s['walk_length']
                + len(s['optimizers']) * (s['opt_budget'] + s['reeval_repeats'])
                + len(s.get('flip_optimizers', [])) * (s['opt_budget'] + s['reeval_repeats'])
                  * len(s.get('flip_schemes', [])) / len(s['cv_schemes'])
                + 2 * (1 + s['reeval_repeats']))
    sec_per_eval = rt / evals_per_unit(P)
    n_full = 0
    for ds in F['datasets']:
        X_, y_, inf_ = load_dataset(ds)
        n_full += sum(level_allowed(ds, t, inf_['n_pos'], inf_['n_neg'], F) for t in F['ir_targets'])
    n_full *= len(F['cv_schemes']) * F['n_runs']
    est_h = n_full * evals_per_unit(F) * sec_per_eval / 3600
    est_h_par = est_h / max(N_WORKERS, 1) * 1.3   # paralel: efisiensi ~75% (perkiraan kasar)
    print(f'\n⏱  Rata-rata {rt/60:.1f} menit/unit pilot ≈ {sec_per_eval:.2f} detik/evaluasi (rata-rata cv5x1 & cv5x3)')
    print(f'   Estimasi FULL: {n_full} unit × {evals_per_unit(F)} evaluasi ≈ {est_h:.0f} jam '
          f'(kasar; dataset lain lebih kecil dari Cervical)')
    for w in (4, 6, 8):
        print(f'   Dengan {w} worker paralel di PC/laptop: ≈ {est_h / w * 1.3:.0f} jam '
              f'(kecepatan per core PC bisa berbeda dari Colab)')

# %%
# ============================================================
# 12. VISUALISASI SANITY-CHECK PILOT
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
# (a) noise vs signal
ax = axes[0, 0]
g = noise_df.groupby(['ir_target', 'scheme'], dropna=False)[['sigma_noise', 'sigma_signal']].mean()
g.plot.bar(ax=ax); ax.set_title('RQ1: σ noise vs σ sinyal (wF1)'); ax.set_xlabel('(IR target, skema CV)')
# (b) konvergensi
ax = axes[0, 1]
for u in UNITS[:1]:
    for n, o in u['optimizers'].items():
        ax.plot(o['trace_best'], label=n)
    ax.set_title(f"Best-so-far wF1 — {u['unit']}"); ax.set_xlabel('evaluasi'); ax.legend()
# (c) diversity (XPL%)
ax = axes[1, 0]
for u in UNITS[:1]:
    for n, o in u['optimizers'].items():
        if o['div']:
            d = np.array(o['div']); ax.plot(o['div_evals'], 100 * d / d.max(), label=n)
    ax.set_title('Eksplorasi % (diversity / max)'); ax.set_xlabel('evaluasi'); ax.legend()
# (d) winner's curse
ax = axes[1, 1]
opt_df.boxplot(column='winners_curse', by='optimizer', ax=ax)
ax.axhline(0, color='k', lw=0.8); ax.set_title("RQ3: winner's curse (bestCV − re-eval CV)"); plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pilot_sanity.png'), dpi=150)
plt.show()
