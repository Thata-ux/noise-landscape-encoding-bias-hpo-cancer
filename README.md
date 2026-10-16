# Evaluation Noise, Fitness Landscape, and Encoding Bias in Metaheuristic Hyperparameter Optimization of XGBoost on Small Imbalanced Cancer Data

Code, raw results, and analysis for the article of the same title by
**Tahta Herdian Andika**, **Widowati**, and **Catur Edi Widodo** (Universitas Diponegoro, Indonesia).
The manuscript is written in Indonesian; this repository is documented in English.

The study tests one chain on XGBoost hyperparameter optimization (HPO) for small, imbalanced cancer data:

```
class composition + CV scheme  →  evaluation noise & SNR  →  measured fitness landscape
                               →  search behaviour (RS, BFO, HHO, SMA)  →  reliability of the optimum
```

BFO, HHO, and SMA are used as **behavioural probes**, not as candidates for "the best optimizer";
Random Search (RS) is the reference without directional bias.

## Main findings

1. **Evaluation noise depends on per-class sample counts, not on the imbalance ratio.** Noise falls with the
   number of majority samples, signal grows with the number of minority samples; adding majority samples can
   *raise* the SNR while the imbalance ratio gets worse. Repeating CV (5×3) lowers noise by 0.565 (theory 1/√3 = 0.577).
2. **Noise changes almost every landscape metric.** Lowering noise on the same data raises ρ₁ (random walk) and the
   learnability (surrogate R²) and lowers neutrality, so landscape metrics on small data must be read together with SNR.
3. **SMA's advantage over RS is an encoding-induced performance advantage.** SMA beats RS by +0.0068 weighted F1 in the
   original encoding but loses by −0.0035 when the search space is mirrored (x → 1 − x). The cause is a
   coordinate-dependent structural bias: its population is pulled to the coordinate origin, which in a [0,1] encoding is
   the lower bound of every hyperparameter. The same bias appears with the third-party **mealpy** implementation, on
   random/constant fitness, and moves to the centre when the domain is [−1, 1]. A Bayesian optimization baseline (**TPE**,
   Optuna) is practically invariant to mirroring (median Δ 0.0007) and still beats RS (+0.0043).
4. **Winner's curse is as large as the differences between optimizers** and halves with repeated CV.

## Repository layout

```
data/                      Input datasets (3 of 4 included; see data/README.md for the Lung file)
src/                       Plain Python versions of the notebooks (# %% cells)
  01_experiment_v5_2.py      Main experiment: 280 units (noise, FLA, optimizers, flip test, re-evaluation)
  02_analysis.py             All tables and figures of the article from the raw units
  03_robustness_mealpy.py    Robustness check with mealpy 3.0.1 (12 units)
  05_baseline_tpe.py         Bayesian optimization baseline: TPE (Optuna 4.1), original + mirrored space (140 units)
  04a_sensitivity_dependency.py     Hierarchical bootstrap, condition-level tests, leave-one-dataset-out (Table S1)
  04b_sensitivity_fitness_metrics.py Noise and landscape recomputed with minority F1 and MCC (Tables S2a–c)
  04c_structural_bias_no_fitness.py  Optimizers on random / constant fitness (Table S3)
  fig1_framework.py          Conceptual framework figure (Figure 1)
notebooks/                 The same three scripts as Jupyter/Colab notebooks
results/
  units/                     Raw results of all 280 units: one .json (summary) + one .npz (all raw evaluations) per unit
  tidy/                      Tidy tables extracted from the units (noise, fla, optimizers, baselines, timing)
  tables/                    Article tables (CSV), file numbers = article table numbers
  figures/                   Article figures (PNG 300 dpi + PDF), file numbers = article figure numbers
  robustness_mealpy/         Robustness-check units, Table 8, and supplementary Figure S1
  sensitivity/               Sensitivity analyses (supplementary Tables S1–S3 of the manuscript)
  baseline_tpe/              TPE units (140 × original + mirrored), paired tests, summary, and figure
  Paper2_all_tables.xlsx     All analysis tables in one workbook
  key_numbers.json           Key numbers quoted in the text
manuscript/                Figures and tables in English (Word) and the reference list (RIS, BibTeX)
```

## Experimental design

| Factor | Levels |
|---|---|
| Datasets | Lung Cancer (survey), Differentiated Thyroid Cancer Recurrence, Breast Cancer Wisconsin (Diagnostic), Cervical Cancer (Risk Factors) |
| Class composition | 14 conditions from subsampling to IR 3:1, 6:1, 12:1, 20:1 and original (Series A: minority fixed; Series B: majority fixed) |
| CV scheme (noise manipulation) | `cv5x1` (stratified 5-fold) and `cv5x3` (5-fold repeated 3×) |
| Runs | 10 per condition and scheme → **280 units** |
| Fitness | mean weighted F1 over CV folds (minority F1 and MCC recorded) |
| Search space | 9 XGBoost hyperparameters, normalised to [0,1]⁹ |
| Optimizers | RS, BFO, HHO, SMA, SMAc (centred SMA), 300 evaluations, population 20; flip test for BFO/HHO/SMA on `cv5x1`; TPE baseline (original + mirrored) on `cv5x1` |
| Reliability | best configuration re-evaluated on 5 fresh CV partitions; winner's curse = best CV − re-evaluated score |

Data decisions (to avoid label leakage): Thyroid `Response` and Cervical `Hinselmann`, `Schiller`, `Citology` are dropped;
duplicates removed; the minority class is always coded 1; preprocessing is fitted inside each training fold only.

## Mapping between the article and the files

File names follow the article numbering. All tables and figures are also collected, with captions, in
`manuscript/Paper2_Figures_Tables_EN.docx`.

| Article | File(s) |
|---|---|
| Table 1 (positioning vs. literature) | `manuscript/Paper2_Figures_Tables_EN.docx` |
| Table 2 (design of 14 conditions) | `results/tables/Table2_condition_design.csv` |
| Table 3 (search space) | `PARAM_SPACE` in `src/01_experiment_v5_2.py` |
| Table 4 (noise, signal, SNR per condition) | `results/tables/Table4_noise_per_condition*.csv` |
| Table 5 (log–log models) | `results/tables/Table5_noise_signal_snr_models.csv` |
| Table 6 (noise effect on landscape) | `results/tables/Table6a_*.csv`, `Table6b_*.csv` |
| Table 7 (flip test) | `results/tables/Table7_flip_test.csv`; bound fractions from `results/tidy/optimizers.csv` |
| Table 8 (mealpy robustness check) | `results/robustness_mealpy/Table8_robustness_mealpy.csv` |
| Table 9 (winner's curse) | `results/tables/Table9a_winners_curse.csv`, `Table9b_*.csv` |
| Table 10 (vs. RS and XGBoost default) | `results/tables/Table10_vs_random_search_and_default.csv` |
| Figures 1–9 | `results/figures/Fig1_*` … `Fig9_*` |
| Additional tables (repository only) | `results/tables/TableS4_*` … `TableS8_*` (landscape per condition, hyperparameter importance, winner's-curse model, compute cost, search time) |
| Supplementary Tables S1, S2a–b, S3 (sensitivity) | `results/sensitivity/TableS1_dependency_aware.csv`, `TableS2a/b/c_*.csv`, `TableS3_structural_bias_no_fitness.csv` |
| Supplementary Figure S1 | `results/robustness_mealpy/FigS1_robustness_mealpy.png` |

Tables and figures are produced in English by default. Set the environment variable `P2_LANG=id` to produce the
Indonesian versions used in the Indonesian manuscript.

## Sensitivity analyses

* **Dependence between units.** Runs within a condition share a data source, so the main paired results are re-tested with a
  hierarchical bootstrap (dataset → condition → run), Wilcoxon tests on condition means (14 conditions), and
  leave-one-dataset-out. The SMA flip-test effect is positive in 14/14 conditions.
* **Fitness metric.** Noise, SNR, and landscape metrics are recomputed with minority F1 and MCC from the stored per-evaluation
  scores. The effect of noise on the landscape holds for all three metrics; the class-count mechanism is metric-specific
  (with minority F1/MCC, the minority count dominates).
* **Structural bias without fitness information.** On random or constant fitness, SMA (own code and mealpy) collapses towards the
  coordinate origin (lower bound in [0,1], centre in [−1,1]); BFO and centred SMA do not. HHO's bias depends on whether greedy
  selection is used (absent in the original equations, added in mealpy).

## Reproducing the results

**Only the tables and figures (minutes, no re-running):**

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute 02_analysis.ipynb   # reads ../results/units
```

Outputs go to `results/analysis_final_en/` (identical to `results/tables` and `results/figures`).

**The full experiment (≈ 190 hours of summed unit run time):** put the four CSV files in `data/`, then run
`notebooks/01_experiment_v5_2.ipynb` (Colab or local). Each unit is check-pointed, so the run can be split over
sessions or machines with `CHUNK` (or the environment variables `P2_DATASETS`, `P2_SCHEMES`, `P2_RUNS`, `P2_IRS`), e.g.

```bash
P2_RUN_MODE=full P2_WORKERS=4 P2_DATA_DIR=./data P2_OUTPUT_DIR=./outputs python src/01_experiment_v5_2.py
```

**The TPE baseline (≈ 3 h for 140 units with 2 workers):** run `notebooks/05_baseline_tpe.ipynb`; `02_analysis` picks up
`results/baseline_tpe/units` automatically and adds TPE to Tables 7, 9, 10 and Figures 7, 9 as a separate test family.

**The mealpy robustness check (≈ 30 min with 2 workers):** run `notebooks/03_robustness_mealpy.ipynb`; it re-uses the
same units (seeds, subsampling, CV partitions) and pairs its results with `results/units`.

All units in `results/units` were produced with Python 3.13, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1,
SciPy 1.16.3, and XGBoost 3.4.1 (recorded in each unit's `meta` field); `requirements.txt` pins these versions.

### Raw unit files

Each `results/units/<Dataset>_IR-<level>_<scheme>_run<NN>.json` holds the summary of one unit (noise, landscape,
baselines, and per-optimizer behaviour and reliability). The matching `.npz` holds every raw evaluation:
noise matrix (`noise_*`), landscape samples and random walk (`fla_sample_*`, `fla_walk_*`), every optimizer
evaluation with its position and per-fold scores (`opt_<NAME>_X`, `opt_<NAME>_wf1`, `opt_<NAME>_fold_*`),
population snapshots (`opt_<NAME>_pops`), re-evaluation scores, hold-out probabilities, and the row indices of the
development and test sets (`idx_dev`, `idx_test`). New analyses can be computed from these files without re-running.

## Citation

If you use this code or data, please cite the article (see `CITATION.cff`; details will be updated on publication).

## License

Code: MIT (see `LICENSE`). Datasets keep the licences of their original sources (see `data/README.md`).

## Acknowledgement

The authors gratefully acknowledge the research funding provided by the Foundation of Universitas Aisyah Pringsewu
(Yayasan Aisyah Lampung).
