# Evaluation Noise, Fitness Landscape, and Encoding-Induced Performance Advantage in Metaheuristic Hyperparameter Optimization of XGBoost on Small Imbalanced Cancer Data

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
   Because the noise design is crossed (the same 10 partitions for all 30 configurations), about 40% of the reported
   noise variance is a uniform partition shift that does not change rankings; the reported SNR is therefore conservative,
   but the ranking of units by SNR is unchanged (Spearman ρ = 0.98).
2. **Noise changes the measured landscape.** Lowering noise on the same data raises ρ₁ (random walk), the
   learnability (surrogate R²), and the fitness–distance correlation. The neutrality ratio depends on its threshold:
   with a fixed threshold it *rises* when noise falls (noise hides real plateaus), so it is not used as evidence.
   Landscape metrics on small data must be read together with SNR.
3. **SMA's advantage over RS is an encoding-induced performance advantage.** SMA beats RS by +0.0068 weighted F1 in the
   original encoding but loses by −0.0035 when the search space is mirrored (x → 1 − x). The cause is a
   coordinate-dependent structural bias: its population is pulled to the coordinate origin, which in a [0,1] encoding is
   the lower bound of every hyperparameter. The bias appears with the third-party **mealpy** implementation and on
   random/constant fitness (all 9 dimensions reject uniformity). On 12 units it persists with a **log encoding**
   (SMA − RS shrinks from +0.0132 to +0.0084; mirroring effect +0.0161), follows a dose–response when only the three
   influential hyperparameters are mirrored, appears with **LightGBM**, and affects **MCC and minority F1** when MCC is the
   fitness. Making the SMA contraction relative to the best solution (**SMAb**) removes most of the mirroring effect
   (0.0239 → 0.0049) and, with it, SMA's advantage over RS. DE and CMA-ES show no mirroring effect, and for a Bayesian
   optimization baseline (**TPE**, Optuna) the effect is small and inconclusive.
4. **HHO's weaker bias comes from two operators.** The second exploration branch and the soft-besiege equation, which
   accept new positions without selection, pull towards the origin; hard besiege and the greedy rapid dives follow the
   best solution, so on informative fitness HHO follows the optimum.
5. **Winner's curse is as large as the differences between optimizers** and halves with repeated CV.

## Repository layout

```
data/                      Input datasets (3 of 4 included; see data/README.md for the Lung file)
src/                       Plain Python versions of the notebooks (# %% cells)
  01_experiment_v5_2.py      Main experiment: 280 units (noise, FLA, optimizers, flip test, re-evaluation)
  02_analysis.py             Tables and figures of the main experiment from the raw units
  03_robustness_mealpy.py    Robustness check with mealpy 3.0.1 (12 units)
  04a_sensitivity_dependency.py      Hierarchical bootstrap, condition-level tests, leave-one-dataset-out
  04b_sensitivity_fitness_metrics.py Noise and landscape recomputed with minority F1 and MCC
  04c_structural_bias_no_fitness.py  Optimizers on random / constant fitness (centroid summary)
  05_baseline_tpe.py         Bayesian optimization baseline: TPE (Optuna 4.1), original + mirrored space (140 units)
  core.py                    Shared core for 06–08; executes the data, evaluator, and optimizer code of 01 unchanged
  06_reanalysis_stored_units.py      Analyses of the stored units (no model training, ~1–5 min)
  07_subset_encoding_variants.py     12 units: log encoding, partial mirroring, invariant SMA variants, DE, CMA-ES
  08_subset_lightgbm_mcc.py          12 units: LightGBM and MCC as search fitness
  fig1_framework.py          Conceptual framework figure (Figure 1)
notebooks/                 The same scripts as Jupyter/Colab notebooks; 07 and 08 also as Kaggle notebooks (*_kaggle)
results/
  units/                     Raw results of all 280 units: one .json (summary) + one .npz (all raw evaluations) per unit
  tidy/                      Tidy tables extracted from the units (noise, fla, optimizers, baselines, timing)
  tables/, figures/          Tables (CSV) and figures (PNG 300 dpi + PDF) of the main experiment
  robustness_mealpy/         mealpy robustness units, table, and figure
  sensitivity/               Dependency-aware, fitness-metric, and structural-bias analyses
  baseline_tpe/              TPE units (140 × original + mirrored), paired tests, summary, and figure
  reanalysis/                Output of 06 (SNR CIs and two-way decomposition, neutrality thresholds, plateau indicators,
                             Figure 8 with TPE, uniformity tests of structural bias, HHO operator log)
  subset_encoding_variants/  Output of 07: units/ (48 unit × block files), long.csv, paired_tests.csv, positions_by_arm.csv
  subset_lightgbm_mcc/       Output of 08: units/ (24 files), long.csv, paired_tests.csv, LightGBM hyperparameter correlations
  Paper2_all_tables.xlsx     Main-experiment analysis tables in one workbook
  key_numbers.json           Key numbers of the main experiment quoted in the text
manuscript/                Reference list in citation order (RIS, BibTeX)
```

## Experimental design

| Factor | Levels |
|---|---|
| Datasets | Lung Cancer (survey), Differentiated Thyroid Cancer Recurrence, Breast Cancer Wisconsin (Diagnostic), Cervical Cancer (Risk Factors) |
| Class composition | 14 conditions from subsampling to IR 3:1, 6:1, 12:1, 20:1 and original (Series A: minority fixed; Series B: majority fixed) |
| CV scheme (noise manipulation) | `cv5x1` (stratified 5-fold) and `cv5x3` (5-fold repeated 3×) |
| Runs | 10 per condition and scheme → **280 units** |
| Fitness | mean weighted F1 over CV folds (minority F1 and MCC recorded) |
| Search space | 9 XGBoost hyperparameters, normalised to [0,1]⁹ with linear decoding |
| Optimizers | RS, BFO, HHO, SMA, SMAc (centred SMA), 300 evaluations, population 20; flip test for BFO/HHO/SMA on `cv5x1`; TPE baseline (original + mirrored) on `cv5x1` |
| Reliability | best configuration re-evaluated on 5 fresh CV partitions; winner's curse = best CV − re-evaluated score |
| Subset experiments (07, 08) | 12 `cv5x1` units (Cervical original, Cervical 6:1, Breast original; runs 0–3), identical data, partitions, seeds, budget, and re-evaluation, paired with the main units |

Data decisions (to avoid label leakage): Thyroid `Response` and Cervical `Hinselmann`, `Schiller`, `Citology` are dropped;
duplicates removed; the minority class is always coded 1; preprocessing is fitted inside each training fold only.

## Mapping between the article and the files

Some file names were fixed before the final numbering, so a file's number can differ from the article number; use this table.

**Main text**

| Article | Content | File(s) |
|---|---|---|
| Table 1 | Positioning vs. literature | article text |
| Table 2 | Design of the 14 class-composition conditions | `results/tables/Table2_condition_design.csv` |
| Table 3 | Search space (9 XGBoost hyperparameters) | `PARAM_SPACE` in `src/01_experiment_v5_2.py` |
| Table 4 | Noise, signal, SNR, ICC, Kendall τ per condition | `results/tables/Table4_noise_per_condition*.csv` |
| Table 5 | Log–log models of noise, signal, and SNR | `results/tables/Table5_noise_signal_snr_models.csv` |
| Table 6 | Effect of noise on landscape metrics | `results/tables/Table6a_*.csv`, `Table6b_*.csv` |
| Table 7 | Flip test (incl. TPE) | `results/tables/Table7_flip_test.csv`; bound fractions from `results/tidy/optimizers.csv` |
| Table 8 | Implementation robustness (mealpy 3.0.1, 12 units) | `results/robustness_mealpy/Table8_robustness_mealpy.csv` |
| Table 9 | Winner's curse (incl. TPE) | `results/tables/Table9a_winners_curse.csv`, `Table9b_*.csv` |
| Table 10 | Re-evaluated wF1 vs. RS and default XGBoost (incl. TPE) | `results/tables/Table10_vs_random_search_and_default.csv` |
| Table 11 | Subset experiments (encoding, partial mirroring, SMA variants, DE/CMA-ES, LightGBM, MCC) | `results/subset_encoding_variants/paired_tests.csv`, `results/subset_lightgbm_mcc/paired_tests.csv` |
| Figure 1 | Conceptual framework | `results/figures/Fig1_framework.*` |
| Figure 2 | Noise/signal mechanism | `results/figures/Fig2_noise_signal_mechanism.*` |
| Figure 3 | Validation of the noise manipulation | `results/figures/Fig3_noise_manipulation.*` |
| Figure 4 | Landscape vs. noise | `results/figures/Fig4_landscape_vs_noise.*` |
| Figure 5 | Hyperparameter importance | `results/figures/Fig5_hyperparameter_importance.*` |
| Figure 6 | Search behaviour | `results/figures/Fig6_search_behaviour.*` |
| Figure 7 | Flip test and boundary bias | `results/figures/Fig7_flip_test_boundary_bias.*` |
| Figure 8 | Winner's curse (incl. TPE) | `results/reanalysis/Fig8_winners_curse_withTPE.*` |
| Figure 9 | Forest plot vs. Random Search | `results/figures/Fig9_forest_vs_random_search.*` |

**Supplementary Material**

| Article | Content | File(s) |
|---|---|---|
| Table S1 | Dependency-aware analyses (hierarchical bootstrap, condition level, leave-one-dataset-out) | `results/sensitivity/TableS1_dependency_aware.csv` |
| Table S2a–b | Noise model and landscape effect for wF1, minority F1, and MCC | `results/sensitivity/TableS2a_*.csv`, `TableS2b_*.csv` (`TableS2c_*.csv`: SNR agreement between metrics) |
| Table S3 | Structural bias without fitness information (centroids) | `results/sensitivity/TableS3_structural_bias_no_fitness.csv` |
| Table S4 | Mixed model of the winner's curse | `results/tables/TableS6_winners_curse_model.csv` |
| Table S5 | Compute cost per unit | `results/tables/TableS7_compute_cost.csv` |
| Table S6 | SNR per condition with bootstrap CIs and two-way variance decomposition | `results/reanalysis/1b_snr_per_condition.csv` (per unit: `1a_snr_per_unit.csv`) |
| Table S7 | Uniformity tests of final positions on uninformative fitness (KS, Cramér–von Mises) | `results/reanalysis/5a_bias_tests.csv` (raw: `5_bias_raw.csv`) |
| Table S8 | Neutrality ratio under different thresholds | `results/reanalysis/2_neutrality_fixed_threshold.csv` |

**Repository only:** `results/tables/TableS4_*`, `TableS5_*`, `TableS8_*` (landscape per condition, hyperparameter
importance per unit, search time per optimizer); `results/robustness_mealpy/FigS1_*`;
`results/reanalysis/3a_*`, `3b_*` (SNR–learnability per dataset, plateau indicators), `6_*` (HHO operator log).

Tables and figures are produced in English by default. Set the environment variable `P2_LANG=id` to produce the
Indonesian versions used in the Indonesian manuscript.

## Additional analyses

* **Dependence between units.** Runs within a condition share a data source, so the main paired results are re-tested
  with a hierarchical bootstrap (dataset → condition → run), Wilcoxon tests on condition means (14 conditions), and
  leave-one-dataset-out. The SMA flip-test effect is positive in 14/14 conditions.
* **Fitness metric.** Noise, SNR, and landscape metrics are recomputed with minority F1 and MCC from the stored
  per-evaluation scores; `08_subset_lightgbm_mcc.py` additionally runs searches with MCC as fitness.
* **Structural bias without fitness information.** On random or constant fitness, SMA (own code and mealpy) collapses
  towards the coordinate origin (lower bound in [0,1], centre in [−1,1]); per-dimension KS and Cramér–von Mises tests
  reject uniformity for HHO, SMA, SMAc, and SMAm, but not for RS, BFO, SMAb, or CMA-ES.
* **Encoding, model, and optimizer variants (12 units).** `07` and `08` reuse exactly the same units as the mealpy check;
  every unit re-computes the default-XGBoost CV score and compares it with the main experiment
  (`repro_check_default_cv_wf1_diff`, 0 for all stored units).

## Reproducing the results

**Only the tables and figures (minutes, no re-running):**

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute 02_analysis.ipynb                 # main tables and figures
cd ../src && python 06_reanalysis_stored_units.py                           # results/reanalysis (~1–5 min)
python 07_subset_encoding_variants.py && python 08_subset_lightgbm_mcc.py   # re-analyse the stored subset units
```

`02_analysis` writes to `results/analysis_final_en/` (identical to `results/tables` and `results/figures`).
`07` and `08` skip every unit that already exists in `results/subset_*/units/` and only redo the paired analysis.

**The full experiment (≈ 190 hours of summed unit run time):** put the four CSV files in `data/`, then run
`notebooks/01_experiment_v5_2.ipynb` (Colab or local). Each unit is check-pointed, so the run can be split over
sessions or machines with `CHUNK` (or the environment variables `P2_DATASETS`, `P2_SCHEMES`, `P2_RUNS`, `P2_IRS`), e.g.

```bash
P2_RUN_MODE=full P2_WORKERS=4 P2_DATA_DIR=./data P2_OUTPUT_DIR=./outputs python src/01_experiment_v5_2.py
```

**The TPE baseline (≈ 3 h for 140 units with 2 workers):** run `notebooks/05_baseline_tpe.ipynb`; `02_analysis` picks up
`results/baseline_tpe/units` automatically and adds TPE as a separate test family.

**The mealpy robustness check (≈ 30 min with 2 workers):** run `notebooks/03_robustness_mealpy.ipynb`.

**The subset experiments from scratch:** delete `results/subset_*/units/` and run `07` (216 searches) and `08`
(156 searches). On Kaggle (4 CPU cores) each took about 2–3 hours with `notebooks/*_kaggle.ipynb`
(Save & Run All; resumable across sessions). Set `TEST_MODE = True` first for a 2–5 minute smoke test.

All units in `results/units` were produced with Python 3.13, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1,
SciPy 1.16.3, and XGBoost 3.4.1 (recorded in each unit's `meta` field); `requirements.txt` pins these versions.

### Raw unit files

Each `results/units/<Dataset>_IR-<level>_<scheme>_run<NN>.json` holds the summary of one unit (noise, landscape,
baselines, and per-optimizer behaviour and reliability). The matching `.npz` holds every raw evaluation:
noise matrix (`noise_*`), landscape samples and random walk (`fla_sample_*`, `fla_walk_*`), every optimizer
evaluation with its position and per-fold scores (`opt_<NAME>_X`, `opt_<NAME>_wf1`, `opt_<NAME>_fold_*`),
population snapshots (`opt_<NAME>_pops`), re-evaluation scores, hold-out probabilities, and the row indices of the
development and test sets (`idx_dev`, `idx_test`). New analyses can be computed from these files without re-running.

The subset units (`results/subset_*/units/<unit>__<block>.json|npz`) store, per arm, the method, coordinate transform,
best configuration (in the original linear coordinates), search and re-evaluation scores (wF1, minority F1, MCC),
hold-out scores, final-population positions, all evaluated positions (`opt_<ARM>_Xreal`), and population snapshots.

## Citation

If you use this code or data, please cite the article (see `CITATION.cff`; details will be updated on publication).

## License

Code: MIT (see `LICENSE`). Datasets keep the licences of their original sources (see `data/README.md`).

## Acknowledgement

The authors gratefully acknowledge the research funding provided by the Foundation of Universitas Aisyah Pringsewu
(Yayasan Aisyah Lampung).
