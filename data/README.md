# Data

The experiment expects four CSV files in this folder (file names are fixed in `DATASET_REGISTRY`):

| File | Dataset | Source | Included |
|---|---|---|---|
| `Breast_Cancer_Winconsin.csv` | Breast Cancer Wisconsin (Diagnostic), 569 rows | W. Wolberg, O. Mangasarian, N. Street, W. Street, UCI Machine Learning Repository, 1993, doi:10.24432/C5DW2B (CC BY 4.0) | yes |
| `Thyroid_Diff.csv` | Differentiated Thyroid Cancer Recurrence, 383 rows | S. Borzooei, A. Tarokhian, UCI Machine Learning Repository, 2023, doi:10.24432/C5632J (CC BY 4.0) | yes |
| `risk_factors_cervical_cancer.csv` | Cervical Cancer (Risk Factors), 858 rows | K. Fernandes, J. Cardoso, J. Fernandes, UCI Machine Learning Repository, 2017, doi:10.24432/C5Z310 (CC BY 4.0) | yes |
| `survey_lung_cancer.csv` | Lung Cancer (online survey), 309 rows | M. A. Bhat, Kaggle, 2021, https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer | **no** – download from Kaggle and place here |

The Lung file is not redistributed because its licence on Kaggle should be checked by the user.
Targets: `diagnosis` (Breast), `Recurred` (Thyroid), `Biopsy` (Cervical), `LUNG_CANCER` (Lung).
Columns dropped to prevent label leakage: `Response` (Thyroid); `Hinselmann`, `Schiller`, `Citology` (Cervical).
Duplicate rows are removed before splitting (Lung 309 → 276, Thyroid 383 → 364, Cervical 858 → 835).
