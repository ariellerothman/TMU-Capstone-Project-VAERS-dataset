# COVID-19 Vaccine Adverse Event Phenotyping and Severity Prediction

## Project Overview

This project applies natural language processing (NLP) and predictive analytics to COVID-19 vaccine-related reports from the Vaccine Adverse Event Reporting System (VAERS) spanning 2020–2025. By combining text mining, classification, clustering, and temporal analysis, we demonstrate how machine learning methods can enhance interpretation of large-scale vaccine safety surveillance data.

### Research Questions
1. Can symptom narratives improve prediction of severe adverse event outcomes beyond structured demographic and medical history variables?
2. Do unsupervised clustering methods identify stable adverse event phenotypes?
3. Do reporting volumes, severity rates, and narrative clusters shift over time?

---

## Dataset

**Source:** Adverse Drug Effects (ADE) Detection dataset compiled from VAERS  
**Download:** [Kaggle – VAERS ADE Detection](https://www.kaggle.com/datasets/khalilartan/adverse-drug-effects-detection) (search "VAERS ADE detection")  
**File needed:** `COMBINED_DATA_ALL_YEARS.csv`  
**File size:** > 1 GB — **not included in this repository**  
**Time Period:** December 12, 2020 – July 25, 2025  
**Original Records:** 2,132,431 total adverse event reports  
**COVID-19 Subset:** 986,096 unique case-level records (VAERS_ID)  

### Key Characteristics
- **19.85%** classified as serious outcomes (composite: death, life-threatening, hospitalization, ER visit, disability)
- **Structured variables:** Demographics (age, sex, state), vaccine info (manufacturer, type), dates, binary outcomes
- **Unstructured variables:** Symptom narratives, medical history, allergies, medications, lab data, prior vaccinations
- **Missingness:** Moderate (11–14%) in numeric fields; high (50–70%) in free-text clinical context

---

## Repository Structure

```text
Capstone/
├── README.md
├── requirements.txt
├── Data preparation.ipynb        ← Step 1: run first
├── severity_prediction.ipynb     ← Step 2
├── clustering_code.ipynb         ← Step 3
├── time_series_analysis.ipynb    ← Step 4
└── Outputs/                      ← auto-created by notebooks
    ├── df_clean_imputed.csv
    ├── df_clean_engineered.csv
    ├── comorbidity_indicators.csv
    ├── data_dictionary.csv
    └── ... (figures, feature-importance tables, cluster summaries)
```

> **Raw data is external to the repo.** Place `COMBINED_DATA_ALL_YEARS.csv` in a sibling folder
> (`../VAERS_data/`) or set the `VAERS_DATA_PATH` environment variable (see Setup below).

---

## Environment & Dependencies

**Python:** 3.12

### Required packages

| Package | Tested version |
|---------|---------------|
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.15.3 |
| matplotlib | 3.10.0 |
| statsmodels | 0.14.6 |
| joblib | 1.5.1 |

### Install

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Register the kernel with Jupyter
pip install ipykernel
python -m ipykernel install --user --name capstone-venv --display-name "Capstone (venv)"
```

A `requirements.txt` pinning the above versions is included in the repository.

---

## Input File Requirements

| Notebook | Required inputs | Where they come from |
|----------|----------------|----------------------|
| `Data preparation.ipynb` | `COMBINED_DATA_ALL_YEARS.csv` | Downloaded from Kaggle (see Dataset section) |
| `severity_prediction.ipynb` | `Outputs/df_clean_imputed.csv`, `Outputs/comorbidity_indicators.csv` | Produced by `Data preparation.ipynb` |
| `clustering_code.ipynb` | `Outputs/df_clean_imputed.csv`, `Outputs/comorbidity_indicators.csv` | Produced by `Data preparation.ipynb` |
| `time_series_analysis.ipynb` | `Outputs/df_clean_with_clusters.csv` | Produced by `clustering_code.ipynb` |

---

## Setup: Raw Data Path

The raw VAERS combined CSV is **not** in the repository (>1 GB). Two options:

**Option A — environment variable (recommended):**
```bash
export VAERS_DATA_PATH=/absolute/path/to/COMBINED_DATA_ALL_YEARS.csv
```

**Option B — default folder layout:**  
Place the file at `../VAERS_data/COMBINED_DATA_ALL_YEARS.csv` relative to this repository.
That is, the folder containing `Capstone/` should also contain a `VAERS_data/` folder with the CSV.

All other paths (input CSVs and the `Outputs/` folder) are resolved **relative to each notebook's location**
using `pathlib.Path().resolve()`, so they work on any machine without editing.

---

## Recreating the Outputs Folder

Run the notebooks **in order**. Each notebook creates or reads from the `Outputs/` subfolder automatically.

```bash
# 1 — generate cleaned + imputed dataset and comorbidity indicators
jupyter nbconvert --to notebook --execute "Data preparation.ipynb" --output "Data preparation.ipynb"

# 2 — train severity models and export feature importance tables
jupyter nbconvert --to notebook --execute "severity_prediction.ipynb" --output "severity_prediction.ipynb"

# 3 — cluster symptom narratives and characterize phenotypes
jupyter nbconvert --to notebook --execute "clustering_code.ipynb" --output "clustering_code.ipynb"

# 4 — analyze temporal trends across clusters
jupyter nbconvert --to notebook --execute "time_series_analysis.ipynb" --output "time_series_analysis.ipynb"
```

Or open each notebook in VS Code / JupyterLab and run all cells in order.

---

## Project Stages

### 1) Data Preparation (`Data preparation.ipynb`)
- Load and audit VAERS raw data
- Subset to COVID-19 reports
- Deduplicate and aggregate to one row per `VAERS_ID`
- Build composite severity label (`SERIOUS`)
- Engineer structured features: age cleanup, onset interval, year/month, dose-derived features (`MAX_DOSE`, `DOSE_COUNT`, etc.), manufacturer multi-hot encoding
- Build rule-based comorbidity indicators from free-text fields (medical history, medications, allergies, prior vaccinations, lab notes) converted to interpretable binary indicators; missingness retained as a signal
- Export: `df_clean_imputed.csv`, `df_clean_engineered.csv`, `comorbidity_indicators.csv`, `data_dictionary.csv`

### 2) Severity Prediction (`severity_prediction.ipynb`)
- Load imputed engineered data and comorbidity indicators
- Build NLP text features from symptom narratives
- Remove leakage-prone terms from supervised text view
- Combine structured + text features in pipeline
- Train/evaluate: Logistic Regression, Decision Tree, Random Forest
- Use stratified CV + grid search
- Report PR-AUC, F1, precision, recall
- Extract feature importance (structured and text signals separately)
- Compare structured-only vs. structured+text feature sets
- Measure computational efficiency and model stability across repeated CV folds
- Export feature importance, model comparisons, efficiency metrics, and confidence intervals for PR-AUC differences to `Outputs/`

### 3) Clustering (`clustering_code.ipynb`)
- Build cluster-optimized symptom narratives (removing report boilerplate while retaining clinical signal)
- TF-IDF + TruncatedSVD (LSA) dimensionality reduction (200 dimensions)
- Select k with elbow + sampled silhouette metrics across candidate k values
- Assess stability with Adjusted Rand Index (ARI) across random seeds and subsample sizes
- Fit final MiniBatchKMeans model (k=8, default)
- Interpret clusters via top reconstructed terms
- Characterize clusters across severity, age, sex, comorbidities, and manufacturers
- Export clustered dataset (`df_clean_with_clusters.csv`) for time-series stage, plus detailed cluster summaries and stability diagnostics

### 4) Time Series Analysis (`time_series_analysis.ipynb`)
- Aggregate monthly reporting volume (2020–2025)
- Model monthly report count trend with linear regression
- Model monthly severe-proportion trend with linear regression
- Track cluster prevalence trends over time
- Generate annotated timeline plots and export summaries to `Outputs/`

---

## Key Data Handling Decisions

- Case-level unit of analysis: one `VAERS_ID` per row
- Serious outcome is a composite of severe event fields
- Missingness represented both through imputation and `_MISSING` indicators
- Free-text clinical context converted into interpretable rule-based binary indicators; missingness is preserved as a potentially informative signal
- Separate text-cleaning views for supervised prediction (leakage-reduced) and clustering (admin/boilerplate reduced)

---

## Outputs

All artifacts are written to `Outputs/`:

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `df_clean_imputed.csv` | Data preparation | severity_prediction, clustering_code |
| `df_clean_engineered.csv` | Data preparation | (fallback input for severity/clustering) |
| `comorbidity_indicators.csv` | Data preparation | severity_prediction, clustering_code |
| `data_dictionary.csv` | Data preparation | — |
| `supervised_model_comparison_raw_tfidf.csv` | severity_prediction | — |
| `structured_vs_raw_tfidf_comparison.csv` | severity_prediction | — |
| `logreg_all_feature_coefficients_raw_tfidf.csv` | severity_prediction | — |
| `logreg_top_positive_features_raw_tfidf.csv`, `logreg_top_negative_features_raw_tfidf.csv` | severity_prediction | — |
| `logreg_top_positive_text_features_raw_tfidf.csv`, `logreg_top_negative_text_features_raw_tfidf.csv` | severity_prediction | — |
| `random_forest_feature_importance_raw_tfidf.csv` | severity_prediction | — |
| `decision_tree_feature_importance_raw_tfidf.csv`, `decision_tree_top_features_raw_tfidf.csv` | severity_prediction | — |
| `decision_tree_rules_raw_tfidf.txt`, `decision_tree_top_structured_features_raw_tfidf.csv`, `decision_tree_top_text_features_raw_tfidf.csv` | severity_prediction | — |
| `model_efficiency_times.csv`, `efficiency_report_table.csv` | severity_prediction | — |
| `run_metadata.json` | severity_prediction | — |
| `repeated_cv_supervised_models.csv`, `repeated_cv_supervised_models_summary.csv` | severity_prediction | — |
| `heldout_pr_auc_scores.csv`, `pr_auc_paired_bootstrap_differences.csv` | severity_prediction | — |
| `cluster_selection_metrics_cluster_cleaned.csv` | clustering_code | — |
| `cluster_stability_ari_by_sample_size_cluster_cleaned.csv` | clustering_code | — |
| `cluster_stability_ari_by_sample_size_summary_cluster_cleaned.csv` | clustering_code | — |
| `cluster_stability_ari_hist_by_sample_size_cluster_cleaned.png` | clustering_code | — |
| `cluster_top_terms_cluster_cleaned.csv` | clustering_code | — |
| `cluster_summary_cluster_cleaned.csv` | clustering_code | — |
| `cluster_row_indices.npy`, `symptom_text_tfidf_cluster_cleaned.npz`, `tfidf_vocabulary_cluster_cleaned.csv` | clustering_code | — |
| `symptom_text_lsa_200d_cluster_cleaned.npy`, `svd_components_200d_cluster_cleaned.npy` | clustering_code | — |
| `cluster_vs_serious_cluster_cleaned.csv`, `cluster_vs_sex_cluster_cleaned.csv`, `cluster_vs_manufacturer_cluster_cleaned.csv`, `cluster_vs_comorbidities_cluster_cleaned.csv` | clustering_code | — |
| `cluster_age_summary_cluster_cleaned.csv`, `cluster_sizes_cluster_cleaned.csv` | clustering_code | — |
| `symptom_text_clean_variants_cluster_only.csv` | clustering_code | — |
| `df_clean_with_clusters.csv` | clustering_code | time_series_analysis |
| `monthly_report_count.csv`, `monthly_serious_proportion.csv` | Data preparation | time_series_analysis |
| `*.png` | All notebooks | — |

## Limitations

- VAERS is a passive reporting system (underreporting and reporting bias)
- Some fields are incomplete or inconsistently populated
- Associations are not causal
- Temporal trends may reflect reporting behavior shifts, not only clinical shifts

---

## AI Tools

GitHub Copilot and ChatGPT were used selectively to assist with code refinement, debugging, and documentation, with all outputs manually reviewed and adapted for the project context.

---

## Reproducibility

1. Download the raw data from Kaggle (see Dataset section above)
2. Set `VAERS_DATA_PATH` or place the file at `../VAERS_data/COMBINED_DATA_ALL_YEARS.csv`
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order: Data preparation → severity_prediction → clustering_code → time_series_analysis
5. All intermediate and final outputs are written to `Outputs/` automatically

