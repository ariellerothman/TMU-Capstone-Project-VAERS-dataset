# COVID-19 Vaccine Adverse Event Phenotyping and Severity Prediction

## Project Overview

This project applies natural language processing (NLP) and predictive analytics to COVID-19 vaccine-related reports from the Vaccine Adverse Event Reporting System (VAERS) spanning 2020–2025. By combining text mining, classification, clustering, and temporal analysis, we demonstrate how machine learning methods can enhance interpretation of large-scale vaccine safety surveillance data.

### Research Questions
1. Can symptom narratives improve prediction of severe adverse event outcomes beyond structured demographic and medical history variables?
2. Do unsupervised clustering methods identify stable adverse event phenotypes?
3. Do reporting volumes, severity rates, and narrative clusters shift over time?
4. Can NLP models distinguish COVID-19 from non-COVID-19 vaccine narratives?

---

## Dataset

**Source:** Adverse Drug Effects (ADE) Detection dataset (Kaggle) compiled from VAERS  
**Time Period:** December 12, 2020 – July 25, 2025  
**Original Records:** 2,132,431 total adverse event reports  
**COVID-19 Subset:** 986,096 unique case-level records (VAERS_ID)  

### Key Characteristics
- **19.85%** classified as serious outcomes (composite: death, life-threatening, hospitalization, ER visit, disability)
- **Structured variables:** Demographics (age, sex, state), vaccine info (manufacturer, type), dates, binary outcomes
- **Unstructured variables:** Symptom narratives (~727 chars mean, ~313 chars median), medical history, allergies, medications, lab data, prior vaccinations
- **Missingness:** Moderate (11–14%) in numeric fields; high (50–70%) in free-text clinical context
- **Sex distribution:** 63.9% female, 31.9% male, 4.2% unknown

---

## Repository Structure

```text
/Capstone
├── README.md
├── COMBINED_DATA_ALL_YEARS.csv
├── Data preparation.ipynb
├── severity_prediction.ipynb
├── clustering_code.ipynb
├── time_series_analysis.ipynb
└── Outputs/
    ├── df_clean_engineered.csv
    ├── df_clean_imputed.csv
    ├── df_clean_with_clusters.csv
    ├── comorbidity_indicators.csv
    ├── monthly_reporting_summary_2020_2025.csv
    ├── cluster_top_terms_cluster_cleaned.csv
    └── ...other model and figure outputs...
```

---

## Project Stages

### 1) Data Preparation (`Data preparation.ipynb`)
- Load and audit VAERS raw data
- Subset to COVID-19 reports
- Deduplicate and aggregate to one row per `VAERS_ID`
- Build composite severity label (`SERIOUS`)
- Engineer structured features:
  - age cleanup, onset interval, year/month
  - dose-derived features (`MAX_DOSE`, `DOSE_COUNT`, etc.)
  - manufacturer multi-hot encoding
- Build rule-based comorbidity indicators from free-text fields
- Export final engineered dataset

### 2) Severity Prediction (`severity_prediction.ipynb`)
- Load imputed engineered data
- Build NLP text features from symptom narratives (TF-IDF unigrams + bigrams)
- Remove leakage-prone terms from supervised text view
- Combine structured + text features in pipeline
- Train/evaluate:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Use stratified CV + grid search
- Report PR-AUC, F1, precision, recall, confusion matrix
- Export feature importance and model comparison outputs

### 3) Clustering (`clustering_code.ipynb`)
- Build cluster-optimized symptom text view (admin/boilerplate cleanup)
- TF-IDF + TruncatedSVD (LSA) dimensionality reduction
- Select k with elbow + sampled silhouette
- Assess stability with ARI across seeds
- Fit final MiniBatchKMeans model
- Interpret clusters via top reconstructed terms
- Characterize clusters vs severity, age, sex, comorbidities, manufacturers
- Export clustered dataset for time-series stage

### 4) Time Series Analysis (`time_series_analysis.ipynb`)
- Aggregate monthly reporting volume (2020–2025)
- Model monthly report count trend with linear regression
- Model monthly severe-proportion trend with linear regression
- Track cluster prevalence trends over time
- Generate annotated timeline plots and export summaries

---

## Methods Summary

- **Structured ML:** Logistic Regression, Decision Tree, Random Forest  
- **NLP:** regex preprocessing + TF-IDF (unigrams/bigrams)  
- **Unsupervised:** MiniBatchKMeans on LSA-transformed text  
- **Temporal:** monthly aggregation + OLS trend models  
- **Validation:** stratified CV, grid search, ARI stability checks  

---

## Key Data Handling Decisions

- Case-level unit of analysis: one `VAERS_ID` per row
- Serious outcome is a composite of severe event fields
- Missingness represented both through imputation and `_MISSING` indicators
- Free-text clinical context converted into interpretable rule-based binary indicators
- Separate text-cleaning views for:
  - supervised prediction (leakage-reduced)
  - clustering (admin/boilerplate reduced)

---

## Outputs

Main artifacts are written to `Outputs/`, including:
- cleaned and engineered datasets
- comorbidity indicators
- model performance summaries
- feature importance tables
- cluster interpretations and summaries
- monthly trend and regression outputs
- figures for reporting

---

## Limitations

- VAERS is a passive reporting system (underreporting and reporting bias)
- Some fields are incomplete or inconsistently populated
- Associations are not causal
- Temporal trends may reflect reporting behavior shifts, not only clinical shifts

---

## Reproducibility

Recommended run order:
1. `Data preparation.ipynb`
2. `severity_prediction.ipynb`
3. `clustering_code.ipynb`
4. `time_series_analysis.ipynb`

Use the same `Outputs/` directory to preserve notebook dependencies.

