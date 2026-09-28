# Gut-Brain-Axis Modulation in Alzheimer's Disease: A Multiomics Approach

Integrated multiomics analysis of gut microbiome, saliva microbiome, and plasma metabolomics data to predict Alzheimer's disease severity and uncover gut-brain axis biomarkers.

## Overview

This project applies machine learning to a multiomics dataset (73 samples) spanning three molecular layers to classify Alzheimer's disease severity (ADAS-COG groups) and identify biologically meaningful biomarkers linking peripheral molecular changes to cognitive decline.

## Dataset

| Omics Layer | Samples | Features (raw → processed) | Platform |
|---|---|---|---|
| Gut microbiome | 78 | 364 → 125 | 16S rRNA sequencing |
| Saliva microbiome | 78 | 239 → 96 | 16S rRNA sequencing |
| Plasma metabolomics | 87 | 1,143 → 910 | LC-MS |
| **Integrated dataset** | **73** | **1,179 total** | — |

## Methodology

- **Preprocessing:** Missing value filtering (>50% threshold), KNN imputation (microbiome) / median imputation (metabolomics), CLR transformation, standardization
- **Exploratory analysis:** PCA, t-SNE, UMAP, Spearman correlation analysis
- **Integration strategy:** Early (concatenation-based) integration across omics layers
- **Feature selection:** ANOVA F-statistics, Spearman correlation, Random Forest importance (union → 79 features)
- **Models:** Random Forest, XGBoost, Logistic Regression
- **Validation:** Stratified 5-fold cross-validation, paired t-tests for model comparison

## Key Results

- **Random Forest** achieved the best performance: **67.0% ± 15.4% accuracy** (outperforming XGBoost, p=0.041, and Logistic Regression, p=0.023)
- Plasma metabolomics contributed the strongest discriminatory signal (67 of 79 selected features)
- Top biomarkers: **beta-alanine**, **kynurenine**, phosphatidylethanolamine species, *Paraprevotella clara*, *Gemmiger formicilis*

## Repository Structure

```
├── data/                  # Raw and processed datasets (or data access instructions)
├── notebooks/             # Jupyter notebooks for each analysis stage
├── src/                   # Preprocessing, feature selection, modeling scripts
├── results/               # Figures, tables, model outputs
├── presentation/          # Dissertation defense slides
├── requirements.txt       # Python dependencies
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

Key libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `matplotlib`, `seaborn`, `umap-learn`

## Usage

```bash
# Example workflow
python src/preprocessing.py
python src/feature_selection.py
python src/train_models.py
```

## Limitations

Cross-sectional design, modest sample size (n=73), no dietary covariates, single-cohort study — see full discussion in the dissertation.

## Contact

Ashutosh Jha — jhaashu9897@gmail.com
