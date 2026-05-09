# ✈️ Flight Delay and Cancellation Analysis (2019–2023)

## 🗂️ Project Overview
This project analyzes ~3 million commercial U.S. domestic flight records from 2019 to 2023, building a full end-to-end machine learning pipeline to:
 
- **Predict** arrival delay duration in minutes (regression)
- **Classify** flights as on-time, short delay, or long delay (classification)
- **Cluster** flights to identify operational patterns across airlines and airports
- **Test** statistical hypotheses about factors driving flight delays
Implemented in **Python and R**, with a modular class-based pipeline, config-driven architecture, and a Jupyter Notebook for reporting.
 
---

## 📄 Dataset
- **Source:** Kaggle – [Flight Delay and Cancellation Dataset (2019–2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data)
- **Origin:** U.S. Department of Transportation On-Time Performance Reporting System
- **Scale:** ~3 million flight records across 5 years
- **Structure:** Each row represents a single scheduled commercial domestic flight

> ⚠️ **The dataset is not included in this repository due to its size.**
> Download it from the link above and place the CSV file in the `ProjectDatasets/` folder as `flights_sample_3m.csv`.
---

## 🎯 Project Objectives
- **Regression:** Predict arrival delay duration in minutes using only pre-departure information
- **Classification:** Categorize flights into:
  - ✅ On-time (< 15 min delay)
  - ⏱️ Short delay (15–30 min)
  - ⏳ Long delay (> 30 min)
- **Clustering:** Identify operational patterns across airlines and airports based on delay behavior
- **Hypothesis Testing:** Statistically assess factors influencing flight delays

---

## 📁 Project Structure

```
project/
│
├── ProjectDatasets/
│   ├── flights_sample_3m.csv                    # Raw dataset (3M rows)
│   ├── flights_cleaned.csv                      # Cleaned dataset (post-preprocessing) Python
│   ├── flights_clean_scaled_featured.csv        # Cleaned + scaled + featured Python
│   ├── flights_cleaned_r.csv                    # Cleaned dataset (post-preprocessing) R
│   └── flights_clean_scaled_featured_r.csv      # Cleaned + scaled + featured R
│
├── OutputFiles/
│   ├── Python/
│   │   ├── preFeatureEngineering/               # EDA, PCA, UMAP plots
│   │   ├── postFeatureEngineering/              # Feature importance plots
│   │   ├── modelResults/                        # CSV evaluation tables
│   │   ├── trainedModels/                      # Saved trained ML/DL models
│   │   └── modelGraphics/                       # Model plots and visualizations
│   └── R/
│       ├── preFeatureEngineering/               # R EDA, PCA, UMAP plots
│       └── postFeatureEngineering/              # R Feature importance plots
│
├── ProjectCodes/
│   ├── config.yaml                              # Shared pipeline configuration
│   ├── ProjetoCDJupyter.ipynb                   # Jupyter Notebook report
│   ├── PythonProject/
│   │   ├── src/
│   │   │   ├── preprocessing.py                 # DataPreprocessor class
│   │   │   ├── eda.py                           # EDAAnalyzer class
│   │   │   ├── pca_umap.py                      # DimensionalityReducer class
│   │   │   ├── hypothesis.py                    # HypothesisTester class
│   │   │   ├── feature_engineering.py           # FeatureEngineer class
│   │   │   ├── model.py                         # ModelTester class
│   │   │   ├── kNN.py                           # Custom kNN implementation
│   │   │   ├── supervised_learning.py           # Decision Tree + SVR
│   │   │   ├── ensemble_learning.py             # Random Forest + Gradient Boosting
│   │   │   ├── deep_learning.py                 # PyTorch MLP model
│   │   │   └── clustering.py                    # KMeans + DBSCAN
│   │   └── main.py                              # Interactive pipeline entry point
│   └── RProject/
│       ├── src/
│       │   ├── preprocessing.R                  # DataPreprocessor R6 class
│       │   ├── eda.R                            # EDAAnalyzer R6 class
│       │   ├── pca_umap.R                       # DimensionalityReducer R6 class
│       │   ├── hypothesis.R                     # HypothesisTester R6 class
│       │   ├── feature_engineering.R            # FeatureEngineer R6 class
│       │   └── model.R                          # ModelTester R6 class
│       └── main.R                               # Interactive pipeline entry point
│
└── ProjectReport/
    └── Report.pdf                               # Mid-journey report (Part 1)
```

---

## ⚙️ How to Run
 
The Python pipeline is fully menu-driven. Run `main.py` and select steps interactively:
 
```bash
cd ProjectCodes/PythonProject
python main.py
```
 
```
1:  Run Preprocessing
2:  Run EDA
3:  Run PCA/UMAP
4:  Run Hypothesis Testing
5:  Run Feature Engineering
6:  Run Post-Engineering EDA
7:  Run Modeling (Baseline — Linear Regression)
8:  Run kNN from Scratch
9:  Run Supervised Learning Models
10: Run Ensemble Learning
11: Run Deep Learning Model
12: Run Clustering Analysis
0:  Exit
```

> ⚠️ **Steps must be run in order.** Preprocessing (step 1) and Feature Engineering (step 5) must be completed before running any model steps.
 
> 📝 The R pipeline follows the same structure but does not include Post-Engineering EDA or Phase 2 model steps.
 
---

## 🔧 Configuration
 
All pipeline parameters are controlled via `config.yaml`:
 
- Dataset paths and output directories
- Distance filtering range
- Columns to drop (leakage prevention)
- Outlier handling and scaling columns
- EDA plot configurations
- PCA/UMAP dimensionality reduction parameters
- Hypothesis testing settings
- Feature engineering binning and encoding
- Modeling hyperparameters (k values, sample sizes, tree depths, etc.)
---

## 📊 What Was Done

### Part 1 — Phases 1–3
 
**Phase 1 — Problem Formulation**

- Defined regression, classification, and clustering tasks
- Specified success criteria and leakage prevention strategy

**Phase 2 — Data Analysis & Cleansing**
 
*Preprocessing:*
- Removed cancelled and diverted flights
- Dropped rows with missing `ARR_DELAY` or `CRS_ELAPSED_TIME`
- Filtered invalid distances (< 50 or > 5500 miles)
- Removed outliers using IQR method
- Dropped all post-event leakage columns
- Applied `StandardScaler` and `MinMaxScaler` to numeric features

*EDA:*
- Descriptive statistics, histograms, boxplots, KDE plots, scatter plots
- Correlation heatmap
- PCA (linear) and UMAP (non-linear) dimensionality reduction

*Hypothesis Testing (5 hypotheses):*
- H1: Pearson correlation — distance vs arrival delay
- H2: Welch's t-test — Southwest vs Delta mean delays
- H3: ANOVA — mean delays across all airlines
- H4: Pearson correlation — weather delays vs arrival delay
- H5: ANOVA — departure hour vs arrival delay

**Phase 3 — Model Selection & Feature Engineering**
 
*13+ engineered features:*
- One-Hot Encoding: `AIRLINE_CODE`
- Label Encoding: `ORIGIN`, `DEST`
- Binning: departure hour, flight duration, distance
- Interaction features: `elapsed_x_distance`, `dep_hour_x_elapsed`
- Time features: `DAY_OF_WEEK`, `MONTH`, `IS_WEEKEND`, `IS_RUSH_HOUR`

---

### Part 2 — Phases 4–6
 
**Phase 4 — Model Building**
 
| Model | Type | Notes |
|-------|------|-------|
| 🔢 **kNN (from scratch)** | Regression + Classification | Pure NumPy implementation using Euclidean distance |
| 🌳 **Decision Tree Regressor** | Regression | Rule-based regression with feature importance |
| 📐 **Support Vector Regressor (SVR)** | Regression | RBF kernel with scaled features |
| 🌲 **Random Forest Regressor** | Regression (Bagging) | Bootstrap aggregation ensemble |
| 🚀 **Gradient Boosting Regressor** | Regression (Boosting) | Sequential boosting ensemble |
| 🧠 **PyTorch MLP** | Regression | Feed-forward neural network with early stopping |
| 🔵 **KMeans** | Clustering | Evaluated using elbow method and silhouette score |
| 🌑 **DBSCAN** | Clustering | Density-based clustering and anomaly detection |
 
**Phase 5 — Model Comparison**
 
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 19.474 | 47.859 | 0.011 |
| Decision Tree Regressor | 19.276 | 47.772 | 0.014 |
| Support Vector Regressor | 13.541 | 51.178 | -0.055 |
| Random Forest Regressor | 19.169 | 47.636 | 0.020 |
| Gradient Boosting Regressor | 19.298 | 47.716 | 0.017 |
| PyTorch Feed-Forward Neural Network | 19.357 | 51.918 | 0.011 |
| kNN Regressor (k=29) | 20.619 | 59.068 | -0.024 |
 
**Phase 6 — Operationalization Planning**
- Trained models saved for future reuse
- Final report and presentation preparation planned
- Project structured for reproducibility and future deployment  

---

## 🔍 Key Findings
 
- Flight delay prediction is highly noisy when restricted to pre-departure information only
- Strict leakage prevention significantly reduced artificially inflated model performance
- Ensemble methods slightly outperformed simpler baseline models
- kNN struggled with class imbalance — majority of flights are on-time
- Deep learning did not substantially outperform traditional ML approaches
- Clustering revealed only weak natural operational separation between flights
- Most models converged to similar MAE values around **19–21 minutes**
---

## 🔒 Data Leakage Prevention
 
The following columns were excluded from all predictive models as they contain post-departure information:
 
`DEP_DELAY` · `ARR_TIME` · `DEP_TIME` · `WHEELS_OFF` · `WHEELS_ON` · `TAXI_OUT` · `TAXI_IN` · `ELAPSED_TIME` · `AIR_TIME` · `DELAY_DUE_CARRIER` · `DELAY_DUE_WEATHER` · `DELAY_DUE_NAS` · `DELAY_DUE_SECURITY` · `DELAY_DUE_LATE_AIRCRAFT`
 
---

## 📚 References
- [Flight Delay and Cancellation Dataset — Kaggle](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data)
- [U.S. Department of Transportation On-Time Performance Reporting System](https://www.transtats.bts.gov/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)
