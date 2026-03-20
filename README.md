# ✈️ Flight Delay and Cancellation Analysis (2019–2023)

## 🗂️ Project Overview
This project analyzes commercial U.S. domestic flights from 2019 to 2023 to understand patterns in flight delays. The analysis covers **data preprocessing, exploratory data analysis, hypothesis testing, feature engineering, and model selection** to predict arrival delays, classify delay types, and identify operational patterns across airlines and airports.

The project is implemented in **both Python and R**, with a modular class-based Python pipeline and a Jupyter Notebook for visual analysis and reporting.

---

## 📄 Dataset
- **Source:** Kaggle – [Flight Delay and Cancellation Dataset (2019–2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data)
- **Origin:** U.S. Department of Transportation On-Time Performance Reporting System
- **Scale:** ~3 million flight records across 5 years
- **Structure:** Each row represents a single scheduled commercial domestic flight

Place the CSV file in the `ProjectDatasets/` folder as `flights_sample_3m.csv`
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
│   │   └── postFeatureEngineering/              # Feature importance plots
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
│   │   │   └── model.py                         # ModelTester class
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

## ⚙️ Pipeline — How to Run

The Python pipeline is menu-driven. Run `main.py` and select steps interactively:

```bash
python main.py
```

```
1: Run Preprocessing
2: Run EDA
3: Run PCA/UMAP
4: Run Hypothesis Testing
5: Run Feature Engineering
6: Run Modeling
0: Exit
```

> ⚠️ Steps must be run in order — preprocessing must be completed before any other step.

---

## 🔧 Configuration

All pipeline parameters are controlled via `config.yaml`:
- Dataset paths and output directories
- Distance filtering range
- Columns to drop (leakage prevention)
- Columns for outlier handling and scaling
- EDA plot configurations
- Dimensionality reduction parameters (PCA/UMAP)
- Hypothesis testing settings
- Feature engineering binning and encoding settings
- Modeling hyperparameters

---

## 📊 What Was Done — Part 1 Summary

### Phase 1 — Problem Formulation
- Defined the regression, classification, and clustering tasks
- Specified success criteria and data leakage prevention strategy

### Phase 2 — Data Analysis & Cleansing

**Preprocessing:**
- Removed cancelled and diverted flights
- Dropped rows with missing `ARR_DELAY` or `CRS_ELAPSED_TIME`
- Filtered invalid distances (< 50 or > 5500 miles)
- Removed outliers using IQR method on `CRS_ELAPSED_TIME`
- Dropped all post-event and leakage columns
- Applied StandardScaler and MinMaxScaler to key numeric features

**EDA:**
- Descriptive statistics for key numeric features
- Histograms, boxplots, KDE plots, and scatter plots
- Correlation heatmap across numeric features
- PCA (linear) and UMAP (non-linear) dimensionality reduction

**Hypothesis Testing (5 hypotheses):**
- H1: Pearson correlation — flight distance vs arrival delay
- H2: Welch's t-test — Southwest vs Delta mean delays
- H3: ANOVA — mean delays across all airlines
- H4: Pearson correlation — weather delays vs arrival delay
- H5: ANOVA — departure hour vs arrival delay

### Phase 3 — Model Selection

**Feature Engineering (13+ new features):**
- One-Hot Encoding: `AIRLINE_CODE`
- Label Encoding: `ORIGIN`, `DEST`
- Binning: departure hour, flight duration, distance
- Interaction features: `elapsed_x_distance`, `dep_hour_x_elapsed`
- Time features: `DAY_OF_WEEK`, `MONTH`, `IS_WEEKEND`, `IS_RUSH_HOUR`

**Models Evaluated:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 23.741 | 51.188 | 0.013 |
| Random Forest | 23.485 | 51.164 | 0.013 |

> Both models show limited predictive performance, expected given the strict exclusion of post-event variables to prevent data leakage. The models represent a realistic pre-departure prediction scenario.

---

## 🔒 Data Leakage Prevention
The following columns were excluded from all predictive models as they contain post-event information:
`DEP_DELAY`, `ARR_TIME`, `DEP_TIME`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_OUT`, `TAXI_IN`, `ELAPSED_TIME`, `AIR_TIME`, `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, `DELAY_DUE_LATE_AIRCRAFT`

---

## 📚 References
- Flight Delay and Cancellation Dataset — Kaggle: [Link](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data)
- U.S. Department of Transportation On-Time Performance Reporting System