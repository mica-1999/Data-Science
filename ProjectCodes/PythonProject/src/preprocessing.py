import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.csv_path = config['preprocessing']['csv_path']
        self.min_distance = config['preprocessing']['distance_range']['min']
        self.max_distance = config['preprocessing']['distance_range']['max']
        self.future_columns = config['preprocessing']['future_columns']
        self.numeric_columns_for_outliers = config['preprocessing']['numeric_columns_for_outliers']
        self.output_path_cleaned = config['output_dataset']['cleaned']
        self.scaling_columns = config['preprocessing']['scaling_columns']
        self.df = None
        self.df_eda = None
        self.df_hyp = None
        self.df_cleaned = None

    # -------------------- LOAD DATA --------------------
    def load_data(self):
        """Load CSV and make a copy for EDA with error handling."""
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {self.csv_path}. Please check the path.")
        except Exception as e:
            print(f"An unexpected error occurred while loading the CSV: {e}")

    # -------------------- ROW CLEANING --------------------
    def initial_cleaning(self):
        """Remove rows with nulls and irrelevant values."""
        self.df.dropna(subset=['CRS_ELAPSED_TIME'], inplace=True)
        self.df = self.df[self.df['CANCELLED'] == 0]
        self.df = self.df[self.df['DIVERTED'] == 0]
        self.df = self.df.dropna(subset=['ARR_DELAY'])
        self.df = self.df[(self.df['DISTANCE'] >= self.min_distance) &
                          (self.df['DISTANCE'] <= self.max_distance)].reset_index(drop=True)

        # Copy for eda and hyp
        self.df_hyp = self.df.copy()
        self.df_eda = self.df.copy() # Only moved here at end of project

    # -------------------- COL CLEANING --------------------
    def drop_future_columns(self):
        """Drop columns that won't help for prediction (future info)."""
        self.df.drop(columns=self.future_columns, inplace=True, errors="ignore")

    # -------------------- OUTLIER HANDLING --------------------
    def handle_outliers(self, numeric_cols=None):
        """Remove outliers using IQR and fill NaNs with median."""
        if numeric_cols is None:
            numeric_cols = self.numeric_columns_for_outliers
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df[col] = np.where((self.df[col] < lower) | (self.df[col] > upper), np.nan, self.df[col])
            self.df[col] = self.df[col].fillna(self.df[col].median())
        self.df = self.df.reset_index(drop=True)

    # -------------------- SAVE CLEAN SET --------------------
    def save_cleaned_copy(self, output_path: str = None):
        """Save a copy of the preprocessed dataset in memory and optionally to CSV."""
        self.df_cleaned = self.df.copy()
        if output_path is not None:
            self.df_cleaned.to_csv(output_path, index=False)

    # -------------------- SCALING FOR PCA/UMAP --------------------
    def apply_scaling(self):
        """Apply standardization and normalization to selected numeric columns."""
        numeric_cols = self.scaling_columns

        # Standardization
        standard_scaler = StandardScaler()
        df_scaled_std = pd.DataFrame(
            standard_scaler.fit_transform(self.df_cleaned[numeric_cols]),
            columns=[f"{col}_std" for col in numeric_cols]
        ).reset_index(drop=True)

        # Normalization
        minmax_scaler = MinMaxScaler()
        df_scaled_minmax = pd.DataFrame(
            minmax_scaler.fit_transform(self.df_cleaned[numeric_cols]),
            columns=[f"{col}_minmax" for col in numeric_cols]
        ).reset_index(drop=True)

        # Combine with dataset
        self.df = pd.concat(
            [self.df_cleaned.reset_index(drop=True), df_scaled_std, df_scaled_minmax],
            axis=1
        )

        print("Standardization and normalization done.")

    # -------------------- RUN PREPROCESSING --------------------
    def preprocess(self):
        """Run all preprocessing steps in order."""
        print("\n" + "="*20 + " PREPROCESSING " + "="*20)

        self.load_data()
        print(f"Data loaded from: {self.config['preprocessing']['csv_path']}")
        print(f"Original dataset shape: {self.df.shape}")

        self.initial_cleaning()
        print(f"After cleaning (nulls, cancelled/diverted, distance filter): {self.df.shape}")

        self.drop_future_columns()
        print(f"Columns dropped (future info): {self.future_columns}")

        self.handle_outliers()
        print(f"Outliers handled for columns: {self.numeric_columns_for_outliers}")

        self.save_cleaned_copy(output_path=self.output_path_cleaned)
        print(f"Cleaned dataset saved to: {self.output_path_cleaned}")

        self.apply_scaling()
        print(f"Scaling applied on columns: {self.scaling_columns}")

        print("Preview of cleaned dataset:")
        print(self.df.head())