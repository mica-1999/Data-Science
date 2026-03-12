import pandas as pd
import numpy as np
import yaml

# Load the YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.csv_path = config['preprocessing']['csv_path']
        self.min_distance = config['preprocessing']['distance_range']['min']
        self.max_distance = config['preprocessing']['distance_range']['max']
        self.future_columns = config['preprocessing']['future_columns']
        self.numeric_columns_for_outliers = config['preprocessing']['numeric_columns_for_outliers']
        self.df = None
        self.df_eda = None
        self.df_hyp = None
        self.df_cleaned = None

    def load_data(self):
        """Load CSV and make a copy for EDA."""
        self.df = pd.read_csv(self.csv_path)
        self.df_eda = self.df.copy()
        print("Data loaded. Shape:", self.df.shape)

    def initial_cleaning(self):
        """Remove rows with nulls and irrelevant values."""
        self.df.dropna(subset=['CRS_ELAPSED_TIME'], inplace=True)
        self.df = self.df[self.df['CANCELLED'] == 0]
        self.df = self.df[self.df['DIVERTED'] == 0]
        self.df = self.df.dropna(subset=['ARR_DELAY'])
        self.df = self.df[(self.df['DISTANCE'] >= self.min_distance) &
                          (self.df['DISTANCE'] <= self.max_distance)].reset_index(drop=True)

        """ Copy for hypothesis testing """
        self.df_hyp = self.df.copy()

    def drop_future_columns(self):
        """Drop columns that won't help for prediction (future info)."""
        cols_to_drop = self.future_columns
        self.df.drop(columns=cols_to_drop, inplace=True)

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

    def save_cleaned_copy(self, output_path: str = None):
        """Save a copy of the preprocessed dataset in memory and optionally to CSV."""
        self.df_cleaned = self.df.copy()
        if output_path is not None:
            self.df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned dataset saved to {output_path}")

    def preprocess(self):
        """Run all preprocessing steps in order."""
        print("Starting preprocessing...")
        self.load_data()
        self.initial_cleaning()
        self.drop_future_columns()
        self.handle_outliers()
        self.save_cleaned_copy(output_path=config['preprocessing']['output_path_cleaned'])
        print("Processing done. Final shape:", self.df.shape)