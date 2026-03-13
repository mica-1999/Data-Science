import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

    # -------------------- CATEGORICAL ENCODING --------------------
    def encode_categorical(self):
        """Encode categorical columns: OneHot for few unique, LabelEncode for many unique."""
        print("Encoding categorical features...")

        # OneHotEncoding for airlines
        onehot_cols = self.config['feature_engineering']['categorical']['onehot']
        for col in onehot_cols:
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(self.df[[col]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))

            encoded_df.columns = [c.replace(col, f"encoded_{col.lower()}") for c in encoded_df.columns]
            self.df = pd.concat([self.df, encoded_df], axis=1)
            self.df.drop(columns=[col], inplace=True)

        # LabelEncoding for high-cardinality categorical columns
        label_cols = self.config['feature_engineering']['categorical']['label']
        for col in label_cols:
            le = LabelEncoder()
            self.df[f"{col}_label"] = le.fit_transform(self.df[col])
            self.df.drop(columns=[col], inplace=True)

        print("Encoding done.")

    # -------------------- BINNING --------------------
    def apply_binning(self):
        """Create bins for departure hour, flight duration, and distance."""
        # Dep hour bin
        self.df['DEP_HOUR'] = self.df['CRS_DEP_TIME'] // 100

        dep_bins = self.config['feature_engineering']['binning']['dep_hour']['bins']
        dep_labels = self.config['feature_engineering']['binning']['dep_hour']['labels']
        self.df['dep_time_bin'] = pd.cut(self.df['DEP_HOUR'], bins=dep_bins, labels=dep_labels, right=False)

        # Flight duration bin
        dur_bins = self.config['feature_engineering']['binning']['elapsed_time']['bins']
        dur_labels = self.config['feature_engineering']['binning']['elapsed_time']['labels']
        self.df['flight_duration_bin'] = pd.cut(self.df['CRS_ELAPSED_TIME'], bins=dur_bins, labels=dur_labels, right=False)

        # Distance bin
        dist_bins = self.config['feature_engineering']['binning']['distance']['bins']
        dist_labels = self.config['feature_engineering']['binning']['distance']['labels']
        self.df['distance_bin'] = pd.cut(self.df['DISTANCE'], bins=dist_bins, labels=dist_labels, include_lowest=True)

        print("Binning done.")

    # -------------------- INTERACTION FEATURES --------------------
    def add_interactions(self):
        """Create interaction features for modeling."""
        self.df['elapsed_x_distance'] = self.df['CRS_ELAPSED_TIME'] * self.df['DISTANCE']
        self.df['dep_hour_x_elapsed'] = self.df['DEP_HOUR'] * self.df['CRS_ELAPSED_TIME']
        print("Interaction features added.")

    # -------------------- TIME FEATURES --------------------
    def add_time_features(self):
        """Extract time-based features from flight date and departure hour."""
        self.df['FL_DATE'] = pd.to_datetime(self.df['FL_DATE'])
        self.df['DAY_OF_WEEK'] = self.df['FL_DATE'].dt.dayofweek
        self.df['MONTH'] = self.df['FL_DATE'].dt.month
        self.df['IS_WEEKEND'] = self.df['DAY_OF_WEEK'].isin([5,6]).astype(int)
        self.df['IS_RUSH_HOUR'] = self.df['DEP_HOUR'].between(16, 20).astype(int)
        self.df.drop(columns=['FL_DATE'], inplace=True)
        print("Time features added.")

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run all feature engineering steps in order."""
        self.encode_categorical()
        self.apply_binning()
        self.add_interactions()
        self.add_time_features()
        print("Feature engineering complete.")

        self.df.to_csv(self.config['output_dataset']['cleaned_scaled_new_features'], index=False)
        print(f"✅ Dataset saved to {self.config['output_dataset']['cleaned_scaled_new_features']}")
        return self.df