#%% Phase 3: Model Selection / Baseline Modeling

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTester:
    """
    Baseline model used during Phase 3: Model Selection.

    Linear Regression is used as a simple baseline to evaluate whether
    the selected features have a basic linear relationship with ARR_DELAY.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config
        self.target_col = config['modeling']['target_col']
        self.test_size = config['modeling']['test_size']
        self.random_state = config['modeling']['random_state']
        self.drop_cols = config['modeling']['drop_columns']

        self.output_dir_results = config['output_dir_model_results']
        os.makedirs(self.output_dir_results, exist_ok=True)

    # -------------------- PREPARING DATA --------------------
    def prepare_data(self):
        """Prepare X/y and split into train and test sets."""

        X = self.df.drop(columns=self.drop_cols, errors='ignore')
        y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.X = X
        return self.X_train, self.X_test, self.y_train, self.y_test

    # -------------------- LINEAR REGRESSION --------------------
    def run_linear_regression(self):
        """Train and evaluate Linear Regression baseline."""

        print("\nRunning Linear Regression baseline...")

        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)

        y_pred = lr.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        print("Linear Regression complete.")
        print(f"MAE : {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2  : {r2:.3f}")

        self.lr_model = lr
        self.lr_pred = y_pred
        self.lr_results = {
            "Model": "Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        results_path = os.path.join(
            self.output_dir_results,
            "linear_regression_results.csv"
        )
        pd.DataFrame([self.lr_results]).to_csv(results_path, index=False)
        print(f"Linear Regression results saved to: {results_path}")

        return self.lr_results

    # -------------------- RUN ALL --------------------
    def run_all_models(self):
        """Run baseline model."""
        print("\n" + "=" * 20 + " BASELINE MODELING " + "=" * 20)

        if not hasattr(self, "X_train"):
            self.prepare_data()

        results = self.run_linear_regression()

        print("Baseline modeling complete.")
        return results