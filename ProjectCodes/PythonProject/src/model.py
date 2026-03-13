#%% 8- Phase 3: Model Selection / Classes for Modeling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelTester:
    def __init__(self, df: pd.DataFrame, config: dict = None):
        self.df = df.copy()
        self.target_col = config['modeling']['target_col']
        self.test_size = config['modeling']['test_size']
        self.random_state = config['modeling']['random_state']
        self.drop_cols = config['modeling']['drop_columns']
        self.rf_n_estimators = config['modeling']['random_forest']['n_estimators']
        self.rf_max_depth = config['modeling']['random_forest']['max_depth']
        self.rf_plot_top = config['modeling']['random_forest']['plot_top_features']
        self.rf_save_path = config['modeling']['random_forest']['save_feature_importance_path']

    # -------------------- PREPARING DATA FOR MODELS  --------------------
    def prepare_data(self):
        """Prepare feature matrix X and target vector y, and split into train/test sets"""

        X = self.df.drop(columns=self.drop_cols, errors='ignore')
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.X = X
        return self.X_train, self.X_test, self.y_train, self.y_test

    # -------------------- LINEAR REGRESSION MODEL --------------------
    def run_linear_regression(self):
        """Fit Linear Regression and print evaluation metrics"""
        print("\nRunning Linear Regression...")
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        print("Linear Regression complete.")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.3f}")

        self.lr_model = lr
        self.lr_pred = y_pred

    # -------------------- RANDOM FOREST MODEL --------------------
    def run_random_forest(self):
        """Fit Random Forest Regressor, print evaluation metrics, plot feature importance"""
        print("\nRunning Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        y_pred = rf.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        print("Random Forest complete.")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values(by='importance', ascending=False)

        print("\nTop Features by Importance:")
        print(feature_importance.head(self.rf_plot_top))

        # Plot top features
        plt.figure(figsize=(10,6))
        plt.barh(
            feature_importance['feature'].head(self.rf_plot_top)[::-1],
            feature_importance['importance'].head(self.rf_plot_top)[::-1]
        )
        plt.xlabel("Importance")
        plt.title("Top Features - Random Forest")
        plt.savefig(self.rf_save_path, bbox_inches='tight')
        plt.show()
        print(f"Feature importance plot saved to: {self.rf_save_path}")

        self.rf_model = rf
        self.rf_pred = y_pred
        self.rf_feature_importance = feature_importance

    # -------------------- RUN ALL  --------------------
    def run_all_models(self):
        """Run both Linear Regression and Random Forest sequentially"""
        print("\n" + "=" * 20 + " MODELING " + "=" * 20)
        self.run_linear_regression()
        self.run_random_forest()
        print("Modeling complete.")