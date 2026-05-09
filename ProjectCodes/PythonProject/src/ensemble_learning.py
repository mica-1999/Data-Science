#%% Phase 4: Model Building / Ensemble Learning Models

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EnsembleLearningRunner:
    """
    Runs ensemble learning models for the flight delay prediction task.

    This module covers the ensemble requirement of the project:
        1. Bagging method  -> Random Forest Regressor
        2. Boosting method -> Gradient Boosting Regressor

    Both models are evaluated using regression metrics:
        - MAE
        - RMSE
        - R2
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

        self.target_col = config["modeling"]["target_col"]
        self.test_size = config["modeling"]["test_size"]
        self.random_state = config["modeling"]["random_state"]
        self.drop_cols = config["modeling"]["drop_columns"]

        self.output_dir_results = config["output_dir_model_results"]
        self.output_dir_graphics = config["output_dir_model_graphics"]
        self.output_dir_trained_models = config["output_dir_trained_models"]

        os.makedirs(self.output_dir_results, exist_ok=True)
        os.makedirs(self.output_dir_graphics, exist_ok=True)
        os.makedirs(self.output_dir_trained_models, exist_ok=True)

        rf_cfg = config.get("ensemble_learning", {}).get(
            "random_forest",
            config.get("modeling", {}).get("random_forest", {})
        )

        self.rf_n_estimators = rf_cfg.get("n_estimators", 100)
        self.rf_max_depth = rf_cfg.get("max_depth", 15)
        self.rf_min_samples_leaf = rf_cfg.get("min_samples_leaf", 20)
        self.rf_plot_top = rf_cfg.get("plot_top_features", 10)

        gb_cfg = config.get("ensemble_learning", {}).get("gradient_boosting", {})

        self.gb_n_estimators = gb_cfg.get("n_estimators", 150)
        self.gb_learning_rate = gb_cfg.get("learning_rate", 0.05)
        self.gb_max_depth = gb_cfg.get("max_depth", 3)
        self.gb_min_samples_leaf = gb_cfg.get("min_samples_leaf", 30)
        self.gb_plot_top = gb_cfg.get("plot_top_features", 10)

        self.results = []

    # -------------------- PREPARE DATA --------------------
    def prepare_data(self):
        """Prepare X/y and split the data into train/test sets."""

        X = self.df.drop(columns=self.drop_cols, errors="ignore")
        y = self.df[self.target_col]

        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.X = X

        print(f"Train: {self.X_train.shape} | Test: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    # -------------------- EVALUATE REGRESSION --------------------
    def _evaluate_regression(self, model_name: str, y_pred: np.ndarray):
        """Calculate and print regression metrics."""

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        print(f"\n{model_name} results:")
        print(f"MAE : {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2  : {r2:.3f}")

        result = {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        self.results.append(result)

        return result

    # -------------------- PLOT ACTUAL VS PREDICTED --------------------
    def _plot_actual_vs_predicted(self, y_pred: np.ndarray, model_name: str, filename: str):
        """Plot actual ARR_DELAY values against predicted values."""

        plt.figure(figsize=(7, 5))
        plt.scatter(self.y_test, y_pred, alpha=0.25)

        plt.xlabel("Actual ARR_DELAY")
        plt.ylabel("Predicted ARR_DELAY")
        plt.title(f"{model_name}: Actual vs Predicted")

        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())

        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        save_path = os.path.join(self.output_dir_graphics, filename)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Actual vs predicted plot saved: {save_path}")

    # -------------------- PLOT RESIDUALS --------------------
    def _plot_residuals(self, y_pred: np.ndarray, model_name: str, filename: str):
        """Plot residual distribution: actual - predicted."""

        residuals = self.y_test - y_pred

        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=50)

        plt.xlabel("Residuals (Actual - Predicted)")
        plt.ylabel("Frequency")
        plt.title(f"{model_name}: Residual Distribution")

        save_path = os.path.join(self.output_dir_graphics, filename)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Residual plot saved: {save_path}")

    # -------------------- SAVE FEATURE IMPORTANCE --------------------
    def _save_feature_importance(
        self,
        model,
        model_name: str,
        filename_csv: str,
        filename_plot: str,
        top_n: int
    ):
        """Save and plot feature importance for tree-based ensemble models."""

        feature_importance = pd.DataFrame({
            "feature": self.X.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        csv_path = os.path.join(self.output_dir_results, filename_csv)
        feature_importance.to_csv(csv_path, index=False)

        print(f"\nTop {model_name} features:")
        print(feature_importance.head(top_n))
        print(f"Feature importance saved: {csv_path}")

        plt.figure(figsize=(10, 6))

        top_features = feature_importance.head(top_n)

        plt.barh(
            top_features["feature"][::-1],
            top_features["importance"][::-1]
        )

        plt.xlabel("Importance")
        plt.title(f"Top Features - {model_name}")

        plot_path = os.path.join(self.output_dir_graphics, filename_plot)

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        print(f"Feature importance plot saved: {plot_path}")

        return feature_importance

    # -------------------- RANDOM FOREST: BAGGING --------------------
    def run_random_forest(self):
        """
        Run Random Forest Regressor.

        Random Forest is a bagging ensemble because it trains many decision trees
        independently on bootstrap samples and averages their predictions.
        """

        print("\n" + "=" * 20 + " RANDOM FOREST REGRESSOR (BAGGING) " + "=" * 20)

        rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )

        rf.fit(self.X_train, self.y_train)

        y_pred = rf.predict(self.X_test)

        result = self._evaluate_regression(
            "Random Forest Regressor (Bagging)",
            y_pred
        )

        self._plot_actual_vs_predicted(
            y_pred,
            "Random Forest Regressor",
            "random_forest_actual_vs_predicted.png"
        )

        self._plot_residuals(
            y_pred,
            "Random Forest Regressor",
            "random_forest_residuals.png"
        )

        self.rf_feature_importance = self._save_feature_importance(
            rf,
            "Random Forest",
            "random_forest_feature_importance.csv",
            "random_forest_feature_importance.png",
            self.rf_plot_top
        )

        self.rf_model = rf
        self.rf_pred = y_pred

        rf_model_path = os.path.join(
            self.output_dir_trained_models,
            "random_forest_model.pkl"
        )

        joblib.dump(rf, rf_model_path)

        print(f"Random Forest model saved: {rf_model_path}")

        return result

    # -------------------- GRADIENT BOOSTING: BOOSTING --------------------
    def run_gradient_boosting(self):
        """
        Run Gradient Boosting Regressor.

        Gradient Boosting is a boosting ensemble because trees are trained
        sequentially. Each new tree attempts to correct the errors left by
        the previous trees.
        """

        print("\n" + "=" * 20 + " GRADIENT BOOSTING REGRESSOR (BOOSTING) " + "=" * 20)

        gb = GradientBoostingRegressor(
            n_estimators=self.gb_n_estimators,
            learning_rate=self.gb_learning_rate,
            max_depth=self.gb_max_depth,
            min_samples_leaf=self.gb_min_samples_leaf,
            random_state=self.random_state
        )

        gb.fit(self.X_train, self.y_train)

        y_pred = gb.predict(self.X_test)

        result = self._evaluate_regression(
            "Gradient Boosting Regressor (Boosting)",
            y_pred
        )

        self._plot_actual_vs_predicted(
            y_pred,
            "Gradient Boosting Regressor",
            "gradient_boosting_actual_vs_predicted.png"
        )

        self._plot_residuals(
            y_pred,
            "Gradient Boosting Regressor",
            "gradient_boosting_residuals.png"
        )

        self.gb_feature_importance = self._save_feature_importance(
            gb,
            "Gradient Boosting",
            "gradient_boosting_feature_importance.csv",
            "gradient_boosting_feature_importance.png",
            self.gb_plot_top
        )

        self.gb_model = gb
        self.gb_pred = y_pred

        gb_model_path = os.path.join(
            self.output_dir_trained_models,
            "gradient_boosting_model.pkl"
        )

        joblib.dump(gb, gb_model_path)

        print(f"Gradient Boosting model saved: {gb_model_path}")

        return result

    # -------------------- SAVE RESULTS --------------------
    def save_results_table(self):
        """Save ensemble model comparison table."""

        results_df = pd.DataFrame(self.results).sort_values(by="MAE")

        save_path = os.path.join(
            self.output_dir_results,
            "ensemble_learning_results.csv"
        )

        results_df.to_csv(save_path, index=False)

        print("\n" + "=" * 20 + " ENSEMBLE MODEL COMPARISON " + "=" * 20)
        print(results_df)
        print(f"Results table saved: {save_path}")

        return results_df

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run all ensemble learning models."""

        print("\n" + "=" * 20 + " ENSEMBLE LEARNING " + "=" * 20)

        self.prepare_data()

        self.run_random_forest()
        self.run_gradient_boosting()

        results_df = self.save_results_table()

        print("Ensemble learning complete.")
        print("Metric outputs saved to:", self.output_dir_results)
        print("Graphics saved to:", self.output_dir_graphics)
        print("Trained models saved to:", self.output_dir_trained_models)

        return results_df
