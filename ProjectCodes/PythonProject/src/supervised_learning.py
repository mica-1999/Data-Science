#%% Phase 4: Model Building / Supervised Learning Models

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SupervisedLearningRunner:
    """
    Runs two supervised learning models from sklearn, excluding the custom kNN:

    1. Decision Tree Regressor
       - Learns rule-based splits from labeled data.
       - Captures non-linear relationships.
       - Easy to interpret, but can overfit if depth is not controlled.

    2. Support Vector Regressor (SVR)
       - Learns a regression function with a margin/tolerance zone.
       - Uses scaled numeric features.
       - Can model non-linear patterns with an RBF kernel.
       - Expensive on large datasets, so a sample is used.

    Target:
        ARR_DELAY, already clipped at 0 according to the project/instructor rule.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

        modeling_cfg = config.get("modeling", {})
        supervised_cfg = config.get("supervised_learning", {})

        self.target_col = modeling_cfg.get("target_col", "ARR_DELAY")
        self.drop_cols = modeling_cfg.get("drop_columns", [self.target_col])
        self.test_size = modeling_cfg.get("test_size", 0.2)
        self.random_state = modeling_cfg.get("random_state", 42)

        self.output_dir_results = config["output_dir_model_results"]
        self.output_dir_graphics = config["output_dir_model_graphics"]
        self.output_dir_trained_models = config["output_dir_trained_models"]

        os.makedirs(self.output_dir_results, exist_ok=True)
        os.makedirs(self.output_dir_graphics, exist_ok=True)
        os.makedirs(self.output_dir_trained_models, exist_ok=True)

        self.dt_max_depth = supervised_cfg.get("decision_tree", {}).get("max_depth", 10)
        self.dt_min_samples_leaf = supervised_cfg.get("decision_tree", {}).get("min_samples_leaf", 50)

        self.svr_sample_size = supervised_cfg.get("svr", {}).get("sample_size", 20000)
        self.svr_kernel = supervised_cfg.get("svr", {}).get("kernel", "rbf")
        self.svr_c = supervised_cfg.get("svr", {}).get("C", 10.0)
        self.svr_epsilon = supervised_cfg.get("svr", {}).get("epsilon", 1.0)

        self.results = []

    # -------------------- PREPARE DATA --------------------
    def prepare_data(self, sample_size=None):
        """
        Prepare X and y for sklearn models.

        Steps:
            - Drop target/leakage columns listed in config.
            - Keep numeric columns only.
            - Fill remaining missing values with median.
            - Split into train and test sets.
        """

        df_model = self.df.copy()

        if sample_size is not None:
            df_model = df_model.sample(
                n=min(sample_size, len(df_model)),
                random_state=self.random_state
            )

        y = df_model[self.target_col]

        X = df_model.drop(columns=self.drop_cols, errors="ignore")
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test

    # -------------------- EVALUATE REGRESSION --------------------
    def evaluate_regression(self, model_name: str, y_test, y_pred):
        """Calculate standard regression metrics."""

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

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
    def plot_predictions(self, y_test, y_pred, model_name: str):
        """Save scatter plot comparing actual and predicted delays."""

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)

        plt.xlabel("Actual ARR_DELAY")
        plt.ylabel("Predicted ARR_DELAY")
        plt.title(f"{model_name}: Actual vs Predicted Arrival Delay")

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(
            self.output_dir_graphics,
            f"{filename}_actual_vs_predicted.png"
        )

        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Actual vs predicted plot saved: {path}")

    # -------------------- PLOT RESIDUALS --------------------
    def plot_residuals(self, y_test, y_pred, model_name: str):
        """Save residual distribution plot."""

        residuals = y_test - y_pred

        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=50)

        plt.xlabel("Residual Error")
        plt.ylabel("Frequency")
        plt.title(f"{model_name}: Residual Distribution")

        filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(
            self.output_dir_graphics,
            f"{filename}_residuals.png"
        )

        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Residual plot saved: {path}")

    # -------------------- DECISION TREE --------------------
    def run_decision_tree(self):
        """
        Run Decision Tree Regressor.

        The model learns if/then style rules from the data and predicts ARR_DELAY
        based on the average delay of samples that fall into each leaf.
        """

        print("\n" + "=" * 20 + " DECISION TREE REGRESSOR " + "=" * 20)

        X_train, X_test, y_train, y_test = self.prepare_data()

        dt = DecisionTreeRegressor(
            max_depth=self.dt_max_depth,
            min_samples_leaf=self.dt_min_samples_leaf,
            random_state=self.random_state
        )

        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        self.evaluate_regression("Decision Tree Regressor", y_test, y_pred)
        self.plot_predictions(y_test, y_pred, "Decision Tree Regressor")
        self.plot_residuals(y_test, y_pred, "Decision Tree Regressor")

        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": dt.feature_importances_
        }).sort_values(by="importance", ascending=False)

        importance_path = os.path.join(
            self.output_dir_results,
            "decision_tree_feature_importance.csv"
        )

        importance_df.to_csv(importance_path, index=False)

        print("\nTop Decision Tree features:")
        print(importance_df.head(10))
        print(f"Feature importance saved: {importance_path}")

        plt.figure(figsize=(24, 10))
        plot_tree(
            dt,
            feature_names=X_train.columns,
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=8
        )

        tree_path = os.path.join(
            self.output_dir_graphics,
            "decision_tree_preview.png"
        )

        plt.savefig(tree_path, bbox_inches="tight")
        plt.close()

        print(f"Decision tree preview saved: {tree_path}")

        self.dt_model = dt
        self.dt_feature_importance = importance_df

        dt_model_path = os.path.join(
            self.output_dir_trained_models,
            "decision_tree_model.pkl"
        )

        joblib.dump(dt, dt_model_path)

        print(f"Decision Tree model saved: {dt_model_path}")

    # -------------------- SVR --------------------
    def run_svr(self):
        """
        Run Support Vector Regression.

        SVR is sensitive to feature scale, so StandardScaler is used inside a Pipeline.
        Since SVR can be slow on very large datasets, a sample is used.
        """

        print("\n" + "=" * 20 + " SUPPORT VECTOR REGRESSOR " + "=" * 20)
        print(f"Using sample size: {self.svr_sample_size}")

        X_train, X_test, y_train, y_test = self.prepare_data(
            sample_size=self.svr_sample_size
        )

        svr = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(
                kernel=self.svr_kernel,
                C=self.svr_c,
                epsilon=self.svr_epsilon
            ))
        ])

        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)

        self.evaluate_regression("Support Vector Regressor", y_test, y_pred)
        self.plot_predictions(y_test, y_pred, "Support Vector Regressor")
        self.plot_residuals(y_test, y_pred, "Support Vector Regressor")

        self.svr_model = svr

        svr_model_path = os.path.join(
            self.output_dir_trained_models,
            "svr_model.pkl"
        )

        joblib.dump(svr, svr_model_path)

        print(f"SVR model saved: {svr_model_path}")

    # -------------------- SAVE RESULTS --------------------
    def save_results(self):
        """Save supervised learning model results in one comparison table."""

        results_df = pd.DataFrame(self.results)

        results_path = os.path.join(
            self.output_dir_results,
            "supervised_learning_results.csv"
        )

        results_df.to_csv(results_path, index=False)

        print("\n" + "=" * 20 + " SUPERVISED MODEL COMPARISON " + "=" * 20)
        print(results_df.sort_values(by="MAE"))
        print(f"Results table saved: {results_path}")

        return results_df

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run both supervised sklearn models."""

        print("\n" + "=" * 20 + " SUPERVISED LEARNING " + "=" * 20)

        self.run_decision_tree()
        self.run_svr()

        results_df = self.save_results()

        print("Supervised learning complete.")
        print("Metric outputs saved to:", self.output_dir_results)
        print("Graphics saved to:", self.output_dir_graphics)
        print("Trained models saved to:", self.output_dir_trained_models)

        return results_df
