#%% Phase 4: Model Building / kNN From Scratch

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils import resample

class KNNFromScratch:
    """
    K-Nearest Neighbors algorithm implemented from scratch using NumPy.

    Supports:
        - regression: predicts ARR_DELAY in minutes
        - classification: predicts delay class
            0 = on-time
            1 = short delay
            2 = long delay

    Main idea:
        1. Store training samples and labels.
        2. For each test sample, compute distance to all training samples.
        3. Select the k closest samples.
        4. Regression: average their target values.
        5. Classification: use majority vote.
    """

    def __init__(self, k: int = 5, mode: str = "regression"):
        self.k = k
        self.mode = mode
        self.X_train = None
        self.y_train = None

    # -------------------- FIT --------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Store training data."""
        self.X_train = X_train
        self.y_train = y_train

    # -------------------- EUCLIDEAN DISTANCE --------------------
    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance from one test sample to all training samples."""
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    # -------------------- PREDICT SINGLE SAMPLE --------------------
    def _predict_single(self, x: np.ndarray):
        """Predict one sample using k nearest neighbors."""

        distances = self._compute_distances(x)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_targets = self.y_train[k_nearest_indices]

        if self.mode == "regression":
            return np.mean(k_nearest_targets)

        return np.argmax(np.bincount(k_nearest_targets.astype(int)))

    # -------------------- PREDICT --------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict all samples in the test set."""
        return np.array([self._predict_single(x) for x in X_test])


class KNNRunner:
    """
    Runs kNN regression and classification experiments.

    This class:
        - prepares the data
        - samples the dataset for runtime feasibility
        - evaluates multiple k values
        - saves metrics to CSV
        - saves plots to the graphics folder
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

        self.target_col = config["modeling"]["target_col"]
        self.test_size = config["modeling"]["test_size"]
        self.random_state = config["modeling"]["random_state"]
        self.drop_cols = config["modeling"]["drop_columns"]

        self.k_values = config["knn"]["k_values"]
        self.sample_size = config["knn"]["sample_size"]

        self.output_dir_results = config["output_dir_model_results"]
        self.output_dir_graphics = config["output_dir_model_graphics"]

        os.makedirs(self.output_dir_results, exist_ok=True)
        os.makedirs(self.output_dir_graphics, exist_ok=True)

    # -------------------- PREPARE DATA --------------------
    def _prepare_data(self, mode: str):
        """
        Prepare X and y for kNN.

        Regression:
            y = ARR_DELAY

        Classification:
            0 = On-time      ARR_DELAY < 15
            1 = Short delay  15 <= ARR_DELAY <= 30
            2 = Long delay   ARR_DELAY > 30
        """

        df_sample = self.df.sample(
            n=min(self.sample_size, len(self.df)),
            random_state=self.random_state
        )

        X = df_sample.drop(columns=self.drop_cols, errors="ignore")
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())

        if mode == "regression":
            y = df_sample[self.target_col].values
            X_np = X.values.astype(np.float64)
            y_np = y.astype(np.float64)

        else:
            delay = df_sample[self.target_col]
            y = np.where(delay < 15, 0, np.where(delay <= 30, 1, 2))

            # Balance classes by undersampling majority class
            df_temp = X.copy()
            df_temp["__target__"] = y

            on_time = df_temp[df_temp["__target__"] == 0]
            short = df_temp[df_temp["__target__"] == 1]
            long_ = df_temp[df_temp["__target__"] == 2]

            min_size = min(len(on_time), len(short), len(long_))

            on_time_bal = resample(on_time, n_samples=min_size, random_state=self.random_state)
            short_bal = resample(short, n_samples=min_size, random_state=self.random_state)
            long_bal = resample(long_, n_samples=min_size, random_state=self.random_state)

            df_balanced = pd.concat([on_time_bal, short_bal, long_bal])
            df_balanced = df_balanced.sample(frac=1, random_state=self.random_state)

            X_np = df_balanced.drop(columns=["__target__"]).values.astype(np.float64)
            y_np = df_balanced["__target__"].values.astype(np.float64)

            print(f"Balanced class sizes: {min_size} per class | Total: {len(y_np)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_np,
            y_np,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test

    # -------------------- EVALUATE REGRESSION --------------------
    def _evaluate_regression(self, y_test: np.ndarray, y_pred: np.ndarray, k: int):
        """Calculate regression metrics."""

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"  k={k} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

        return {
            "Model": "kNN Regression",
            "k": k,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

    # -------------------- EVALUATE CLASSIFICATION --------------------
    def _evaluate_classification(self, y_test: np.ndarray, y_pred: np.ndarray, k: int):
        """Calculate classification metrics."""

        acc = accuracy_score(y_test, y_pred)

        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=["On-time", "Short Delay", "Long Delay"],
            zero_division=0,
            output_dict=True
        )

        print(f"  k={k} | Accuracy: {acc:.3f}")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["On-time", "Short Delay", "Long Delay"],
                zero_division=0
            )
        )

        return {
            "Model": "kNN Classification",
            "k": k,
            "Accuracy": acc,
            "Macro Precision": report_dict["macro avg"]["precision"],
            "Macro Recall": report_dict["macro avg"]["recall"],
            "Macro F1": report_dict["macro avg"]["f1-score"],
            "Weighted Precision": report_dict["weighted avg"]["precision"],
            "Weighted Recall": report_dict["weighted avg"]["recall"],
            "Weighted F1": report_dict["weighted avg"]["f1-score"]
        }, report_dict

    # -------------------- PLOT K VS METRIC --------------------
    def _plot_k_vs_metric(self, k_values, metric_values, metric_name: str, mode: str):
        """Plot model performance across k values."""

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, metric_values, marker="o")
        plt.title(f"kNN {mode.capitalize()} - k vs {metric_name}")
        plt.xlabel("k")
        plt.ylabel(metric_name)
        plt.xticks(k_values)
        plt.grid(True)

        filename = f"knn_{mode}_k_vs_{metric_name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(self.output_dir_graphics, filename)

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        print(f"  Plot saved: {plot_path}")

    # -------------------- PLOT CONFUSION MATRIX --------------------
    def _plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray, k: int):
        """Plot confusion matrix for best k classification model."""

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["On-time", "Short Delay", "Long Delay"],
            yticklabels=["On-time", "Short Delay", "Long Delay"]
        )

        plt.title(f"kNN Classification Confusion Matrix (k={k})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plot_path = os.path.join(
            self.output_dir_graphics,
            f"knn_confusion_matrix_k{k}.png"
        )

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        print(f"  Confusion matrix saved: {plot_path}")

    # -------------------- SAVE RESULTS --------------------
    def _save_results(self, results: list, filename: str):
        """Save list of metric dictionaries to CSV."""

        results_path = os.path.join(self.output_dir_results, filename)
        pd.DataFrame(results).to_csv(results_path, index=False)

        print(f"Results saved: {results_path}")

    # -------------------- RUN REGRESSION --------------------
    def run_regression(self):
        """Run kNN regression for every k."""

        print("\n" + "=" * 20 + " kNN REGRESSION " + "=" * 20)
        print(f"Sample size: {self.sample_size} | k values: {self.k_values}")

        X_train, X_test, y_train, y_test = self._prepare_data(mode="regression")

        print(f"Train: {X_train.shape} | Test: {X_test.shape}")

        regression_results = []
        mae_list = []

        for k in self.k_values:
            print(f"\nFitting kNN Regressor with k={k}...")

            knn = KNNFromScratch(k=k, mode="regression")
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            result = self._evaluate_regression(y_test, y_pred, k)

            regression_results.append(result)
            mae_list.append(result["MAE"])

        self._plot_k_vs_metric(self.k_values, mae_list, "MAE", "regression")

        best_result = min(regression_results, key=lambda x: x["MAE"])
        print(f"\nBest k for regression: k={best_result['k']}")

        self._save_results(
            regression_results,
            "knn_regression_results.csv"
        )

        return regression_results

    # -------------------- RUN CLASSIFICATION --------------------
    def run_classification(self):
        """Run kNN classification for every k."""

        print("\n" + "=" * 20 + " kNN CLASSIFICATION " + "=" * 20)
        print(f"Sample size: {self.sample_size} | k values: {self.k_values}")

        X_train, X_test, y_train, y_test = self._prepare_data(mode="classification")

        print(f"Train: {X_train.shape} | Test: {X_test.shape}")

        classification_results = []
        accuracy_list = []

        best_k = None
        best_acc = -1
        best_pred = None
        best_report = None

        for k in self.k_values:
            print(f"\nFitting kNN Classifier with k={k}...")

            knn = KNNFromScratch(k=k, mode="classification")
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            result, report_dict = self._evaluate_classification(y_test, y_pred, k)

            classification_results.append(result)
            accuracy_list.append(result["Accuracy"])

            if result["Accuracy"] > best_acc:
                best_acc = result["Accuracy"]
                best_k = k
                best_pred = y_pred
                best_report = report_dict

        self._plot_k_vs_metric(
            self.k_values,
            accuracy_list,
            "Accuracy",
            "classification"
        )

        self._plot_confusion_matrix(y_test, best_pred, best_k)

        print(f"\nBest k for classification: k={best_k}")

        self._save_results(
            classification_results,
            "knn_classification_results.csv"
        )

        best_report_path = os.path.join(
            self.output_dir_results,
            "knn_best_classification_report.csv"
        )

        pd.DataFrame(best_report).transpose().to_csv(best_report_path)

        print(f"Best classification report saved: {best_report_path}")

        return classification_results

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run both kNN regression and classification."""

        regression_results = self.run_regression()
        classification_results = self.run_classification()

        print("\nkNN complete.")
        print("Metric outputs saved to:", self.output_dir_results)
        print("Graphics saved to:", self.output_dir_graphics)

        return regression_results, classification_results
