import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class KNNFromScratch:
    """
    K-Nearest Neighbors algorithm implemented from scratch using only NumPy.

    Supports two modes:
        - 'regression'     : predicts arrival delay in minutes (continuous output)
        - 'classification' : predicts delay class (0=on-time, 1=short, 2=long)

    How it works:
        1. Store all training samples and their labels
        2. For each test sample, compute the Euclidean distance to every training sample
        3. Find the k training samples with the smallest distances (nearest neighbors)
        4. Regression   -> return the average of the k neighbors' target values
           Classification -> return the most common class among the k neighbors (majority vote)
    """

    def __init__(self, k: int = 5, mode: str = 'regression'):
        self.k = k
        self.mode = mode
        self.X_train = None
        self.y_train = None

    # -------------------- FIT --------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """ Stores the training data in memory. """
        self.X_train = X_train
        self.y_train = y_train

    # -------------------- EUCLIDEAN DISTANCE --------------------
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
            Computes the Euclidean distance between two flight feature vectors.
            Smaller distance = more similar flights.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # -------------------- PREDICT SINGLE SAMPLE --------------------
    def _predict_single(self, x: np.ndarray):
        """
        Predicts the output for a single flight by finding the k most similar
        training flights and averaging their delays (regression) or
        taking the majority class (classification).
        """

        # Step 1: compute distance from x to every training sample
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])

        # Step 2: get indices of the k nearest neighbors (sorted ascending by distance)
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Step 3: retrieve the target values of those k neighbors
        k_nearest_targets = self.y_train[k_nearest_indices]

        # Step 4: aggregate
        if self.mode == 'regression':
            # Average the delay values of the k nearest flights
            return np.mean(k_nearest_targets)
        else:
            # Return the most common class among the k nearest flights
            # np.bincount counts occurrences of each integer class
            return np.argmax(np.bincount(k_nearest_targets.astype(int)))

    # -------------------- PREDICT --------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """ Runs _predict_single for every flight in the test set."""
        return np.array([self._predict_single(x) for x in X_test])


class KNNRunner:
    """
    Handles data preparation, training, evaluation, and plotting
    for the KNNFromScratch model on the flight delay dataset.

    Runs both regression and classification tasks and saves all outputs.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

        # Read config values
        self.target_col = config['modeling']['target_col']           # 'ARR_DELAY'
        self.test_size = config['modeling']['test_size']             # 0.2
        self.random_state = config['modeling']['random_state']       # 42
        self.drop_cols = config['modeling']['drop_columns']          # columns to exclude from features
        self.k_values = config['knn']['k_values']                    # list of k values to try e.g. [3,5,7,9,11] (updated)
        self.sample_size = config['knn']['sample_size']              # rows to use (kNN is slow on millions of rows)
        self.output_dir = config['output_dir_knn']               # where to save plots
        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------- PREPARE DATA --------------------
    def _prepare_data(self, mode: str):
        """
        Prepares features and target for kNN.
        Regression: y = ARR_DELAY in minutes.
        Classification: y = 0 (on-time <15min), 1 (short 15-30min), 2 (long >30min).
        Dataset is sampled down to self.sample_size rows since kNN is slow on large data.
        """

        # Sample to keep runtime manageable
        df_sample = self.df.sample(n=min(self.sample_size, len(self.df)), random_state=self.random_state)

        # Build feature matrix — drop target and leakage columns
        X = df_sample.drop(columns=self.drop_cols, errors='ignore')

        # Keep only numeric columns (kNN needs numbers for distance)
        X = X.select_dtypes(include=[np.number])

        # Fill any remaining NaNs with column median
        X = X.fillna(X.median())

        if mode == 'regression':
            y = df_sample[self.target_col].values

        else:
            # Convert continuous ARR_DELAY into 3 classes
            delay = df_sample[self.target_col]
            y = np.where(delay < 15, 0, np.where(delay <= 30, 1, 2))

        # Convert to numpy arrays
        X_np = X.values.astype(np.float64)
        y_np = y.astype(np.float64)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test

    # -------------------- EVALUATE REGRESSION --------------------
    def _evaluate_regression(self, y_test: np.ndarray, y_pred: np.ndarray, k: int):
        """
        Print MAE, RMSE, and R2 for a regression prediction.

        MAE  : average absolute error in minutes
        RMSE : penalizes large errors more heavily
        R2   : proportion of variance explained (1.0 = perfect)
        """
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"  k={k} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
        return mae, rmse, r2

    # -------------------- EVALUATE CLASSIFICATION --------------------
    def _evaluate_classification(self, y_test: np.ndarray, y_pred: np.ndarray, k: int):
        """
        Print accuracy and full classification report for a classification prediction.

        Accuracy : fraction of correctly predicted delay classes
        Report   : precision, recall, F1 per class
        """
        acc = accuracy_score(y_test, y_pred)
        print(f"  k={k} | Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred,
                                    target_names=['On-time', 'Short Delay', 'Long Delay'],
                                    zero_division=0))
        return acc

    # -------------------- PLOT K vs METRIC --------------------
    def _plot_k_vs_metric(self, k_values, metric_values, metric_name: str, mode: str):
        """
        Plot how performance changes as k increases.
        Helps identify the best k value visually. We can change on YAML to visualize better results
        """
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, metric_values, marker='o', color='steelblue')
        plt.title(f"kNN {mode.capitalize()} — k vs {metric_name}")
        plt.xlabel("k (Number of Neighbors)")
        plt.ylabel(metric_name)
        plt.xticks(k_values)
        plt.grid(True)
        filename = f"knn_{mode}_k_vs_{metric_name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {filename}")

    # -------------------- PLOT CONFUSION MATRIX --------------------
    def _plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray, k: int):
        """
        Plot a heatmap confusion matrix for the best classification k.
        Shows how many flights were correctly/incorrectly classified per class.
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['On-time', 'Short Delay', 'Long Delay'],
                    yticklabels=['On-time', 'Short Delay', 'Long Delay'])
        plt.title(f"kNN Classification Confusion Matrix (k={k})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(self.output_dir, f"knn_confusion_matrix_k{k}.png"), bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix saved for k={k}")

    # -------------------- RUN REGRESSION --------------------
    def run_regression(self):
        """
        Run kNN regression for each k in k_values.
        Reports MAE, RMSE, R2 for each k and plots k vs MAE.
        """
        print("\n" + "=" * 20 + " kNN REGRESSION " + "=" * 20)
        print(f"Sample size: {self.sample_size} | k values: {self.k_values}")

        X_train, X_test, y_train, y_test = self._prepare_data(mode='regression')
        print(f"Train: {X_train.shape} | Test: {X_test.shape}")

        mae_list = []

        for k in self.k_values:
            print(f"\nFitting kNN Regressor with k={k}...")
            knn = KNNFromScratch(k=k, mode='regression')
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            mae, rmse, r2 = self._evaluate_regression(y_test, y_pred, k)
            mae_list.append(mae)

        # Plot k vs MAE to visualize best k
        self._plot_k_vs_metric(self.k_values, mae_list, 'MAE', 'regression')

        # Identify and report best k
        best_k = self.k_values[np.argmin(mae_list)]
        print(f"\nBest k for regression (lowest MAE): k={best_k}")

    # -------------------- RUN CLASSIFICATION --------------------
    def run_classification(self):
        """
        Run kNN classification for each k in k_values.
        Reports accuracy and classification report for each k.
        Plots confusion matrix for the best k.
        """
        print("\n" + "=" * 20 + " kNN CLASSIFICATION " + "=" * 20)
        print(f"Sample size: {self.sample_size} | k values: {self.k_values}")

        X_train, X_test, y_train, y_test = self._prepare_data(mode='classification')
        print(f"Train: {X_train.shape} | Test: {X_test.shape}")

        acc_list = []
        best_k = None
        best_acc = 0
        best_pred = None

        for k in self.k_values:
            print(f"\nFitting kNN Classifier with k={k}...")
            knn = KNNFromScratch(k=k, mode='classification')
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = self._evaluate_classification(y_test, y_pred, k)
            acc_list.append(acc)

            # Track best k
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_pred = y_pred

        # Plot k vs accuracy
        self._plot_k_vs_metric(self.k_values, acc_list, 'Accuracy', 'classification')

        # Confusion matrix for best k
        self._plot_confusion_matrix(y_test, best_pred, best_k)
        print(f"\nBest k for classification (highest accuracy): k={best_k}")

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run both regression and classification kNN experiments."""
        self.run_regression()
        self.run_classification()
        print("\nkNN complete. All outputs saved to:", self.output_dir)