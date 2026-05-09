#%% Phase 4: Model Building / Deep Learning Model using PyTorch

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class FlightDelayMLP(nn.Module):
    """
    Feed-forward neural network for flight delay regression.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    # -------------------- FORWARD PASS --------------------
    def forward(self, x):
        """Forward pass through the neural network."""
        return self.network(x)


class DeepLearningRunner:
    """
    Runs a PyTorch feed-forward neural network for ARR_DELAY regression.

    Model:
        Feed-Forward Neural Network / Multilayer Perceptron (MLP)

    This architecture was selected because the dataset is tabular rather than
    image-based or sequential.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

        self.target_col = config["modeling"]["target_col"]
        self.drop_cols = config["modeling"]["drop_columns"]
        self.test_size = config["modeling"]["test_size"]
        self.random_state = config["modeling"]["random_state"]

        dl_cfg = config.get("deep_learning", {})

        self.sample_size = dl_cfg.get("sample_size", 200000)
        self.epochs = dl_cfg.get("epochs", 50)
        self.batch_size = dl_cfg.get("batch_size", 512)
        self.learning_rate = dl_cfg.get("learning_rate", 0.001)
        self.patience = dl_cfg.get("patience", 5)

        self.output_dir_results = config["output_dir_model_results"]
        self.output_dir_graphics = config["output_dir_model_graphics"]
        self.output_dir_trained_models = config["output_dir_trained_models"]

        os.makedirs(self.output_dir_results, exist_ok=True)
        os.makedirs(self.output_dir_graphics, exist_ok=True)
        os.makedirs(self.output_dir_trained_models, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- PREPARE DATA --------------------
    def prepare_data(self):
        """Prepare numeric features, scale them, and create PyTorch DataLoaders."""

        df_sample = self.df.sample(
            n=min(self.sample_size, len(self.df)),
            random_state=self.random_state
        )

        X = df_sample.drop(columns=self.drop_cols, errors="ignore")
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())

        y = df_sample[self.target_col].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.X_test_tensor = X_test_tensor.to(self.device)
        self.y_test_tensor = y_test_tensor.to(self.device)

        self.y_test = y_test.flatten()
        self.input_dim = X_train.shape[1]

        print(f"Using device: {self.device}")
        print(f"Sample size: {len(df_sample)}")
        print(f"Train batches: {len(self.train_loader)} | Test shape: {X_test.shape}")

    # -------------------- TRAIN MODEL --------------------
    def train_model(self):
        """Train the PyTorch MLP using MSE loss and early stopping."""

        self.model = FlightDelayMLP(self.input_dim).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        best_loss = np.inf
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_losses = []

            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))

            self.model.eval()

            with torch.no_grad():
                val_predictions = self.model(self.X_test_tensor)
                val_loss = criterion(
                    val_predictions,
                    self.y_test_tensor
                ).item()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_loss:.3f} | "
                f"Validation MSE: {val_loss:.3f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.best_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        self.model.load_state_dict(self.best_state)

    # -------------------- EVALUATE MODEL --------------------
    def evaluate_model(self):
        """Evaluate model using MAE, RMSE, and R2."""

        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(self.X_test_tensor).cpu().numpy().flatten()

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        print("\nDeep Learning MLP results:")
        print(f"MAE : {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2  : {r2:.3f}")

        self.results = {
            "Model": "PyTorch Feed-Forward Neural Network",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        self.y_pred = y_pred

        return self.results

    # -------------------- PLOT TRAINING HISTORY --------------------
    def plot_training_history(self):
        """Plot train and validation loss over epochs."""

        plt.figure(figsize=(8, 5))

        plt.plot(self.train_losses, label="Training MSE")
        plt.plot(self.val_losses, label="Validation MSE")

        plt.title("Deep Learning Training History")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)

        path = os.path.join(
            self.output_dir_graphics,
            "deep_learning_training_history.png"
        )

        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Training history plot saved: {path}")

    # -------------------- PLOT ACTUAL VS PREDICTED --------------------
    def plot_actual_vs_predicted(self):
        """Scatter plot of actual vs predicted ARR_DELAY."""

        plt.figure(figsize=(7, 6))

        plt.scatter(self.y_test, self.y_pred, alpha=0.25)

        plt.xlabel("Actual ARR_DELAY")
        plt.ylabel("Predicted ARR_DELAY")
        plt.title("Deep Learning: Actual vs Predicted")
        plt.grid(True)

        path = os.path.join(
            self.output_dir_graphics,
            "deep_learning_actual_vs_predicted.png"
        )

        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Actual vs predicted plot saved: {path}")

    # -------------------- PLOT RESIDUALS --------------------
    def plot_residuals(self):
        """Histogram of prediction residuals."""

        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=50)

        plt.title("Deep Learning Residual Distribution")
        plt.xlabel("Residual: Actual - Predicted")
        plt.ylabel("Frequency")
        plt.grid(True)

        path = os.path.join(
            self.output_dir_graphics,
            "deep_learning_residuals.png"
        )

        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"Residual plot saved: {path}")

    # -------------------- SAVE RESULTS --------------------
    def save_results(self):
        """Save metrics to CSV and trained model to .pt file."""

        results_df = pd.DataFrame([self.results])

        results_path = os.path.join(
            self.output_dir_results,
            "deep_learning_results.csv"
        )

        results_df.to_csv(results_path, index=False)

        print(f"Results table saved: {results_path}")

        model_path = os.path.join(
            self.output_dir_trained_models,
            "deep_learning_model.pt"
        )

        torch.save(self.model.state_dict(), model_path)

        print(f"Trained PyTorch model saved: {model_path}")

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run the complete deep learning pipeline."""

        print("\n" + "=" * 20 + " DEEP LEARNING - PYTORCH MLP " + "=" * 20)

        self.prepare_data()
        self.train_model()
        self.evaluate_model()

        self.plot_training_history()
        self.plot_actual_vs_predicted()
        self.plot_residuals()

        self.save_results()

        print("Deep learning complete.")
        print("Metric outputs saved to:", self.output_dir_results)
        print("Graphics saved to:", self.output_dir_graphics)
        print("Trained model saved to:", self.output_dir_trained_models)

        return self.results
