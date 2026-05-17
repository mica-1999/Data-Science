#%% Phase 4: Model Building / Clustering Analysis

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class ClusteringRunner:
    """
    Clustering analysis for identifying operational flight delay patterns.

    Algorithms:
        - KMeans: centroid-based clustering
        - DBSCAN: density-based clustering

    This satisfies the project requirement of applying at least two clustering
    algorithms and varying the number of clusters.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

        clustering_cfg = config["clustering"]

        self.sample_size = clustering_cfg["sample_size"]
        self.kmeans_clusters = clustering_cfg["kmeans_clusters"]

        self.dbscan_eps = clustering_cfg["dbscan_eps"]
        self.dbscan_min_samples = clustering_cfg["dbscan_min_samples"]

        self.random_state = clustering_cfg["random_state"]

        self.output_dir_results = config["output_dir_model_results"]
        self.output_dir_graphics = config["output_dir_model_graphics"]

        os.makedirs(self.output_dir_results, exist_ok=True)
        os.makedirs(self.output_dir_graphics, exist_ok=True)

        self.kmeans_results = []
        self.dbscan_results = []

    # -------------------- PREPARE DATA --------------------
    def prepare_data(self):
        """
        Prepare clustering dataset.

        Unlike predictive modeling, ARR_DELAY can be used here because
        clustering is unsupervised and the goal is pattern discovery, not
        leakage-free prediction.
        """

        clustering_features = [
            "ARR_DELAY",
            "DISTANCE",
            "CRS_ELAPSED_TIME",
            "CRS_DEP_TIME_std",
            "MONTH",
            "DAY_OF_WEEK",
            "dep_hour_x_elapsed"
        ]

        available_features = [
            col for col in clustering_features
            if col in self.df.columns
        ]

        data = self.df[available_features].dropna()

        if len(data) > self.sample_size:
            data = data.sample(
                n=self.sample_size,
                random_state=self.random_state
            )

        self.feature_names = available_features
        self.data_used = data.copy()

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(data)

        print(f"Features used for clustering: {self.feature_names}")
        print(f"Clustering dataset shape: {self.X_scaled.shape}")

        return self.X_scaled

    # -------------------- KMEANS --------------------
    def run_kmeans(self):
        """
        Run KMeans clustering with multiple values of k.

        The elbow method and silhouette score are used to assess whether
        meaningful cluster structure exists.
        """

        print("\n" + "=" * 20 + " KMEANS CLUSTERING " + "=" * 20)

        inertias = []
        silhouette_scores = []

        cluster_range = range(2, self.kmeans_clusters + 1)

        for k in cluster_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            labels = kmeans.fit_predict(self.X_scaled)

            inertia = kmeans.inertia_
            sil_score = silhouette_score(self.X_scaled, labels)

            inertias.append(inertia)
            silhouette_scores.append(sil_score)

            self.kmeans_results.append({
                "Algorithm": "KMeans",
                "k": k,
                "Inertia": inertia,
                "Silhouette Score": sil_score
            })

            print(f"k={k} | Inertia: {inertia:.3f} | Silhouette Score: {sil_score:.4f}")

        self._save_kmeans_results()

        self._plot_kmeans_elbow(cluster_range, inertias)
        self._plot_kmeans_silhouette(cluster_range, silhouette_scores)

        best_result = max(
            self.kmeans_results,
            key=lambda row: row["Silhouette Score"]
        )

        optimal_k = int(best_result["k"])

        print(f"Selected KMeans k based on silhouette score: {optimal_k}")

        self._run_final_kmeans(optimal_k)

    # -------------------- SAVE KMEANS RESULTS --------------------
    def _save_kmeans_results(self):
        """Save KMeans evaluation results to CSV."""

        results_path = os.path.join(
            self.output_dir_results,
            "clustering_kmeans_results.csv"
        )

        pd.DataFrame(self.kmeans_results).to_csv(results_path, index=False)

        print(f"KMeans results saved: {results_path}")

    # -------------------- PLOT KMEANS ELBOW --------------------
    def _plot_kmeans_elbow(self, cluster_range, inertias):
        """Plot KMeans elbow method."""

        plt.figure(figsize=(8, 5))
        plt.plot(list(cluster_range), inertias, marker="o")

        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("KMeans Elbow Method")
        plt.grid(True)

        elbow_path = os.path.join(
            self.output_dir_graphics,
            "kmeans_elbow_method.png"
        )

        plt.savefig(elbow_path, bbox_inches="tight")
        plt.close()

        print(f"Elbow plot saved: {elbow_path}")

    # -------------------- PLOT KMEANS SILHOUETTE --------------------
    def _plot_kmeans_silhouette(self, cluster_range, silhouette_scores):
        """Plot KMeans silhouette scores."""

        plt.figure(figsize=(8, 5))
        plt.plot(list(cluster_range), silhouette_scores, marker="o")

        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("KMeans Silhouette Scores")
        plt.grid(True)

        silhouette_path = os.path.join(
            self.output_dir_graphics,
            "kmeans_silhouette_scores.png"
        )

        plt.savefig(silhouette_path, bbox_inches="tight")
        plt.close()

        print(f"Silhouette plot saved: {silhouette_path}")

    # -------------------- FINAL KMEANS --------------------
    def _run_final_kmeans(self, optimal_k: int):
        """Run final KMeans model and save PCA visualization."""

        kmeans_final = KMeans(
            n_clusters=optimal_k,
            random_state=self.random_state,
            n_init=10
        )

        self.kmeans_labels = kmeans_final.fit_predict(self.X_scaled)

        cluster_counts = pd.Series(self.kmeans_labels).value_counts().sort_index()

        summary_df = pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Count": cluster_counts.values
        })

        summary_path = os.path.join(
            self.output_dir_results,
            "clustering_kmeans_cluster_counts.csv"
        )

        summary_df.to_csv(summary_path, index=False)

        print(f"KMeans cluster counts saved: {summary_path}")
        print("\nKMeans cluster counts:")
        print(summary_df)

        self._plot_pca_clusters(
            labels=self.kmeans_labels,
            title=f"KMeans Clustering PCA Projection (k={optimal_k})",
            filename="kmeans_clusters_pca.png"
        )

    # -------------------- DBSCAN --------------------
    def run_dbscan(self):
        """
        Run DBSCAN clustering.

        DBSCAN identifies dense regions as clusters and labels isolated points
        as noise/anomalies using label -1.
        """

        print("\n" + "=" * 20 + " DBSCAN CLUSTERING " + "=" * 20)

        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        )

        self.dbscan_labels = dbscan.fit_predict(self.X_scaled)

        unique_clusters = set(self.dbscan_labels)
        n_clusters = len(unique_clusters - {-1})
        n_noise = list(self.dbscan_labels).count(-1)

        total_points = len(self.dbscan_labels)
        noise_ratio = n_noise / total_points

        print(f"Clusters found: {n_clusters}")
        print(f"Noise points: {n_noise}")
        print(f"Noise ratio: {noise_ratio:.4f}")

        dbscan_summary = {
            "Algorithm": "DBSCAN",
            "eps": self.dbscan_eps,
            "min_samples": self.dbscan_min_samples,
            "Clusters Found": n_clusters,
            "Noise Points": n_noise,
            "Noise Ratio": noise_ratio
        }

        self.dbscan_results.append(dbscan_summary)

        results_path = os.path.join(
            self.output_dir_results,
            "clustering_dbscan_results.csv"
        )

        pd.DataFrame(self.dbscan_results).to_csv(results_path, index=False)

        print(f"DBSCAN results saved: {results_path}")

        cluster_counts = pd.Series(self.dbscan_labels).value_counts().sort_index()

        counts_df = pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Count": cluster_counts.values
        })

        counts_path = os.path.join(
            self.output_dir_results,
            "clustering_dbscan_cluster_counts.csv"
        )

        counts_df.to_csv(counts_path, index=False)

        print(f"DBSCAN cluster counts saved: {counts_path}")
        print("\nDBSCAN cluster counts:")
        print(counts_df)

        self._plot_pca_clusters(
            labels=self.dbscan_labels,
            title="DBSCAN Clustering PCA Projection",
            filename="dbscan_clusters_pca.png"
        )

    def prepare_airline_profiles(self):
        profiles = self.df.groupby("AIRLINE_CODE").agg(
            avg_delay=("ARR_DELAY", "mean"),
            delay_std=("ARR_DELAY", "std"),
            avg_distance=("DISTANCE", "mean"),
            on_time_rate=("ARR_DELAY", lambda x: (x < 15).mean()),
            long_delay_rate=("ARR_DELAY", lambda x: (x > 30).mean()),
            avg_dep_hour=("DEP_HOUR", "mean"),
        ).dropna().reset_index()

        self.airline_names = profiles["AIRLINE_CODE"].values

        scaler = StandardScaler()
        self.X_airline = scaler.fit_transform(
            profiles.drop(columns=["AIRLINE_CODE"])
        )

        print(f"Airline profiles shape: {self.X_airline.shape}")
        return self.X_airline

    def run_airline_clustering(self):
        print("\n" + "=" * 20 + " AIRLINE CLUSTERING " + "=" * 20)

        self.prepare_airline_profiles()

        best_k, best_score = 2, -1

        for k in range(2, min(6, len(self.airline_names))):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(self.X_airline)
            score = silhouette_score(self.X_airline, labels)

            print(f"k={k} | Silhouette: {score:.4f}")

            if score > best_score:
                best_score, best_k = score, k

        km_final = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
        labels = km_final.fit_predict(self.X_airline)

        results = pd.DataFrame({
            "AIRLINE_CODE": self.airline_names,
            "Cluster": labels
        })

        path = os.path.join(self.output_dir_results, "airline_clustering_results.csv")
        results.to_csv(path, index=False)

        print(results.sort_values("Cluster"))
        print(f"Airline clustering results saved: {path}")

    # -------------------- PCA CLUSTER PLOT --------------------
    def _plot_pca_clusters(self, labels, title: str, filename: str):
        """Use PCA to visualize clustering labels in two dimensions."""

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            alpha=0.6
        )

        plt.colorbar(scatter)

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(title)

        path = os.path.join(self.output_dir_graphics, filename)

        plt.savefig(path, bbox_inches="tight")
        plt.close()

        print(f"PCA cluster visualization saved: {path}")

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run all clustering algorithms."""

        print("\n" + "=" * 20 + " CLUSTERING ANALYSIS " + "=" * 20)

        self.prepare_data()

        self.run_kmeans()
        self.run_dbscan()
        self.run_airline_clustering()

        print("Clustering complete.")
        print("Metric outputs saved to:", self.output_dir_results)
        print("Graphics saved to:", self.output_dir_graphics)
