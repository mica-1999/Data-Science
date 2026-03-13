import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
import warnings

# Suppress specific UMAP warning
warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state"
)

class DimensionalityReducer:
    def __init__(self, df_scaled: pd.DataFrame, df_eda: pd.DataFrame, config: dict):
        self.df_scaled = df_scaled.copy()
        self.df_eda = df_eda.copy()
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.features = config['dim_reduction']['features_for_dr']
        self.n_components = config['dim_reduction'].get('n_components', 2)

        self.umap_sample_size = config['dim_reduction']['umap_sample_size']
        self.umap_n_neighbors = config['dim_reduction']['umap_n_neighbors']
        self.umap_min_dist = config['dim_reduction']['umap_min_dist']
        self.umap_alpha = config['dim_reduction']['umap_alpha']

    # -------------------- PCA --------------------
    def run_pca(self):
        """Run PCA and plot 2D projection."""
        print("\n" + "=" * 20 + " PCA " + "=" * 20)

        # Select features
        features_for_dr = self.df_scaled[self.features]

        # Fit PCA
        pca = PCA(n_components=self.n_components)
        pca_result = pca.fit_transform(features_for_dr)

        # Create result DataFrame
        df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(self.n_components)])
        df_pca['AIRLINE'] = self.df_eda.loc[features_for_dr.index, 'AIRLINE'].values

        # Scatterplot
        print("Plotting PCA Scatter projection...")
        plt.figure(figsize=(14,6))
        sns.scatterplot(
            data=df_pca,
            x='PC1',
            y='PC2',
            hue='AIRLINE',
            palette='tab20',
            alpha=0.6
        )
        plt.title("PCA Projection of Flights")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.savefig(os.path.join(self.output_dir, "pca_projection.png"), bbox_inches='tight')
        plt.close()

        # Scree
        print("Plotting PCA Scree plot...")
        plt.figure(figsize=(6, 4))
        plt.bar(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
            color='steelblue'
        )
        plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1),
                   [f'PC{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)])
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Scree Plot")
        plt.savefig(os.path.join(self.output_dir, "pca_scree.png"), bbox_inches='tight')
        plt.close()

        # Print explained variance
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        print("Total variance explained:", sum(pca.explained_variance_ratio_))

        return df_pca, pca

    # -------------------- UMAP --------------------
    def run_umap(self):
        """Run UMAP on a sample of the data and plot 2D projection."""
        print("\n" + "=" * 20 + " UMAP " + "=" * 20)

        # Sample features for faster computation
        features_sample = self.df_scaled[self.features].sample(
            min(self.umap_sample_size, len(self.df_scaled)), random_state=42
        )
        airline_sample = self.df_eda.loc[features_sample.index, 'AIRLINE'].values

        # Initialize UMAP
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=42
        )

        # Fit and transform
        umap_result = umap_reducer.fit_transform(features_sample)

        # Create DataFrame for plotting
        df_umap = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
        df_umap['AIRLINE'] = airline_sample

        # Plot UMAP projection
        print("Plotting UMAP projection...")
        plt.figure(figsize=(14, 6))
        sns.scatterplot(
            data=df_umap,
            x='UMAP1',
            y='UMAP2',
            hue='AIRLINE',
            palette='tab20',
            alpha=self.umap_alpha
        )
        plt.title("UMAP Projection of Flights (Sample)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(self.output_dir, "umap_projection.png"), bbox_inches='tight')
        plt.close()

        print(f"UMAP completed. Sample of first 5 rows of embedding:\n{umap_result[:5]}")
        print("🎉 Both Processes Done. All outputs saved to:", self.output_dir)