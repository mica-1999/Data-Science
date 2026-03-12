import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class DimensionalityReducer:
    def __init__(self, df_scaled: pd.DataFrame, df_eda: pd.DataFrame, config: dict):
        self.df_scaled = df_scaled.copy()
        self.df_eda = df_eda.copy()
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.features = config['dim_reduction']['features_for_dr']
        self.n_components = config['dim_reduction'].get('n_components', 2)

    def run_pca(self):
        """Run PCA and plot 2D projection."""
        # Select features
        features_for_dr = self.df_scaled[self.features]

        # Fit PCA
        pca = PCA(n_components=self.n_components)
        pca_result = pca.fit_transform(features_for_dr)

        # Create result DataFrame
        df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(self.n_components)])
        df_pca['AIRLINE'] = self.df_eda.loc[features_for_dr.index, 'AIRLINE'].values

        # Scatterplot
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

        # Print explained variance
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        print("Total variance explained:", sum(pca.explained_variance_ratio_))

        return df_pca, pca