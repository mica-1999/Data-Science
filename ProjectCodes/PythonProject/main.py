import yaml
from preprocessing import DataPreprocessor
from eda import EDAAnalyzer
from pca_umap import DimensionalityReducer
from hypothesis import HypothesisTester
from feature_engineering import FeatureEngineer

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    # Preprocessing Step
    preprocessor = DataPreprocessor(config)
    preprocessor.preprocess()

    # EDA Step
    #eda = EDAAnalyzer(preprocessor.df_eda, config)
    #eda.run_all()

    # PCA/UMAP Step
    #dim_red = DimensionalityReducer(preprocessor.df,preprocessor.df_eda,config)
    #dim_red.run_pca()
    #dim_red.run_umap()

    # Hypothesis Testing
    hyp = HypothesisTester(preprocessor.df_eda, config)
    hyp.run_all()

    # Feature Engineering
    fe = FeatureEngineer(preprocessor.df, config)
    fe.run_all()

if __name__ == "__main__":
    main()