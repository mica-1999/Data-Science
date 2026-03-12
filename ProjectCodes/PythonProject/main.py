import yaml
from preprocessing import DataPreprocessor
from eda import EDAAnalyzer
from pca_umap import DimensionalityReducer

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    preprocessor = DataPreprocessor(config)
    preprocessor.preprocess()

    eda = EDAAnalyzer(preprocessor.df_eda, config)
    eda.run_all()

    dim_red = DimensionalityReducer(preprocessor.df,preprocessor.df_eda,config)
    dim_red.run_pca()

if __name__ == "__main__":
    main()