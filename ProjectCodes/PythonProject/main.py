import yaml
from preprocessing import DataPreprocessor

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    preprocessor = DataPreprocessor(config)
    preprocessor.preprocess()

if __name__ == "__main__":
    main()