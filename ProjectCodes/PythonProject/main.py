import yaml
from ProjectCodes.PythonProject.src.preprocessing import DataPreprocessor
from ProjectCodes.PythonProject.src.eda import EDAAnalyzer
from ProjectCodes.PythonProject.src.pca_umap import DimensionalityReducer
from ProjectCodes.PythonProject.src.hypothesis import HypothesisTester
from ProjectCodes.PythonProject.src.feature_engineering import FeatureEngineer
from ProjectCodes.PythonProject.src.model import ModelTester

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    preprocessor = DataPreprocessor(config)
    df_final = None  # Will hold feature-engineered df

    menu = {
        "1": "Run Preprocessing",
        "2": "Run EDA",
        "3": "Run PCA/UMAP",
        "4": "Run Hypothesis Testing",
        "5": "Run Feature Engineering",
        "6": "Run Modeling",
        "0": "Exit"
    }

    while True:
        print("\nChoose a step to run:")
        for k, v in menu.items():
            print(f"{k}: {v}")

        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Exiting pipeline.")
            break
        elif choice == "1":
            preprocessor.preprocess()
        elif choice == "2":
            if preprocessor.df_eda is None:
                print("⚠️ Please run preprocessing first! (Loads the Dataset)")
            else:
                eda = EDAAnalyzer(preprocessor.df_eda, config)
                eda.run_all()
        elif choice == "3":
            if preprocessor.df is None or preprocessor.df_eda is None:
                print("⚠️ Please run preprocessing first! (Loads and STD)")
            else:
                dim_red = DimensionalityReducer(preprocessor.df, preprocessor.df_eda, config)
                dim_red.run_pca()
                dim_red.run_umap()
        elif choice == "4":
            if preprocessor.df_eda is None:
                print("⚠️ Please run preprocessing first! (Loads the Dataset)")
            else:
                hyp = HypothesisTester(preprocessor.df_hyp, config)
                hyp.run_all()
        elif choice == "5":
            if preprocessor.df is None:
                print("⚠️ Please run preprocessing first! (Loads and STD)")
            else:
                fe = FeatureEngineer(preprocessor.df, config)
                df_final = fe.run_all()
        elif choice == "6":
            if df_final is None:
                print("⚠️ Please run Feature Engineering first!")
            else:
                model_tester = ModelTester(df_final, config=config)
                model_tester.prepare_data()
                model_tester.run_all_models()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()