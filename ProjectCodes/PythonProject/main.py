import yaml
from ProjectCodes.PythonProject.src.preprocessing import DataPreprocessor
from ProjectCodes.PythonProject.src.eda import EDAAnalyzer
from ProjectCodes.PythonProject.src.pca_umap import DimensionalityReducer
from ProjectCodes.PythonProject.src.hypothesis import HypothesisTester
from ProjectCodes.PythonProject.src.feature_engineering import FeatureEngineer
from ProjectCodes.PythonProject.src.model import ModelTester
from ProjectCodes.PythonProject.src.kNN import KNNRunner
from ProjectCodes.PythonProject.src.supervised_learning import SupervisedLearningRunner
from ProjectCodes.PythonProject.src.ensemble_learning import EnsembleLearningRunner
from ProjectCodes.PythonProject.src.deep_learning import DeepLearningRunner
from ProjectCodes.PythonProject.src.clustering import ClusteringRunner

# Load config
with open("../config.yaml", "r") as f:
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
        "6": "Run Post-Engineering EDA",
        "7": "Run Baseline Modeling (Linear Regression)",
        "8": "Run kNN from Scratch (Phase 2)",
        "9": "Run Supervised Learning Models (Decision Tree + SVR)",
        "10": "Run Ensemble Learning (Random Forest + Gradient Boosting)",
        "11": "Run Deep Learning Model",
        "12": "Run Clustering Analysis",
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
                print("⚠️ Please run Feature Engineering first! (Step 5)")
            else:
                eda_post = EDAAnalyzer(df_final, config)
                eda_post.run_post_engineering()
        elif choice == "7":
            if df_final is None:
                print("⚠️ Please run Feature Engineering first! (Step 5)")
            else:
                model_tester = ModelTester(df_final, config=config)
                model_tester.prepare_data()
                model_tester.run_all_models()
        elif choice == "8":
            if df_final is None:
                print("⚠️ Please run Feature Engineering first! (Step 5)")
            else:
                knn_runner = KNNRunner(df_final, config=config)
                knn_runner.run_all()
        elif choice == "9":
            if df_final is None:
                print("⚠️ Please run Feature Engineering first! (Step 5)")
            else:
                supervised_runner = SupervisedLearningRunner(df_final, config=config)
                supervised_runner.run_all()
        elif choice == "10":
            if df_final is None:
                print("⚠️ Please run Feature Engineering first! (Step 5)")
            else:
                ensemble_runner = EnsembleLearningRunner(df_final, config=config)
                ensemble_runner.run_all()
        elif choice == "11":
            if df_final is None:
                print("⚠️ Please run Feature Engineering first! (Step 5)")
            else:
                dl_runner = DeepLearningRunner(df_final, config=config)
                dl_runner.run_all()
        elif choice == "12":
            if df_final is None:
                print("⚠️ Please run Feature Engineering first! (Step 5)")
            else:
                clustering_runner = ClusteringRunner(df_final, config=config)
                clustering_runner.run_all()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()