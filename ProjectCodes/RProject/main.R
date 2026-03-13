library(yaml)
library(R6)
script_dir <- getSrcDirectory(function() {})
if (nchar(script_dir) == 0) {
  script_dir <- "C:/Users/Utilizador/Desktop/Ciencia Dados/ProjectCodes/RProject"
}
setwd(script_dir)

source(file.path(script_dir, "src/preprocessing.R"))
source(file.path(script_dir, "src/eda.R"))
source(file.path(script_dir, "src/pca_umap.R"))
source(file.path(script_dir, "src/hypothesis.R"))
source(file.path(script_dir, "src/feature_engineering.R"))
source(file.path(script_dir, "src/model.R"))
config <- read_yaml(file.path(script_dir, "../config.yaml"))

df_final <- NULL # used in the model , is returned by FE

# -------------------- MAIN --------------------
main <- function() {
  preprocessor <- DataPreprocessor$new(config)

  menu <- list(
    "1" = "Run Preprocessing",
    "2" = "Run EDA",
    "3" = "Run PCA/UMAP",
    "4" = "Run Hypothesis Testing",
    "5" = "Run Feature Engineering",
    "6" = "Run Modeling",
    "0" = "Exit"
  )

  repeat {
    cat("\nChoose a step to run:\n")
    for (k in names(menu)) {
      cat(sprintf("%s: %s\n", k, menu[[k]]))
    }

    choice <- trimws(readline("Enter your choice: "))

    if (choice == "0") {
      cat("Exiting pipeline.\n")
      break

    } else if (choice == "1") {
      preprocessor$preprocess()

    } else if (choice == "2") {
      if (is.null(preprocessor$df_eda)) {
        cat("Please run preprocessing first! (Loads the Dataset)\n")
      } else {
        eda <- EDAAnalyzer$new(preprocessor$df_eda, config)
        eda$run_all()
      }

    } else if (choice == "3") {
      if (is.null(preprocessor$df) || is.null(preprocessor$df_eda)) {
        cat("Please run preprocessing first! (Loads and STD)\n")
      } else {
        dim_red <- DimensionalityReducer$new(preprocessor$df, preprocessor$df_eda, config)
        dim_red$run_pca()
        dim_red$run_umap()
      }

    } else if (choice == "4") {
      if (is.null(preprocessor$df_hyp)) {
        cat("Please run preprocessing first! (Loads the Dataset)\n")
      } else {
        hyp <- HypothesisTester$new(preprocessor$df_hyp, config)
        hyp$run_all()
      }

    } else if (choice == "5") {
      if (is.null(preprocessor$df)) {
        cat("Please run preprocessing first! (Loads and STD)\n")
      } else {
        fe <- FeatureEngineer$new(preprocessor$df, config)
        df_final <<- fe$run_all()
      }

    } else if (choice == "6") {
      if (is.null(df_final)) {
        cat("Please run Feature Engineering first!\n")
      } else {
        model_tester <- ModelTester$new(df_final, config)
        model_tester$prepare_data()
        model_tester$run_all_models()
      }

    } else {
      cat("Invalid choice. Please try again.\n")
    }
  }
}

# -------------------- RUN --------------------
main()