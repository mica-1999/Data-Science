library(R6)
library(dplyr)
library(ggplot2)
library(randomForest)

ModelTester <- R6Class("ModelTester",
  public = list(
    df = NULL,
    config = NULL,
    target_col = NULL,
    test_size = NULL,
    random_state = NULL,
    drop_cols = NULL,
    rf_n_estimators = NULL,
    rf_max_depth = NULL,
    rf_plot_top = NULL,
    rf_save_path = NULL,

    # Split data
    X_train = NULL,
    X_test = NULL,
    y_train = NULL,
    y_test = NULL,
    X = NULL,

    # Models
    lr_model = NULL,
    lr_pred = NULL,
    rf_model = NULL,
    rf_pred = NULL,
    rf_feature_importance = NULL,

    # -------------------- INIT --------------------
    initialize = function(df, config) {
      self$df <- df
      self$config <- config
      self$target_col <- config$modeling$target_col
      self$test_size <- config$modeling$test_size
      self$random_state <- config$modeling$random_state
      self$drop_cols <- config$modeling$drop_columns
      self$rf_n_estimators <- config$modeling$random_forest$n_estimators
      self$rf_max_depth <- config$modeling$random_forest$max_depth
      self$rf_plot_top <- config$modeling$random_forest$plot_top_features
      self$rf_save_path <- config$modeling$random_forest$save_feature_importance_path
    },

    # -------------------- PREPARE DATA --------------------
    prepare_data = function() {
      set.seed(self$random_state)

      # Drop leakage columns and separate target
      cols_to_drop <- self$drop_cols[self$drop_cols %in% colnames(self$df)]
      self$X <- self$df %>% select(-all_of(c(cols_to_drop)))
      self$X <- self$X %>% select(where(is.numeric))  # Keep only numeric cols
      y <- self$df[[self$target_col]]

      # Train/test split
      n <- nrow(self$X)
      test_idx <- sample(n, size = floor(self$test_size * n))
      train_idx <- setdiff(1:n, test_idx)

      self$X_train <- self$X[train_idx, ]
      self$X_test  <- self$X[test_idx, ]
      self$y_train <- y[train_idx]
      self$y_test  <- y[test_idx]

      cat("Data prepared. Train size:", nrow(self$X_train),
          "| Test size:", nrow(self$X_test), "\n")
    },

    # -------------------- LINEAR REGRESSION --------------------
    run_linear_regression = function() {
      cat("\nRunning Linear Regression...\n")

      train_df <- cbind(self$X_train, ARR_DELAY = self$y_train)
      self$lr_model <- lm(ARR_DELAY ~ ., data = train_df)
      self$lr_pred <- predict(self$lr_model, newdata = self$X_test)

      mae  <- mean(abs(self$y_test - self$lr_pred))
      rmse <- sqrt(mean((self$y_test - self$lr_pred)^2))
      ss_res <- sum((self$y_test - self$lr_pred)^2)
      ss_tot <- sum((self$y_test - mean(self$y_test))^2)
      r2 <- 1 - ss_res / ss_tot

      cat("Linear Regression complete.\n")
      cat(sprintf("MAE:  %.3f\n", mae))
      cat(sprintf("RMSE: %.3f\n", rmse))
      cat(sprintf("R2:   %.3f\n", r2))
    },

    # -------------------- RANDOM FOREST --------------------
    run_random_forest = function() {
      cat("\nRunning Random Forest...\n")

      # Sample for faster training during model selection phase
      sample_size <- min(500000, nrow(self$X_train))
      set.seed(self$random_state)
      sample_idx <- sample(nrow(self$X_train), sample_size)
      X_train_sample <- self$X_train[sample_idx, ]
      y_train_sample <- self$y_train[sample_idx]

      self$rf_model <- randomForest(
        x = X_train_sample,
        y = y_train_sample,
        ntree = self$rf_n_estimators,
        maxnodes = 2^self$rf_max_depth,
        importance = TRUE
      )

      self$rf_pred <- predict(self$rf_model, newdata = self$X_test)

      mae  <- mean(abs(self$y_test - self$rf_pred))
      rmse <- sqrt(mean((self$y_test - self$rf_pred)^2))
      ss_res <- sum((self$y_test - self$rf_pred)^2)
      ss_tot <- sum((self$y_test - mean(self$y_test))^2)
      r2 <- 1 - ss_res / ss_tot

      cat("Random Forest complete.\n")
      cat(sprintf("MAE:  %.3f\n", mae))
      cat(sprintf("RMSE: %.3f\n", rmse))
      cat(sprintf("R2:   %.3f\n", r2))

      # Feature importance
      importance_vals <- importance(self$rf_model)[, "%IncMSE"]
      self$rf_feature_importance <- data.frame(
        feature = names(importance_vals),
        importance = as.numeric(importance_vals)
      ) %>% arrange(desc(importance))

      cat("\nTop Features by Importance:\n")
      print(head(self$rf_feature_importance, self$rf_plot_top))

      # Plot feature importance
      top_features <- head(self$rf_feature_importance, self$rf_plot_top)

      p <- ggplot(top_features, aes(x = importance, y = reorder(feature, importance))) +
        geom_bar(stat = "identity", fill = "steelblue") +
        labs(
          title = "Top Features - Random Forest",
          x = "Importance (%IncMSE)",
          y = "Feature"
        ) +
        theme_minimal()

      ggsave(self$rf_save_path, plot = p, width = 10, height = 6, bg = "white")
      cat("Feature importance plot saved to:", self$rf_save_path, "\n")
    },

    # -------------------- RUN ALL --------------------
    run_all_models = function() {
      cat("\n", strrep("=", 20), " MODELING ", strrep("=", 20), "\n")
      self$run_linear_regression()
      self$run_random_forest()
      cat("Modeling complete.\n")
    }
  )
)