library(R6)
library(readr)
library(dplyr)
library(yaml)

DataPreprocessor <- R6Class("DataPreprocessor",
  public = list(
    config = NULL,
    csv_path = NULL,
    min_distance = NULL,
    max_distance = NULL,
    future_columns = NULL,
    numeric_columns_for_outliers = NULL,
    output_path_cleaned = NULL,
    scaling_columns = NULL,
    df = NULL,
    df_eda = NULL,
    df_hyp = NULL,
    df_cleaned = NULL,

    # -------------------- INIT --------------------
    initialize = function(config) {
      self$config <- config
      self$csv_path <- config$preprocessing$csv_path
      self$min_distance <- config$preprocessing$distance_range$min
      self$max_distance <- config$preprocessing$distance_range$max
      self$future_columns <- config$preprocessing$future_columns
      self$numeric_columns_for_outliers <- config$preprocessing$numeric_columns_for_outliers
      self$output_path_cleaned <- config$output_dataset$cleaned_r
      self$scaling_columns <- config$preprocessing$scaling_columns
    },

    # -------------------- LOAD DATA --------------------
    load_data = function() {
      tryCatch({
        self$df <- read_csv(self$csv_path, show_col_types = FALSE)
        cat("Data loaded from:", self$csv_path, "\n")
        cat("Original dataset shape:", nrow(self$df), "rows x", ncol(self$df), "cols\n")
      }, error = function(e) {
        cat("Error loading CSV:", conditionMessage(e), "\n")
      })
    },

    # -------------------- ROW CLEANING --------------------
    initial_cleaning = function() {
      self$df <- self$df %>%
        filter(!is.na(CRS_ELAPSED_TIME)) %>%
        filter(CANCELLED == 0) %>%
        filter(DIVERTED == 0) %>%
        filter(!is.na(ARR_DELAY)) %>%
        filter(DISTANCE >= self$min_distance & DISTANCE <= self$max_distance)

      # Reset index (row names) 
      rownames(self$df) <- NULL

      # Copy for EDA and hypothesis testing
      self$df_eda <- self$df
      self$df_hyp <- self$df

      cat("After cleaning (nulls, cancelled/diverted, distance filter):",
          nrow(self$df), "rows x", ncol(self$df), "cols\n")
    },

    # -------------------- COL CLEANING --------------------
    drop_future_columns = function() {
      cols_to_drop <- self$future_columns[self$future_columns %in% colnames(self$df)]
      self$df <- self$df %>% select(-all_of(cols_to_drop))
      cat("Columns dropped (future info):", paste(cols_to_drop, collapse = ", "), "\n")
    },

    # -------------------- OUTLIER HANDLING --------------------
    handle_outliers = function(numeric_cols = NULL) {
      if (is.null(numeric_cols)) {
        numeric_cols <- self$numeric_columns_for_outliers
      }

      for (col in numeric_cols) {
        Q1 <- quantile(self$df[[col]], 0.25, na.rm = TRUE)
        Q3 <- quantile(self$df[[col]], 0.75, na.rm = TRUE)
        IQR <- Q3 - Q1
        lower <- Q1 - 1.5 * IQR
        upper <- Q3 + 1.5 * IQR

        # Replace outliers with NA then fill with median
        self$df[[col]] <- ifelse(
          self$df[[col]] < lower | self$df[[col]] > upper,
          NA,
          self$df[[col]]
        )
        col_median <- median(self$df[[col]], na.rm = TRUE)
        self$df[[col]][is.na(self$df[[col]])] <- col_median
      }

      cat("Outliers handled for columns:", paste(numeric_cols, collapse = ", "), "\n")
    },

    # -------------------- SAVE CLEAN SET --------------------
    save_cleaned_copy = function(output_path = NULL) {
      self$df_cleaned <- self$df

      if (!is.null(output_path)) {
        write_csv(self$df_cleaned, output_path)
        cat("Cleaned dataset saved to:", output_path, "\n")
      }
    },

    # -------------------- SCALING FOR PCA/UMAP --------------------
    apply_scaling = function() {
      numeric_cols <- self$scaling_columns

      # Standardization (Z-score)
      df_std <- as.data.frame(scale(self$df_cleaned[, numeric_cols]))
      colnames(df_std) <- paste0(numeric_cols, "_std")

      # Min-Max Normalization
      minmax_scale <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
      df_minmax <- as.data.frame(lapply(self$df_cleaned[, numeric_cols], minmax_scale))
      colnames(df_minmax) <- paste0(numeric_cols, "_minmax")

      # Combine with cleaned dataset
      self$df <- cbind(self$df_cleaned, df_std, df_minmax)

      cat("Standardization and normalization done.\n")
      cat("Scaling applied on columns:", paste(numeric_cols, collapse = ", "), "\n")
    },

    # -------------------- RUN PREPROCESSING --------------------
    preprocess = function() {
      cat("\n", strrep("=", 20), " PREPROCESSING ", strrep("=", 20), "\n")

      self$load_data()
      if (is.null(self$df)) {
        cat("Preprocessing aborted: data failed to load.\n")
        return(invisible(NULL))
      }

      self$initial_cleaning()
      self$drop_future_columns()
      self$handle_outliers()
      self$save_cleaned_copy(output_path = self$output_path_cleaned)
      self$apply_scaling()

      cat("\nPreview of cleaned dataset:\n")
      print(head(self$df))
    }
  )
)