library(R6)
library(dplyr)
library(readr)

FeatureEngineer <- R6Class("FeatureEngineer",
  public = list(
    df = NULL,
    config = NULL,

    # -------------------- INIT --------------------
    initialize = function(df, config) {
      self$df <- df
      self$config <- config
    },

    # -------------------- CATEGORICAL ENCODING --------------------
    encode_categorical = function() {
      # OneHot Encoding
      onehot_cols <- self$config$feature_engineering$categorical$onehot
      for (col in onehot_cols) {
        dummies <- model.matrix(~ . - 1, data = self$df[, col, drop = FALSE])
        colnames(dummies) <- gsub(col, paste0("encoded_", tolower(col)), colnames(dummies))
        colnames(dummies) <- gsub(" ", "_", colnames(dummies))
        self$df <- cbind(self$df, dummies)
        self$df[[col]] <- NULL
      }

      # Label Encoding for high cardinality columns
      label_cols <- self$config$feature_engineering$categorical$label
      for (col in label_cols) {
        self$df[[paste0(col, "_label")]] <- as.integer(factor(self$df[[col]]))
        self$df[[col]] <- NULL
      }

      cat("One-hot columns encoded:", paste(onehot_cols, collapse = ", "), "\n")
      cat("Label-encoded columns:", paste(label_cols, collapse = ", "), "\n")
    },

    # -------------------- BINNING --------------------
    apply_binning = function() {
      # Departure hour
      self$df$DEP_HOUR <- self$df$CRS_DEP_TIME %/% 100

      dep_bins   <- self$config$feature_engineering$binning$dep_hour$bins
      dep_labels <- self$config$feature_engineering$binning$dep_hour$labels
      self$df$dep_time_bin <- cut(self$df$DEP_HOUR,
                                   breaks = dep_bins,
                                   labels = dep_labels,
                                   right = FALSE)

      # Flight duration bin
      dur_bins   <- self$config$feature_engineering$binning$elapsed_time$bins
      dur_labels <- self$config$feature_engineering$binning$elapsed_time$labels
      self$df$flight_duration_bin <- cut(self$df$CRS_ELAPSED_TIME,
                                          breaks = dur_bins,
                                          labels = dur_labels,
                                          right = FALSE)

      # Distance bin
      dist_bins   <- self$config$feature_engineering$binning$distance$bins
      dist_labels <- self$config$feature_engineering$binning$distance$labels
      self$df$distance_bin <- cut(self$df$DISTANCE,
                                   breaks = dist_bins,
                                   labels = dist_labels,
                                   include.lowest = TRUE)

      cat("Departure hour bins:", paste(dep_labels, collapse = ", "), "\n")
      cat("Flight duration bins:", paste(dur_labels, collapse = ", "), "\n")
      cat("Distance bins:", paste(dist_labels, collapse = ", "), "\n")
    },

    # -------------------- INTERACTION FEATURES --------------------
    add_interactions = function() {
      self$df$elapsed_x_distance <- self$df$CRS_ELAPSED_TIME * self$df$DISTANCE
      self$df$dep_hour_x_elapsed <- self$df$DEP_HOUR * self$df$CRS_ELAPSED_TIME
      cat("Interaction features added: elapsed_x_distance, dep_hour_x_elapsed\n")
    },

    # -------------------- TIME FEATURES --------------------
    add_time_features = function() {
      self$df$FL_DATE    <- as.Date(self$df$FL_DATE)
      self$df$DAY_OF_WEEK <- as.integer(format(self$df$FL_DATE, "%u")) - 1  # 0=Mon, 6=Sun
      self$df$MONTH      <- as.integer(format(self$df$FL_DATE, "%m"))
      self$df$IS_WEEKEND <- as.integer(self$df$DAY_OF_WEEK %in% c(5, 6))
      self$df$IS_RUSH_HOUR <- as.integer(self$df$DEP_HOUR >= 16 & self$df$DEP_HOUR <= 20)
      self$df$FL_DATE    <- NULL

      cat("Time-based features added: DAY_OF_WEEK, MONTH, IS_WEEKEND, IS_RUSH_HOUR\n")
    },

    # -------------------- RUN ALL --------------------
    run_all = function() {
      cat("\n", strrep("=", 20), " FEATURE ENGINEERING ", strrep("=", 20), "\n")

      self$encode_categorical()
      self$apply_binning()
      self$add_interactions()
      self$add_time_features()

      output_path <- self$config$output_dataset$cleaned_scaled_new_features_r
      write_csv(self$df, output_path)

      cat("Dataset saved to:", output_path, "\n")
      cat("Feature engineering complete. Preview of dataset:\n")
      print(head(self$df))

      return(self$df)
    }
  )
)