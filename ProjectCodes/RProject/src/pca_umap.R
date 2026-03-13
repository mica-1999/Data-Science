library(R6)
library(dplyr)
library(ggplot2)
library(uwot)

DimensionalityReducer <- R6Class("DimensionalityReducer",
  public = list(
    df_scaled = NULL,
    df_eda = NULL,
    config = NULL,
    output_dir = NULL,
    features = NULL,
    n_components = NULL,
    umap_sample_size = NULL,
    umap_n_neighbors = NULL,
    umap_min_dist = NULL,
    umap_alpha = NULL,

    # -------------------- INIT --------------------
    initialize = function(df_scaled, df_eda, config) {
      self$df_scaled <- df_scaled
      self$df_eda <- df_eda
      self$config <- config
      self$output_dir <- config$output_dir_r
      dir.create(self$output_dir, showWarnings = FALSE, recursive = TRUE)

      self$features <- config$dim_reduction$features_for_dr
      self$n_components <- config$dim_reduction$n_components %||% 2
      self$umap_sample_size <- config$dim_reduction$umap_sample_size
      self$umap_n_neighbors <- config$dim_reduction$umap_n_neighbors
      self$umap_min_dist <- config$dim_reduction$umap_min_dist
      self$umap_alpha <- config$dim_reduction$umap_alpha
    },

    # -------------------- PCA --------------------
    run_pca = function() {
      cat("\n", strrep("=", 20), " PCA ", strrep("=", 20), "\n")

      # Select features
      features_for_dr <- self$df_scaled[, self$features]

      # Fit PCA
      pca_result <- prcomp(features_for_dr, center = FALSE, scale. = FALSE)

      # Extract scores for first n components
      scores <- as.data.frame(pca_result$x[, 1:self$n_components])
      colnames(scores) <- paste0("PC", 1:self$n_components)
      scores$AIRLINE <- self$df_eda$AIRLINE

      # Explained variance
      explained_var <- summary(pca_result)$importance[2, 1:self$n_components]
      total_var <- sum(explained_var)
      cat("Explained variance ratio:", paste(round(explained_var, 4), collapse = ", "), "\n")
      cat("Total variance explained:", round(total_var, 4), "\n")

      # -------------------- SCATTER PLOT --------------------
      cat("Plotting PCA Scatter projection...\n")
      p_scatter <- ggplot(scores, aes(x = PC1, y = PC2, color = AIRLINE)) +
        geom_point(alpha = 0.6, size = 0.5) +
        labs(
          title = "PCA Projection of Flights",
          x = "Principal Component 1",
          y = "Principal Component 2"
        ) +
        theme_minimal() +
        theme(legend.position = "right")

      ggsave(file.path(self$output_dir, "pca_projection.png"), plot = p_scatter,
             width = 14, height = 6, bg = "white")

      # -------------------- SCREE PLOT --------------------
      cat("Plotting PCA Scree plot...\n")
      scree_df <- data.frame(
        PC = paste0("PC", 1:self$n_components),
        Variance = explained_var
      )

      p_scree <- ggplot(scree_df, aes(x = PC, y = Variance)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        labs(
          title = "PCA Scree Plot",
          x = "Principal Component",
          y = "Explained Variance Ratio"
        ) +
        theme_minimal()

      ggsave(file.path(self$output_dir, "pca_scree.png"), plot = p_scree,
             width = 6, height = 4, bg = "white")

      cat("PCA plots saved to:", self$output_dir, "\n")
      return(list(scores = scores, pca = pca_result))
    },

    # -------------------- UMAP --------------------
    run_umap = function() {
      cat("\n", strrep("=", 20), " UMAP ", strrep("=", 20), "\n")

      # Sample for faster computation
      sample_size <- min(self$umap_sample_size, nrow(self$df_scaled))
      set.seed(42)
      sample_idx <- sample(nrow(self$df_scaled), sample_size)

      features_sample <- self$df_scaled[sample_idx, self$features]
      airline_sample <- self$df_eda$AIRLINE[sample_idx]

      # Run UMAP
      cat("Running UMAP on", sample_size, "samples...\n")
      umap_result <- umap(
        as.matrix(features_sample),
        n_components = 2,
        n_neighbors = self$umap_n_neighbors,
        min_dist = self$umap_min_dist,
        seed = 42
      )

      # Create result dataframe
      df_umap <- data.frame(
        UMAP1 = umap_result[, 1],
        UMAP2 = umap_result[, 2],
        AIRLINE = airline_sample
      )

      # -------------------- UMAP PLOT --------------------
      cat("Plotting UMAP projection...\n")
      p_umap <- ggplot(df_umap, aes(x = UMAP1, y = UMAP2, color = AIRLINE)) +
        geom_point(alpha = self$umap_alpha, size = 0.5) +
        labs(
          title = "UMAP Projection of Flights (Sample)",
          x = "UMAP 1",
          y = "UMAP 2"
        ) +
        theme_minimal() +
        theme(legend.position = "right")

      ggsave(file.path(self$output_dir, "umap_projection.png"), plot = p_umap,
             width = 14, height = 6, bg = "white")

      cat("UMAP completed. First 5 rows of embedding:\n")
      print(head(umap_result, 5))
      cat("Both processes done. All outputs saved to:", self$output_dir, "\n")

      return(df_umap)
    }
  )
)

# Helper for NULL default (equivalent to Python's .get())
`%||%` <- function(a, b) if (!is.null(a)) a else b