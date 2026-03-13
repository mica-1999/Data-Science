library(R6)
library(dplyr)
library(ggplot2)

EDAAnalyzer <- R6Class("EDAAnalyzer",
  public = list(
    df = NULL,
    config = NULL,
    numeric_cols_summary = NULL,
    numeric_cols_hist = NULL,
    scatter_plots = NULL,
    output_dir = NULL,

    # -------------------- INIT --------------------
    initialize = function(df, config) {
      self$df <- df
      self$config <- config
      self$numeric_cols_summary <- config$eda$numeric_summary
      self$numeric_cols_hist <- config$eda$numeric_hist
      self$scatter_plots <- config$eda$scatter_plots
      self$output_dir <- config$output_dir_r
      dir.create(self$output_dir, showWarnings = FALSE, recursive = TRUE)
    },

    # -------------------- BASIC SUMMARY STATS --------------------
    summary_statistics = function() {
      cat("\nBasic statistics:\n")
      print(summary(self$df[, self$numeric_cols_summary]))
    },

    # -------------------- TOP AIRLINES --------------------
    plot_top_airlines = function(top_n = 10) {
      airline_counts <- self$df %>%
        count(AIRLINE, sort = TRUE) %>%
        slice_head(n = top_n)

      p <- ggplot(airline_counts, aes(x = n, y = reorder(AIRLINE, n))) +
        geom_bar(stat = "identity", fill = "steelblue") +
        labs(
          title = paste("Top", top_n, "Airlines by Number of Flights"),
          x = "Count",
          y = "Airline"
        ) +
        theme_minimal()

      ggsave(file.path(self$output_dir, "top_airlines_count.png"), plot = p,
             width = 12, height = 6, bg = "white")
      cat("Top airlines plot saved.\n")
    },

    # -------------------- HISTOGRAM PLOTS --------------------
    plot_histograms = function() {
      for (col in self$numeric_cols_hist) {
        if (!col %in% colnames(self$df)) next

        p <- ggplot(self$df, aes(x = .data[[col]])) +
          geom_histogram(aes(y = after_stat(density)), bins = 50,
                         fill = "steelblue", color = "white", alpha = 0.7) +
          geom_density(color = "red", linewidth = 0.8) +
          labs(title = paste("Distribution of", col), x = col, y = "Density") +
          theme_minimal()

        ggsave(file.path(self$output_dir, paste0("hist_", col, ".png")), plot = p,
               width = 8, height = 4, bg = "white")
      }
      cat("Histograms saved.\n")
    },

    # -------------------- BOXPLOT --------------------
    plot_boxplots_by_airline = function(target_cols = c("ARR_DELAY")) {
      for (col in target_cols) {
        if (!col %in% colnames(self$df)) next

        p <- ggplot(self$df, aes(x = AIRLINE, y = .data[[col]])) +
          geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.alpha = 0.1) +
          coord_cartesian(ylim = c(-100, 100)) +
          labs(title = paste("Boxplot of", col, "by Airline"),
               x = "Airline", y = col) +
          theme_minimal() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))

        ggsave(file.path(self$output_dir, paste0("boxplot_", col, "_by_airline.png")),
               plot = p, width = 12, height = 6, bg = "white")
      }
      cat("Boxplots saved.\n")
    },

    # -------------------- FEATURES CORRELATION --------------------
    plot_correlation_heatmap = function() {
      # Select only existing numeric cols
      cols <- self$numeric_cols_hist[self$numeric_cols_hist %in% colnames(self$df)]
      corr_matrix <- cor(self$df[, cols], use = "complete.obs")

      # Melt for ggplot
      corr_df <- as.data.frame(as.table(corr_matrix))
      colnames(corr_df) <- c("Var1", "Var2", "Correlation")

      p <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
        geom_tile() +
        geom_text(aes(label = round(Correlation, 2)), size = 2.5) +
        scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
        labs(title = "Correlation Heatmap") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))

      ggsave(file.path(self$output_dir, "corr_heatmap.png"), plot = p,
             width = 12, height = 10, bg = "white")
      cat("Correlation heatmap saved.\n")
    },

    # -------------------- KDE PLOT --------------------
    plot_kde_by_airline = function(target_cols = c("ARR_DELAY")) {
      for (col in target_cols) {
        if (!col %in% colnames(self$df)) next

        p <- ggplot(self$df, aes(x = .data[[col]], fill = AIRLINE, color = AIRLINE)) +
          geom_density(alpha = 0.4) +
          labs(title = paste("KDE Plot of", col, "by Airline"),
               x = col, y = "Density") +
          theme_minimal()

        ggsave(file.path(self$output_dir, paste0("kde_", col, "_by_airline.png")),
               plot = p, width = 12, height = 6, bg = "white")
      }
      cat("KDE plots saved.\n")
    },

    # -------------------- SCATTER PLOTS --------------------
    plot_scatter = function(scatter_configs = NULL) {
      if (is.null(scatter_configs)) {
        scatter_configs <- self$scatter_plots
      }

      for (entry in scatter_configs) {
        x_col    <- entry[[1]]
        y_col    <- entry[[2]]
        title    <- entry[[3]]
        filename <- entry[[4]]

        if (!x_col %in% colnames(self$df) || !y_col %in% colnames(self$df)) next

        # Sample for performance
        plot_df <- self$df %>% sample_n(min(50000, nrow(self$df)))

        p <- ggplot(plot_df, aes(x = .data[[x_col]], y = .data[[y_col]])) +
          geom_point(alpha = 0.3, size = 0.5, color = "steelblue") +
          labs(title = paste("Scatter Plot:", title), x = x_col, y = y_col) +
          theme_minimal()

        ggsave(file.path(self$output_dir, filename), plot = p,
               width = 8, height = 6, bg = "white")
      }
      cat("Scatter plots saved.\n")
    },

    # -------------------- RUN ALL --------------------
    run_all = function() {
      cat("\n", strrep("=", 20), " EDA STEP ", strrep("=", 20), "\n")

      cat("\n1. Computing summary statistics...\n")
      self$summary_statistics()

      cat("2. Plotting top airlines by flight count...\n")
      self$plot_top_airlines()

      cat("3. Plotting histograms for numeric columns...\n")
      self$plot_histograms()

      cat("4. Plotting boxplots of ARR_DELAY by airline...\n")
      self$plot_boxplots_by_airline()

      cat("5. Plotting correlation heatmap for numeric features...\n")
      self$plot_correlation_heatmap()

      cat("6. Plotting KDE of ARR_DELAY by airline...\n")
      self$plot_kde_by_airline()

      cat("7. Generating scatter plots as per config...\n")
      self$plot_scatter()

      cat("EDA complete. All outputs saved to:", self$output_dir, "\n")
    }
  )
)