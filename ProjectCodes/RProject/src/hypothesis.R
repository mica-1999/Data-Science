library(R6)
library(dplyr)

HypothesisTester <- R6Class("HypothesisTester",
  public = list(
    df_hyp = NULL,
    config = NULL,

    # -------------------- INIT --------------------
    initialize = function(df_hyp, config) {
      self$df_hyp <- df_hyp
      self$config <- config
    },

    # -------------------- HYP 1 --------------------
    run_h1_distance_vs_delay = function() {
      # Hypothesis 1: Correlation between flight distance and arrival delay
      df <- self$df_hyp %>% filter(!is.na(ARR_DELAY), !is.na(DISTANCE))

      test <- cor.test(df$DISTANCE, df$ARR_DELAY, method = "pearson")
      corr <- test$estimate
      p <- test$p.value

      cat("H1: Distance vs Arrival Delay\n")
      cat(sprintf("Pearson correlation coefficient: %.3f, p-value: %.2e\n", corr, p))

      if (p < 0.05) {
        if (abs(corr) < 0.05) {
          cat("Statistically significant but correlation is extremely weak; practically negligible.\n\n")
        } else {
          cat("Significant correlation: flight distance is associated with delays.\n\n")
        }
      } else {
        cat("No significant correlation between distance and delay.\n\n")
      }
    },

    # -------------------- HYP 2 --------------------
    run_h2_airline_ttest = function() {
      # Hypothesis 2: Compare mean arrival delays between two airlines (Welch's t-test)
      airlines <- self$config$hypothesis$ttest_airlines

      a <- self$df_hyp %>% filter(AIRLINE == airlines[[1]]) %>% pull(ARR_DELAY) %>% na.omit()
      b <- self$df_hyp %>% filter(AIRLINE == airlines[[2]]) %>% pull(ARR_DELAY) %>% na.omit()

      # Welch's t-test (equal variances not assumed)
      test <- t.test(a, b, var.equal = FALSE)

      cat(sprintf("H2: %s vs %s mean arrival delays\n", airlines[[1]], airlines[[2]]))
      cat(sprintf("T-statistic: %.3f, p-value: %.2e\n", test$statistic, test$p.value))

      if (test$p.value < 0.05) {
        cat(sprintf("Significant difference in mean delays between %s and %s.\n\n",
                    airlines[[1]], airlines[[2]]))
      } else {
        cat(sprintf("No significant difference in mean delays between %s and %s.\n\n",
                    airlines[[1]], airlines[[2]]))
      }
    },

    # -------------------- HYP 3 --------------------
    run_h3_airline_anova = function() {
      # Hypothesis 3: All airlines have the same mean arrival delay (ANOVA)
      df <- self$df_hyp %>% filter(!is.na(ARR_DELAY))

      # One-way ANOVA
      model <- aov(ARR_DELAY ~ AIRLINE, data = df)
      result <- summary(model)[[1]]

      f_stat <- result$`F value`[1]
      p <- result$`Pr(>F)`[1]

      cat("H3: ANOVA across all airlines\n")
      cat(sprintf("F-statistic: %.3f, p-value: %.2e\n", f_stat, p))

      if (p < 0.05) {
        cat("Significant differences exist in delays between airlines.\n\n")
      } else {
        cat("No significant differences in delays between airlines.\n\n")
      }
    },

    # -------------------- HYP 4 --------------------
    run_h4_weather_vs_delay = function() {
      # Hypothesis 4: Correlation between weather-related delays and arrival delays
      weather_col <- self$config$hypothesis$weather_col

      df <- self$df_hyp %>% filter(!is.na(.data[[weather_col]]), !is.na(ARR_DELAY))

      test <- cor.test(df[[weather_col]], df$ARR_DELAY, method = "pearson")
      corr <- test$estimate
      p <- test$p.value

      cat(sprintf("H4: %s vs Arrival Delay\n", weather_col))
      cat(sprintf("Pearson correlation coefficient: %.3f, p-value: %.2e\n", corr, p))

      if (p < 0.05) {
        cat("Weather-related delays significantly impact arrival delays.\n\n")
      } else {
        cat("Weather-related delays do not significantly impact arrival delays.\n\n")
      }
    },

    # -------------------- HYP 5 --------------------
    run_h5_dep_hour_anova = function() {
      # Hypothesis 5: Differences in mean arrival delays across scheduled departure hours
      df <- self$df_hyp %>%
        filter(!is.na(ARR_DELAY)) %>%
        mutate(DEP_HOUR = CRS_DEP_TIME %/% 100)

      model <- aov(ARR_DELAY ~ factor(DEP_HOUR), data = df)
      result <- summary(model)[[1]]

      f_stat <- result$`F value`[1]
      p <- result$`Pr(>F)`[1]

      cat("H5: Departure Hour vs Arrival Delay\n")
      cat(sprintf("F-statistic: %.3f, p-value: %.2e\n", f_stat, p))

      if (p < 0.05) {
        cat("Significant differences in mean arrival delays across departure hours.\n\n")
      } else {
        cat("No significant differences in mean arrival delays across departure hours.\n\n")
      }
    },

    # -------------------- RUN ALL --------------------
    run_all = function() {
      cat("\n", strrep("=", 20), " HYPOTHESIS TESTING ", strrep("=", 20), "\n")
      self$run_h1_distance_vs_delay()
      self$run_h2_airline_ttest()
      self$run_h3_airline_anova()
      self$run_h4_weather_vs_delay()
      self$run_h5_dep_hour_anova()
    }
  )
)