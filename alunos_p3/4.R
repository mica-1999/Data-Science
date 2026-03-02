# Install and load necessary libraries
required_libraries <- c("dplyr", "stats")

for (lib in required_libraries) {
  if (!requireNamespace(lib, quietly = TRUE)) {
    install.packages(lib)
  }
}

# Load the necessary libraries
library(dplyr)
library(stats)

# Set the working directory to the script's directory
setwd('/Users/fabio/Documents/UMa/Ciencia de dados/TP-03')

# Read the California Housing dataset
data <- read.csv("california_housing.csv")

# Extract features and target variable
features <- data[, -ncol(data)]
target <- data[, ncol(data)]

# Normalize the features
scaled_features <- scale(features)

# Combine scaled features and target variable
df_scaled <- as.data.frame(cbind(scaled_features, target))

# Determine the means of housing prices for houses with high and low median income to then perform t-test
# Using 1 as threshold to define the high and low income groups (in the MedInc column)
high_income_prices <- df_scaled$target[df_scaled$MedInc > 1] 
low_income_prices <- df_scaled$target[df_scaled$MedInc <= 1]  

# Perform t-test
t_test_result <- t.test(high_income_prices, low_income_prices)

# Print t-test results
cat("\nT-Test Results:\n")
cat("------------------------------------------------------\n")
cat("Comparison: Mean Housing Prices for High vs. Low Income\n")
cat("------------------------------------------------------\n")

cat("High Income Mean Price: ", mean(high_income_prices), "\n")
cat("Low Income Mean Price: ", mean(low_income_prices), "\n")

cat("\nT-Test Results:\n")
cat("  - T-Statistic: ", t_test_result$statistic, "\n")
cat("  - Degrees of Freedom: ", t_test_result$parameter, "\n")
cat("  - P-Value: ", t_test_result$p.value, "\n")

cat("\nConclusion:\n")
if (t_test_result$p.value < 0.05) {
  cat("  There is a significant difference in mean housing prices between high and low-income houses.\n")
} else {
  cat("  There is no significant difference in mean housing prices between high and low-income houses.\n")
}
