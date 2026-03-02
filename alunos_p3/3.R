# Install and load necessary libraries
required_libraries <- c("dplyr", "stats")


for (lib in required_libraries) {
  if (!requireNamespace(lib, quietly = TRUE)) {
    install.packages(lib)
  }
  library(lib, character.only = TRUE)
}

# Set the working directory to the script's directory
setwd('Place the path to the file 3.R in here')

perform_correlation_test <- function(feature1, feature2, alpha = 0.05) {
  # Calculate Pearson correlation coefficient and p-value
  correlation_result <- cor.test(feature1, feature2)

  # Interpret the results
  if (correlation_result$p.value < alpha) {
    cat("There is a significant correlation between the two features, the correlation coefficient was ", correlation_result$estimate, " and the p-value was ", correlation_result$p.value, "\n")
  } else {
    cat("There is no significant correlation between the two features, the correlation coefficient was ", correlation_result$estimate, " and the p-value was ", correlation_result$p.value, "\n")
  }

  return(correlation_result)
}

# Load the California Housing dataset
data <- read.csv("california_housing.csv")

# Extract features and target variable
features <- data[, -ncol(data)]
target <- data[, ncol(data)]

# Normalize the features


# Combine scaled features and target variable
df_scaled <- as.data.frame(cbind(scaled_features, target))

# Test all pairs of features

