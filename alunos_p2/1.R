# Install and load necessary libraries
install.packages(c("ggplot2", "reshape2", "dplyr"))

library(ggplot2)
library(reshape2)  
library(dplyr)

# Load the Iris dataset
iris <- as.data.frame(iris)  # Load the dataset

# Display basic information about the dataset


# Display the first few rows of the dataset


# Summary statistics (using the dplyr package)


# Convert to a matrix without using the last column (which is the label)
iris_matrix <- iris[, -5]

# Summary statistics with base R
cat("\nSummary Statistics with base R:\n")
# Mean


# Median


# Standard Deviation


# Data distribution analysis
# Pairplot for overall distribution
dev.new()  # Open a new graphics window
pairplot <- ggplot(iris, aes(x=Sepal.Length, y=Sepal.Width, color=Species)) +
  geom_point() +
  facet_grid(.~Species)
print(pairplot)

# Boxplot for each feature


# Correlation Analysis


# Heatmap for correlation matrix
