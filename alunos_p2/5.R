# Install and Load required libraries
if (!requireNamespace("tidyverse", quietly = TRUE)) {
  install.packages("tidyverse")
}
if (!requireNamespace("MASS", quietly = TRUE)) {
  install.packages("MASS")
}
  
library(tidyverse)
library(MASS)  # For the truehist function

# Load the Iris dataset
iris_data <- as.data.frame(cbind(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width, iris$Species))
colnames(iris_data) <- c("sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target")

# Display the initial dataset


# Choose the 'sepal length (cm)' column for binning


# Define the number of bins (or bin edges)
bins <- c(4, 5, 6, 7, 8)

# Perform binning using cut
iris_data$sepal_length_bin <- cut(iris_data[[feature_to_bin]], breaks = bins, labels = c('Bin 1', 'Bin 2', 'Bin 3', 'Bin 4'), include.lowest = TRUE)

# Display the dataset after binning
