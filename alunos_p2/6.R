# Install and load required libraries
install.packages("tidyverse")
library(tidyverse)

# Load the Iris dataset (built-in dataset)
data(iris)

# Convert the iris dataset to a data frame
iris_df <- as.data.frame(iris)

# Display the initial dataset
cat("Initial Iris dataset:\n")
print(head(iris_df))

# Simple combinations
iris_df <- iris_df %>%
  mutate(
    sepal_length_width_product = Sepal.Length * Sepal.Width,
    petal_length_width_sum = Petal.Length + Petal.Width,
    sepal_length_petal_length_ratio = Sepal.Length / Petal.Length
  )

# Display the dataset after creating simple interactions


# Complex combination with a nonlinear interaction


# Display the dataset after creating complex interactions
