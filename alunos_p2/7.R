# Function to check and install libraries
install_load_library <- function(lib_name) {
  if (!requireNamespace(lib_name, quietly = TRUE)) {
    install.packages(lib_name, dependencies = TRUE)
  }
  library(lib_name, character.only = TRUE)
}

# Check and install required libraries
required_libraries <- c("tibble", "scales", "caret")

for (lib in required_libraries) {
  install_load_library(lib)
}

# Load the Iris dataset
data(iris)
iris_df <- as_tibble(iris)

# Display the initial dataset

# Apply Standardization


# Convert the standardized matrix back to a DataFrame for display
iris_standardized_df <- as_tibble(iris_standardized)
colnames(iris_standardized_df) <- colnames(iris_df[, -5])

# Display the dataset after standardization

# Apply Min-Max Scaling
iris_minmax <- preProcess(iris_df[, -5], method = c("range"))

# Convert the min-max scaled matrix back to a DataFrame for display


# Display the dataset after min-max scaling
