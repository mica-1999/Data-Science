# Install and load necessary packages
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("gridExtra", quietly = TRUE)) {
  install.packages("gridExtra")
}
library(ggplot2)

# Load the iris dataset
data(iris)

# Create a combined violin plot with subplots and boxplots
features <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

dev.new()  # Open a new graphics window
# Create a list to store individual ggplot objects
plots_list <- list()

# Loop through each feature and create combined histogram and boxplot
for (feature in features) {
  p <- ggplot(iris, aes_string(x = feature, fill = "Species")) +
    geom_histogram(position = "identity", alpha = 0.7, bins = 30) +
    geom_boxplot(width = 0.2, position = position_dodge(0.75), outlier.shape = NA) +
    labs(title = paste("Histogram and Box Plot of", feature, "by Species"),
         x = feature,
         y = "Frequency") +
    theme_minimal()
  
  plots_list[[feature]] <- p
}

# Combine individual plots into a single plot with subplots
combined_plot <- do.call(gridExtra::grid.arrange, c(plots_list, ncol = 2))

# Print the combined plot
print(combined_plot)


dev.new()  # Open a new graphics window
# Create a list to store individual ggplot objects
plots_list <- list()

# Loop through each feature and create combined violin and boxplot


# Combine individual plots into a single plot with subplots


# Print the combined plot



dev.new()  # Open a new graphics window
# Create a list to store individual ggplot objects
plots_list <- list()

# Loop through each feature and create combined KDE and boxplot


# Combine individual plots into a single plot with subplots


# Print the combined plot



dev.new()  # Open a new graphics window
# Create a list to store individual ggplot objects
plots_list <- list()

# Loop through each feature and create combined swarm plot and boxplot


# Combine individual plots into a single plot with subplots


# Print the combined plot
