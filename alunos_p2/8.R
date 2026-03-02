# Set the working directory to the script's directory
setwd('Place the path to the file 8.R in here')

# Install necessary libraries
install.packages(c("ROSE", "ggplot2"))

# Load required libraries
library(ROSE)
library(ggplot2)


# Load the dataset
data <- read.csv("data.csv")

dev.new()  # Open a new graphics window

# Show histogram of the original target distribution


# Oversample the data using ROSE
oversampled_data <- ovun.sample(target ~ ., data = data, method = "over", seed = 42)$data

dev.new()  # Open a new graphics window

# Show histogram of the new target distribution after oversampling

  
dev.new()  # Open a new graphics window

# Undersample the data using ROSE


# Show histogram of the new target distribution after undersampling
