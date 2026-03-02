# Set the working directory to the script's directory
setwd('Place the path to the file 1.R in here')

# Install and load necessary libraries
install.packages("caTools")
install.packages("MASS")
library(caTools)
library(MASS)
library(ggplot2)

# Load the California Housing dataset
df <- read.csv("california_housing.csv")

# Use 'price' as the target variable
y <- df$price

# Split the dataset into training and testing sets


# Create a linear regression model
model <- lm(price ~ ., data = train_data)

# Make predictions on the test set


# Evaluate the model
mse <- mean((y_pred - test_data$price)^2)
r2 <- 1 - mse / var(test_data$price)

# Print the resuts


# Plot the predicted vs actual values with tendency line
