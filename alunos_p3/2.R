# Load necessary libraries
install.packages("caTools")
install.packages("MASS")
install.packages("ROCR")
library(caTools)
library(MASS)
library(ggplot2)
library(caret)
library(ROCR)

# Set the working directory to the script's directory
setwd('Place the path to the file 2.R in here')

# Load the California Housing dataset
df <- read.csv("california_housing.csv")

# Define a binary target variable based on the threshold
df$target <- as.factor(ifelse(df$price > 3, 1, 0))

# Remove the 'price' column
df <- df[, -which(names(df) %in% c("price"))]

# Split the dataset into training and testing sets


# Create a logistic regression model
model <- glm(target ~ ., data = train_data, family = "binomial")

# Make predictions on the test set


# Confusion Matrix


# Plot the ROC Curve
roc_pred <- prediction(y_pred_prob, test_data$target)
roc_perf <- performance(roc_pred, "tpr", "fpr")

plot(roc_perf, col = "blue", main = "ROC Curve", lwd = 2)
abline(0, 1, lty = 2, col = "red")
