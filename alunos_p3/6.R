# Install and load necessary libraries
required_packages <- c("caret", "nnet")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, dependencies=TRUE)
lapply(required_packages, library, character.only = TRUE)

# Load the Iris dataset
data(iris)

# Split the data into training and testing sets


# Train a Logistic Regression classifier with multinomial to use multiclass
clf <- nnet::multinom(Species ~ ., data = train_data, maxit = 10000)  # Increase max iterations
summary(clf)

# Evaluate on the test set predicting the accuracy


# Cross-validation with increased max iterations
set.seed(42)
cv_model <- train(Species ~ ., data = iris, method = "multinom",
                  trControl = trainControl(method = "cv", number = 5, summaryFunction = multiClassSummary, classProbs = TRUE),
                  maxit = 10000)  # Increase max iterations
cv_scores <- cv_model$results$Accuracy
cat('Cross-Validation Mean Accuracy:', round(mean(cv_scores), 2), '\n')
cat('Cross-Validation SD Accuracy:', round(sd(cv_scores), 2), '\n')

# Bootstrapping with increased max iterations
set.seed(42)
bootstrap_indices <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
bootstrap_data <- iris[bootstrap_indices, ]
bootstrap_scores <- numeric(100)

for (i in 1:100) {
  bootstrap_sample <- sample(1:nrow(bootstrap_data), replace = TRUE)
  bootstrap_model <- nnet::multinom(Species ~ ., data = bootstrap_data[bootstrap_sample, ], maxit = 10000)  # Increase max iterations
  bootstrap_pred <- predict(bootstrap_model, newdata = test_data, type = "class")
  bootstrap_scores[i] <- sum(bootstrap_pred == test_data$Species) / length(test_data$Species)
}

cat('Test Accuracy With Standard Train-Test Split:', round(accuracy_test, 2), '\n')
cat('Cross-Validation Mean Accuracy:', round(mean(cv_scores), 2), '\n')
cat('Cross-Validation SD Accuracy:', round(sd(cv_scores), 2), '\n')
cat('Bootstrapping Mean Accuracy:', round(mean(bootstrap_scores), 2), '\n')
cat('Bootstrapping SD Accuracy:', round(sd(bootstrap_scores), 2), '\n')
