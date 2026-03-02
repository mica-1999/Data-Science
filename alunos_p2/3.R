# Set the working directory to the script's directory
setwd('Place the path to the file 3.R in here')

# Read the penguins dataset
penguins <- read.csv('penguins.csv')  # Replace with your actual file path

# Display the original dataset
cat("Original Dataset:\n")
print(penguins)

# Dealing with missing values
# Option 1: Delete missing values
penguins_deleted_na <- na.omit(penguins)

# Option 2: Interpolate missing values (linear interpolation within each class, assuming 'island' is a class identifier)
penguins_interpolated <- penguins
numeric_columns <- sapply(penguins, is.numeric)
for (col in names(penguins)[numeric_columns]) {
  penguins_interpolated[[col]] <- ave(penguins[[col]], penguins$island, FUN = function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
}

# Display datasets after handling missing values
cat("\nDataset after deleting missing values:\n")
print(penguins_deleted_na)

cat("\nDataset after interpolating missing values:\n")
print(penguins_interpolated)

# Dealing with outliers for all numerical columns
for (col in names(penguins)[numeric_columns]) {
  # Calculate the IQR (InterQuartile Range)


  # Define the lower and upper bounds to identify outliers


  # Remove outliers
  
}

# Display dataset after handling outliers for all numerical columns


# Impute missing values using the mean value of the column
penguins_imputed <- penguins
numeric_columns_to_impute <- sapply(penguins_imputed, is.numeric)
penguins_imputed[, numeric_columns_to_impute] <- lapply(penguins_imputed[, numeric_columns_to_impute], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Display dataset after imputing missing values

