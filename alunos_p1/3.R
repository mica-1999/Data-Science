# 3- load a dataset in CSV, manipulate the data, and save the resulting dataframe in CSV

# Set the working directory to the script's directory
setwd('Place the path to the file 3.R in here')

# Function to load and manipulate data
load_and_manipulate_data <- function(file_path) {
  # Load the dataset
  tryCatch({
    df <- read.csv(file_path)
    cat("Dataset loaded successfully.\n")

    # Manipulate the data to sort by column: BloodPressure

    # Save the manipulated data to a new CSV file

    cat("\nManipulated data saved to 'manipulated_diabetes.csv'.\n")

  }, error = function(e) {
    cat("An error occurred: ", conditionMessage(e), "\n")
  })
}

# Call the function to load and manipulate the data after the user specifies the path to the file
load_and_manipulate_data(file.choose())  # This will open a file dialog for the user to choose the CSV file
