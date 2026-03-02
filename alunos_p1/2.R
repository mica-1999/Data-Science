# 2- use an API to retrieve data and store it locally

# Set the working directory to the script's directory
setwd('Place the path to the file 2.R in here')

# Install and load necessary libraries
install.packages(c("httr", "jsonlite"))
library(httr)
library(jsonlite)

# Specify the API URL
api_url <- "https://jsonplaceholder.typicode.com/posts"

tryCatch({
  # Send a GET request to the API
  response <- GET(api_url)

  # Check if the request was successful (status code 200)
  if (status_code(response) == 200) {
    # Parse the JSON response
    data <- content(response, "text", encoding = "UTF-8")
    data <- fromJSON(data)

    # Extract the userIds column


    # Specify the file path where you want to save the CSV file


    # Save the user IDs as a CSV file
 

    # Print the array


    # Calculate the mean of user IDs


  } else {
    cat("Error: Unable to fetch data from the API. Status code:", status_code(response), "\n")
  }

}, error = function(e) {
  cat("An error occurred:", conditionMessage(e), "\n")
})
