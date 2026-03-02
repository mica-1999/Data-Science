# Set the working directory to the script's directory
setwd('Place the path to the file 5.R in here')

library('reticulate')

# Function to create a random array
create_random_array <- function(shape) {
  return(matrix(runif(prod(shape)), nrow = shape[1], ncol = shape[2]))
}

# Function to serialize and deserialize the array
serialize_and_deserialize <- function(array) {
  # Serialize the array

  # Deserialize the array
  
}

# Create a random array
random_array <- create_random_array(c(3, 3))

# Print the original array
cat("Original Array:\n")
print(random_array)

# Serialize and deserialize the array
serialize_and_deserialize(random_array)
