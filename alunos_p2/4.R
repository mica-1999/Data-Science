# Set the working directory to the script's directory
setwd('Place the path to the file 4.R in here')


library(Matrix)

# Load the dataset
penguins <- read.csv('penguins.csv')

# Display the initial dataset


# One-hot encode the 'species' column
species_encoded <- model.matrix(~ species - 1, data = penguins)

# Convert the result to a DataFrame and concatenate it with the original dataset
species_encoded_df <- as.data.frame(species_encoded)
colnames(species_encoded_df) <- gsub("species", "", colnames(species_encoded_df))

penguins_encoded <- cbind(penguins, species_encoded_df)

# Display the updated dataset
cat("Encoded dataset with One-Hot Encoding:\n")
print(penguins_encoded)

# Label encode the 'species' column


# Display the updated dataset
