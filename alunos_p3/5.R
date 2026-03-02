# Set the working directory to the script's directory
setwd('Place the path to the file 5.R in here')

# Install and load required libraries
install.packages("dplyr")
install.packages("tidyr")

library(dplyr)
library(tidyr)

# Load the dataset from the CSV file
df <- read.csv("california_housing.csv")

# Add a column for categorical average rooms
df$AveRoomsCategory <- cut(df$AveRooms, breaks=c(0, 3, 6, 9, Inf), labels=c('0-3', '3-6', '6-9', '9+'))

# Perform ANOVA using oneway.test


# Print ANOVA results (F-Statistic and P-Value) and check if is significant

