# 4- create, insert, and print tables using SQL

# Set the working directory to the script's directory
setwd('Place the path to the file 4.R in here')

# Install and load the RSQLite package
if (!requireNamespace("RSQLite", quietly = TRUE)) {
  install.packages("RSQLite")
}
library(RSQLite)

# Function to create tables
create_tables <- function(conn) {
  # Execute the queries
  
}

# Function to insert data into tables
insert_data <- function(conn) {
  # Execute the queries

}

# Function to print the contents of tables
print_tables <- function(conn) {
  # Use dbGetQuery to do the SELECT


  # Print the tables

}

# Connect to the SQLite database (or create a new one if not exists)
db_file_path <- "school_database.db"
conn <- dbConnect(SQLite(), dbname = db_file_path)

# Create tables
create_tables(conn)

# Insert data
insert_data(conn)

# Print tables
print_tables(conn)

# Close the database connection
dbDisconnect(conn)
