#%% 1- perform web scraping
import requests
from bs4 import BeautifulSoup
from pprint import pprint

url = 'https://www.ayush.nz/technology'  # URL of the page to scrape
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(url, headers=headers)

if response.status_code != 200:
    print(f"Failed to retrieve page. Status code: {response.status_code}")
    exit()

html = BeautifulSoup(response.text, 'html.parser')
articles = html.select('div.article-link')  # Select article containers

my_data = []

for article in articles:
    try:
        # Extract title from <a> tag
        title_tag = article.find('a')
        title = title_tag.get_text(strip=True)
        url = title_tag['href']

        # Extract date from the muted span
        date = article.find('span', class_='muted').get_text(strip=True).replace('/', '').strip()

        # Extract excerpt (some articles have images in excerpts)
        excerpt_div = article.find('div', class_='excerpt')
        if excerpt_div:
            # Remove any images from excerpt
            for img in excerpt_div.find_all('img'):
                img.decompose()
            excerpt = excerpt_div.get_text(strip=True)
        else:
            excerpt = "No excerpt available"

        # Append the data to the list
        my_data.append({
            'title': title,
            'url': url,
            'date': date,
            'excerpt': excerpt
        })
        
    except Exception as e:
        print(f"Error processing article: {e}")
        continue

pprint(my_data)


#%% 2- use an API to retrieve data and store it locally
import requests
import numpy as np

# Define the API endpoint
api_url = "https://jsonplaceholder.typicode.com/posts"

# Example of the data:
#  {
#    "userId": 1,
#    "id": 1,
#    "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
#    "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"
#  },
#  {
#    "userId": 1,
#    "id": 2,
#    "title": "qui est esse",
#    "body": "est rerum tempore vitae\nsequi sint nihil reprehenderit dolor beatae ea dolores neque\nfugiat blanditiis voluptate porro vel nihil molestiae ut reiciendis\nqui aperiam non debitis possimus qui neque nisi nulla"
#  },
#  {
#    "userId": 1,
#    "id": 3,
#    "title": "ea molestias quasi exercitationem repellat qui ipsa sit aut",
#    "body": "et iusto sed quo iure\nvoluptatem occaecati omnis eligendi aut ad\nvoluptatem doloribus vel accusantium quis pariatur\nmolestiae porro eius odio et labore et velit aut"
#  }, ...

try:
    # Send a GET request to the API
    response = requests.get(api_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Extract the userIds
        user_ids = [post['userId'] for post in data]

        # Convert the list to a NumPy array


        # Specify the file path where you want to save the CSV file


        # Save the NumPy array as a CSV file


        # Print the array


        # Calculate the mean of user IDs


    else:
        print("Error: Unable to fetch data from the API. Status code:", response.status_code)

except Exception as e:
    print("An error occurred:", str(e))

#%% 3- load a dataset in CSV, manipulate the data, and save the resulting dataframe in CSV
import pandas as pd

def load_and_manipulate_data(file_path):
    # Load the dataset
    try:
        # Load the dataframe


        # Manipulate the data to sort by column: BloodPressure


        # Save the manipulated data to a new CSV file


    except FileNotFoundError:
        print(f"Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to load and manipulate the data after the user specify the path to the file
load_and_manipulate_data(input("Enter the path to the CSV file: ")) # Since the file is in the same folder as the script just type: diabetes.csv

#%% 4- create, insert and print tables using SQL

import sqlite3

# Function to create tables
def create_tables(conn):
    cursor = conn.cursor()

    # Create the "students" table
    cursor.execute('''
        CREATE TABLE students (
            student_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER
        )
    ''')

    # Create the "grades" table with a foreign key reference to students
    cursor.execute('''
        CREATE TABLE grades (
            grade_id INTEGER PRIMARY KEY,
            subject TEXT NOT NULL,
            grade INTEGER,
            student_id INTEGER,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')

    conn.commit()

# Function to insert data into tables
def insert_data(conn):
    cursor = conn.cursor()

    # Insert data into the "students" table


    # Insert data into the "grades" table


    # Commit the data insertion
    conn.commit()

# Function to print the contents of tables
def print_tables(conn):
    cursor = conn.cursor()

    # Print the "students" table
    cursor.execute("SELECT * FROM students")
    print("\nStudents Table:")
    print(cursor.fetchall())

    # Print the "grades" table


# Connect to the SQLite database (or create a new one if not exists)
db_file_path = "school_database.db"
conn = sqlite3.connect(db_file_path)

# Create tables
create_tables(conn)

# Insert data
insert_data(conn)

# Print tables
print_tables(conn)

# Close the database file
conn.close()

#%% 5- serialize and deserialize a randomly created array

import numpy as np
import pickle

# Function to create a random NumPy array
def create_random_array(shape):


# Function to serialize and deserialize the NumPy array using pickle
def serialize_and_deserialize(array):
    # Serialize the array
    with open('serialized_array.pkl', 'wb') as file:


    # Deserialize the array and print it
    with open('serialized_array.pkl', 'rb') as file:


# Create a random NumPy array
random_array = create_random_array((3, 3))

# Print the original array
print("Original Array:")
print(random_array)

# Serialize and deserialize the array
serialize_and_deserialize(random_array)