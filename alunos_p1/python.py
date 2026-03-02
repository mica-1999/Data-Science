# %% 1- perform web scraping
import requests  # Usada para fazer pedidos HTTP (GET) para obter HTML de páginas
from bs4 import BeautifulSoup  # Converte HTML bruto numa estrutura pesquisável (árvore DOM)
from pprint import pprint  # Para print das listas/dicionários de forma organizada

url = 'https://www.ayush.nz/technology'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Envia o pedido GET ao servidor e guarda a resposta
response = requests.get(url, headers=headers)

# Verifica se o pedido foi bem sucedido
if response.status_code != 200:
    print(f"Failed to retrieve page. Status code: {response.status_code}")
    exit()

# Converte o HTML recebido numa estrutura navegável
html = BeautifulSoup(response.text, 'html.parser')
articles = html.select('div.article-link') # Seleciona todos os <div> com classe 'article-link' (cada artigo)

my_data = []

# Loop por cada artigo encontrado
for article in articles:
    try:
        # Extrair o título do <a> dentro do artigo
        title_tag = article.find('a')
        title = title_tag.get_text(strip=True)  # Texto limpo, sem espaços extra
        url = title_tag['href']  # Link do artigo

        # Extrair a data do <span class="muted">
        date = article.find('span', class_='muted').get_text(strip=True).replace('/', '').strip()

        # Extrair o resumo (alguns artigos têm imagens)
        excerpt_div = article.find('div', class_='excerpt')
        if excerpt_div:
            # Remove qualquer <img> dentro do resumo
            for img in excerpt_div.find_all('img'):
                img.decompose()
            excerpt = excerpt_div.get_text(strip=True)  # Texto limpo
        else:
            excerpt = "No excerpt available"  # Caso não exista resumo

        # Adiciona os dados do artigo à lista
        my_data.append({
            'title': title,
            'url': url,
            'date': date,
            'excerpt': excerpt
        })

    except Exception as e:
        print(f"Error processing article: {e}")
        continue

# Mostra todos os artigos extraídos de forma legível
pprint(my_data)


#%% 2- use an API to retrieve data and store it locally
import requests
import numpy as np # Para armazenar dados em arrays e manipular facilmente

# Define the API endpoint
api_url = "https://jsonplaceholder.typicode.com/posts"

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
        user_ids_array = np.array(user_ids)

        # Specify the file path where you want to save the CSV file
        csv_file_path = "user_ids.csv"

        # Save the NumPy array as a CSV file
        np.savetxt(csv_file_path, user_ids_array, delimiter=",", fmt="%d")

        # Print the array
        print(user_ids_array)

        # Calculate the mean of user IDs
        mean_user_id = np.mean(user_ids_array)
        print("Mean of user IDs:", mean_user_id)

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