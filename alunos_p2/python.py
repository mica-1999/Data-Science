#%% 1- Exploratory Data Analysis (EDA)
import numpy as np
import seaborn as sns  # visualizações estatísticas
import matplotlib.pyplot as plt # criação de gráficos básicos
import warnings # controlar mensagens de aviso

# Filter FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display basic information about the dataset
print(iris.info())

# Display the first few rows of the dataset
print(iris.head())  # mostra as primeiras 5 linhas do dataset

# Summary statistics (using the Pandas dataframe)
numeric_cols = iris.select_dtypes(include=np.number)   # seleciona apenas colunas numéricas

print("Basic statistics:\n", numeric_cols.describe())          # estatísticas básicas: count, mean, std, min, quartis e max
print("\nMedian values:\n", numeric_cols.median())            # mediana apenas das colunas numéricas
print("\nMean values:\n", numeric_cols.mean())                # média apenas das colunas numéricas
print("\nStandard deviation:\n", numeric_cols.std())          # desvio padrão apenas das colunas numéricas

# Convert to a numpy array without using the last columns (is the label)
iris_values = iris.iloc[:, :-1].values  # pega todas as linhas e todas as colunas exceto a última

# Summary statistics with NumPy
print("\nSummary Statistics with NumPy:")
# Mean
mean_values = np.mean(iris_values, axis=0)  # calcula a média de cada coluna
print("Mean values:", mean_values)

# Median
median_values = np.median(iris_values, axis=0)  # calcula a mediana de cada coluna
print("Median values:", median_values)

# Standard Deviation
std_values = np.std(iris_values, axis=0, ddof=1)  # calcula o desvio padrão de cada coluna (ddof=1 para amostra)
print("Standard Deviation:", std_values)

# Data distribution analysis
# Pairplot for overall distribution
sns.pairplot(iris, hue='species')
plt.show()

# Boxplot for each feature by species
features = iris.columns[:-1]  # todas as colunas numéricas

for feature in features:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='species', y=feature, data=iris, palette='Set2')  # cores diferentes por espécie
    plt.title(f"Boxplot of {feature} by Species")   # título do gráfico
    plt.show()

# Correlation Analysis
corr_matrix = iris.iloc[:, :-1].corr()  # calcula a correlação entre as colunas numéricas
print(corr_matrix)                       # imprime a matriz de correlação

#%% 2- Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Load the Iris dataset

# Set up the figure for histograms


# Set up the figure for violin plots


# Set up the figure for kernel density estimate plots


# Set up the figure for swarm plots


#%% 3-Data Pre-processing

import pandas as pd
import numpy as np

# Read the penguins dataset
penguins = pd.read_csv('penguins.csv')  # Replace with your actual file path

# Display the original dataset
print("Original Dataset:")
print(penguins)

# Dealing with missing values
# Option 1: Delete missing values


# Option 2: Interpolate missing values (only valid if done withing samples of the same class to be tested, for example, island)


# Display datasets after handling missing values


# Dealing with outliers for all numerical columns
for column_name in penguins.select_dtypes(include=[np.number]).columns:
    # Calculate the IQR (InterQuartile Range)


    # Define the lower and upper bounds to identify outliers


    # Remove outliers
    penguins[column_name] = np.where(
        (penguins[column_name] < lower_bound) | (penguins[column_name] > upper_bound),
        np.nan,
        penguins[column_name]
    )

# Display dataset after handling outliers for all numerical columns


# Impute missing values using the mean value of the column


# Display dataset after imputing missing values


#%% 4- Feature Engineering - Categorical Encoding

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset


# Display the initial dataset


# Create an instance of OneHotEncoder
onehot_encoder = OneHotEncoder()

# Transform the 'species' column
species_encoded = onehot_encoder.fit_transform(penguins[['species']])

# Convert the result to a DataFrame and concatenate it with the original dataset


# Display the updated dataset


# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'species' column


# Display the updated dataset


# Note, check the dataframe in the variable in debugging to observre the added columns

#%% 5- Feature Engineering - Binning

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset


# Display the initial dataset


# Choose the 'sepal length (cm)' column for binning


# Define the number of bins (or bin edges)
bins = [4, 5, 6, 7, 8]

# Perform binning using pandas
iris_df['sepal_length_bin'] = pd.cut(iris_df[feature_to_bin], bins=bins, labels=['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4'])

# Display the dataset after binning


#%% 6- Feature Engineering - Interaction Features

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset


# Display the initial dataset


# Simple combinations


# Display the dataset after creating simple interactions


# Complex combination with a onlinear interaction


# Display the dataset after creating interactions


# Note, check the dataframe in the variable in debugging to observre the added columns

#%% 7- Feature Engineering - Feature Scaling

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the Iris dataset


# Display the initial dataset

# Apply Standardization


# Convert the standardized array back to a DataFrame for display


# Display the dataset after standardization

# Apply Min-Max Scaling


# Convert the min-max scaled array back to a DataFrame for display


# Display the dataset after min-max scaling


#%% 8- Feature Engineering - Data Balancing

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load the Breast Cancer Wisconsin (Diagnostic) dataset


# Display the initial class distribution


# Standard Oversampling

# Display the class distribution after oversampling


# Oveesample with SMOTE


# Display the class distribution after SMOTE


# Standard Undersampling


# Display the class distribution after undersampling


# Save the datasets after each transformation
