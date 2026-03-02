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
iris = sns.load_dataset('iris')  # carrega a dataset sample

# Set up the figure for histograms
features = iris.columns[:-1]  # todas as colunas numéricas
plt.figure(figsize=(12,8))    # define o tamanho da figura

for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)   # organiza 4 gráficos em 2 linhas x 2 colunas
    sns.histplot(iris[feature], kde=False, bins=15, color='skyblue')  # histograma
    plt.title(f'Histogram of {feature}')  # título de cada gráfico

plt.tight_layout()  # ajusta o espaçamento entre os subplots
plt.show()          # exibe os gráficos

# Set up the figure for violin plots
features = iris.columns[:-1]  # todas as colunas numéricas

for feature in features:
    plt.figure(figsize=(8,5))
    sns.violinplot(x='species', y=feature, data=iris, palette='Set2')  # distribuição por espécie
    plt.title(f'Violin Plot of {feature} by Species')  # título do gráfico
    plt.show()

# Set up the figure for kernel density estimate plots
features = iris.columns[:-1]  # todas as colunas numéricas

for feature in features:
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=iris, x=feature, hue='species', fill=True, palette='Set2')  # KDE por espécie
    plt.title(f'KDE Plot of {feature} by Species')  # título do gráfico
    plt.xlabel(feature)                             # nome do eixo X
    plt.ylabel('Density')                           # nome do eixo Y
    plt.show()

# Set up the figure for swarm plots
features = iris.columns[:-1]  # todas as colunas numéricas

for feature in features:
    plt.figure(figsize=(8,5))
    sns.swarmplot(x='species', y=feature, data=iris, palette='Set2')  # cada ponto representa uma amostra
    plt.title(f'Swarm Plot of {feature} by Species')  # título do gráfico
    plt.show()

#%% 3-Data Pre-processing
# TL;DR: Clean and prepare the dataset (handle missing values, outliers, and fill in data) before analysis or modeling
import pandas as pd
import numpy as np

# Read the penguins dataset
penguins = pd.read_csv('alunos_p2/penguins.csv')  # Substitua pelo caminho correto do seu CSV

# Display the original dataset
print("Original Dataset:")
print(penguins.head())  # mostra as primeiras 5 linhas
print("\nDataset Info:")
print(penguins.info())  # informações gerais, incluindo NaNs
print("\nSummary Statistics:")
print(penguins.describe())  # estatísticas básicas das colunas numéricas

#==============================
# Option 1: Delete rows with missing values
# TL;DR: Use if very few missing rows (<5%) and removing them won’t bias dataset
#==============================
penguins_dropna = penguins.dropna()  # remove todas as linhas que tiverem pelo menos um valor nulo
print("\nDataset after dropping rows with missing values:")
print(penguins_dropna.info())  # verifica o novo tamanho e valores nulos

#==============================
# Option 2: Interpolate missing values by species
# TL;DR: Use for continuous numeric data with a logical group (e.g., species), keeps trends
#==============================
cols_numeric = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Interpolação dentro de cada espécie
penguins[cols_numeric] = penguins.groupby('species')[cols_numeric].transform(lambda x: x.interpolate())

# Display dataset after interpolation
print("\nFirst 5 rows after interpolation:")
print(penguins.head())

print("\nSummary statistics after interpolation:")
print(penguins.describe())

#==============================
# Option 3: Remove outliers using IQR
# TL;DR: Use if extreme values are likely errors; keep real extreme measurements if valid
#==============================
for column_name in penguins.select_dtypes(include=[np.number]).columns:
    # Calculate the IQR (InterQuartile Range)
    Q1 = penguins[column_name].quantile(0.25)  # primeiro quartil
    Q3 = penguins[column_name].quantile(0.75)  # terceiro quartil
    IQR = Q3 - Q1

    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with NaN
    penguins[column_name] = np.where(
        (penguins[column_name] < lower_bound) | (penguins[column_name] > upper_bound),
        np.nan,
        penguins[column_name]
    )

# Display dataset after removing outliers
print("\nFirst 5 rows after handling outliers:")
print(penguins.head())

#==============================
# Option 4: Impute missing values using the mean
# TL;DR: Use if you want to keep all rows and missing values are random; median if skewed
#==============================
for column_name in penguins.select_dtypes(include=[np.number]).columns:
    penguins[column_name] = penguins[column_name].fillna(penguins[column_name].mean())

# Display dataset after imputing missing values
print("\nFirst 5 rows after imputing missing values with mean:")
print(penguins.head())

print("\nSummary statistics after imputing missing values:")
print(penguins.describe())  # confirma que não há mais NaNs

#%% 4- Feature Engineering - Categorical Encoding
# TL;DR: Transform text categories into numbers so the model can understand them (One-Hot or Label encoding)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # Converter categorias (texto) em números para que o modelo consiga processar
pd.set_option('display.max_columns', None)

# Load the dataset
penguins = pd.read_csv('alunos_p2/penguins.csv')

# Display the initial dataset
print(penguins.head())

# Create an instance of OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)  # Inicializa o codificador One-Hot

# Transform the 'species' column
species_encoded = onehot_encoder.fit_transform(penguins[['species']]) # Converte texto em números (One-Hot)

# Convert the result to a DataFrame and concatenate it with the original dataset
encoded_df = pd.DataFrame(
    species_encoded,
    columns=onehot_encoder.get_feature_names_out(['species'])
)
encoded_df.columns = [col.replace('species', 'encoded_species') for col in encoded_df.columns] # Para mudar o prefixo do nome da coluna
penguins = pd.concat([penguins, encoded_df], axis=1)  # Adiciona as novas colunas ao dataset

# Display the updated dataset
print(penguins.head())

# Create an instance of LabelEncoder
label_encoder = LabelEncoder() # Prepara o encode para transformar species em números inteiros

# Encode the 'species' column
penguins['species_label'] = label_encoder.fit_transform(penguins['species'])

# Display the updated dataset
print("Species -> Label mapping:")
for species, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{species} -> {label}")

#%% 5- Feature Engineering - Binning
# TL;DR: Everything in a specific range of numbers belongs to a category (bin)
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_sklearn = load_iris()
iris_df = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)

# Display the initial dataset
print(iris_df.head())

# Choose the 'sepal length (cm)' column for binning
feature_to_bin = 'sepal length (cm)'

# Define the number of bins (or bin edges)
bins = [4, 5, 6, 7, 8]

# Perform binning using pandas
iris_df['sepal_length_bin'] = pd.cut(iris_df[feature_to_bin], bins=bins, labels=['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4'])

# Display the dataset after binning
print(iris_df[['sepal length (cm)', 'sepal_length_bin']].head(10))


#%% 6- Feature Engineering - Interaction Features
# TL;DR: Use pandas to create new columns by combining existing ones, capturing relationships (interactions) between features
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_sklearn = load_iris()
iris_df = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)

# Display the initial dataset
print(iris_df.head())

# Simple combinations
iris_df['sepal_length_x_width'] = iris_df['sepal length (cm)'] * iris_df['sepal width (cm)']  # multiply
iris_df['petal_length_plus_width'] = iris_df['petal length (cm)'] + iris_df['petal width (cm)']  # sum
iris_df['sepal_length_div_petal_length'] = iris_df['sepal length (cm)'] / iris_df['petal length (cm)']  # divide

# Display the dataset after creating simple interactions
print(iris_df.head())

# Complex combination with a onlinear interaction
iris_df['non_linear_interaction'] = (
    iris_df['sepal length (cm)']**2 +
    iris_df['petal length (cm)'] * iris_df['petal width (cm)'] +
    iris_df['petal length (cm)'] * iris_df['sepal width (cm)']
)

# Display the dataset after creating interactions
print(iris_df.head())

#%% 7- Feature Engineering - Feature Scaling
# TL;DR: Rescale numerical features so models treat them equally:
# - StandardScaler → z-score scaling (mean=0, std=1)
# - MinMaxScaler → scales features to [0, 1] range
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the Iris dataset
iris_sklearn = load_iris()
iris_df = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)

# Display the initial dataset
print(iris_df.head())

# Apply Standardization
standard_scaler = StandardScaler()  # Initialize the scaler
iris_standard_scaled = standard_scaler.fit_transform(iris_df)  # Fit to data and transform

# Convert the standardized array back to a DataFrame for display
iris_standard_df = pd.DataFrame(
    iris_standard_scaled,
    columns=iris_df.columns
)

# Display the dataset after standardization
print(iris_standard_df.head())

# Apply Min-Max Scaling
minmax_scaler = MinMaxScaler()  # Initialize the scaler
iris_minmax_scaled = minmax_scaler.fit_transform(iris_df)  # Fit to data and transform

# Convert the min-max scaled array back to a DataFrame for display
iris_minmax_df = pd.DataFrame(
    iris_minmax_scaled,
    columns=iris_df.columns
)

# Display the dataset after min-max scaling
print(iris_minmax_df.head())

#%% 8- Feature Engineering - Data Balancing
# TL;DR: Garantir que cada classe tenha aproximadamente o mesmo número de amostras para evitar viés no modelo.
# - Dataset original → desbalanceado (mais benignos que malignos)
# - RandomOverSampler → duplica amostras da classe minoritária
# - SMOTE → gera amostras sintéticas para a classe minoritária
# - RandomUnderSampler → remove amostras da classe majoritária
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load the Breast Cancer Wisconsin (Diagnostic) dataset
cancer_sklearn = load_breast_cancer()
cancer_df = pd.DataFrame(data=cancer_sklearn.data, columns=cancer_sklearn.feature_names)
cancer_df['target'] = cancer_sklearn.target  # add target column

# Display the initial class distribution
sns.countplot(x='target', data=cancer_df)
plt.title('Initial Class Distribution')
plt.xlabel('Class (0 = malignant, 1 = benign)')
plt.ylabel('Number of samples')
plt.show()

# Standard Oversampling
ros = RandomOverSampler(random_state=42)  # initialize oversampler
X_resampled, y_resampled = ros.fit_resample(cancer_df.drop('target', axis=1), cancer_df['target'])
cancer_ros_df = pd.DataFrame(X_resampled, columns=cancer_df.drop('target', axis=1).columns)
cancer_ros_df['target'] = y_resampled

# Display the class distribution after oversampling
sns.countplot(x='target', data=cancer_ros_df)
plt.title('Class Distribution After Random Oversampling')
plt.xlabel('Class (0 = malignant, 1 = benign)')
plt.ylabel('Number of samples')
plt.show()

# Oveesample with SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(cancer_df.drop('target', axis=1), cancer_df['target'])
cancer_smote_df = pd.DataFrame(X_smote, columns=cancer_df.drop('target', axis=1).columns)
cancer_smote_df['target'] = y_smote

# Display the class distribution after SMOTE
sns.countplot(x='target', data=cancer_smote_df)
plt.title('Class Distribution After SMOTE Oversampling')
plt.xlabel('Class (0 = malignant, 1 = benign)')
plt.ylabel('Number of samples')
plt.show()

# Standard Undersampling
rus = RandomUnderSampler(random_state=42)  # initialize undersampler
X_under, y_under = rus.fit_resample(cancer_df.drop('target', axis=1), cancer_df['target'])
cancer_rus_df = pd.DataFrame(X_under, columns=cancer_df.drop('target', axis=1).columns)
cancer_rus_df['target'] = y_under

# Display the class distribution after undersampling
sns.countplot(x='target', data=cancer_rus_df)
plt.title('Class Distribution After Random Undersampling')
plt.xlabel('Class (0 = malignant, 1 = benign)')
plt.ylabel('Number of samples')
plt.show()

# Save the datasets after each transformation
cancer_df.to_csv('alunos_p2/cancer_original.csv', index=False)          # Original dataset
cancer_ros_df.to_csv('alunos_p2/cancer_oversampled.csv', index=False)  # Random Oversampling
cancer_smote_df.to_csv('alunos_p2/cancer_smote.csv', index=False)      # SMOTE Oversampling
cancer_rus_df.to_csv('alunos_p2/cancer_undersampled.csv', index=False) # Random Undersampling