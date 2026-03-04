# Phase 1: Problem Formulation
# Dataset: [Flight Delay and Cancellation]
# Source: U.S. Department of Transportation On-Time Performance Reporting System
# Cada linha corresponde a um voo comercial programado nos EUA, com informações sobre tal

# Problema:
# Prever e analisar atrasos em voos domésticos nos EUA.
# O foco principal é compreender os fatores que influenciam os atrasos na chegada e usá-los para prever atrasos antes da partida.
# Além disso, identificar padrões no desempenho das companhias aéreas e dos aeroportos.
#
# Objetivos: (P-03)
# 1. Regression Task: Predict arrival delay duration (in minutes) for a given flight based on
# scheduled times, airline, route, distance, and operational factors
# 2. Classification task: Categorize flights into three classes:
#    - On-time (<15-min delay)
#    - Short delay (15–30 min)
#    - Long delay (>30 min)
# 3. Clustering Task: Identify patterns in operational performance for airports
#    and airlines based on delay behavior and route profiles.
# 4. Hypothesis testing Task:
#    - Whether certain airlines are systematically delayed.
#    - Whether weather has a significant impact on flight delays.

# Notas:
#   - Utilizar apenas as funcionalidades conhecidas antes da partida para evitar vazamento de dados. (P-02)
#   - Excluir ou tratar os voos cancelados/desviados com cuidado. (P-02)
#   - As colunas derivadas de atrasos reais (ARR_DELAY, DEP_DELAY, DELAY_DUE_*) não devem ser usadas.

# Phase 2: Data Analysis and Clensing
#%% 1- Pre-processing, Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Loading CSV
df = pd.read_csv('flights_sample_3m.csv')
df_eda = df.copy()  # Pre pre-processing, guarda o original para a parte EDA
#print(df.head()) # Checking if it loaded, primeiras linhas
#print(df.info()) # Info sobre as colunas e tipos de valores (string, int, etc..)
#print(df.describe()) # Dados estatísticos sobre as colunas

# Remover linhas desnecessárias ou que contêm null em colunas importantes
#print(df.isnull().sum()) # Verificando o nº de nulls nas colunas por linha
#print("New dataset shape:", df.shape) # linhas x colunas nº
df.dropna(subset=['CRS_ELAPSED_TIME'], inplace=True) # Apaga as linhas onde "CRS_ELAPSED_TIME" é null
df = df[df['CANCELLED'] == 0] # Apenas removemos as linhas dos voos que foram cancelados, podemos agora remover a coluna

# Dataset após apagar linhas
print("\nDataset after dropping rows with missing CRS_ELAPSED_TIME:")
print(df.isnull().sum())

# Remover colunas desnecessárias, estas colunas são dados do futuro e não ajudam a prever
cols_to_drop = [
    'ARR_DELAY', 'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
    'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
    'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
    'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME','CANCELLED','CANCELLATION_CODE'
]
df.drop(columns=cols_to_drop, inplace=True)
print("New dataset shape:", df.shape) # Verificando se foram apagadas
print(df.isnull().sum()) # Verificar se o df ficou limpo.
print(list(df.columns))

# Remover outliers using IQR
numeric_cols = ['CRS_ELAPSED_TIME', 'DISTANCE'] # Colunas que fazem sentido, NOTA: PERGUNTAR ao prof se DISTANCE é bom de remover outliers
#print(df[numeric_cols].describe()) # Before handling
for column_name in numeric_cols:
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with NaN
    df[column_name] = np.where(
        (df[column_name] < lower_bound) | (df[column_name] > upper_bound),
        np.nan,
        df[column_name]
    )

# Preenchendo os nans com valores medianos
for column_name in numeric_cols:
    df[column_name] = df[column_name].fillna(df[column_name].median())

#print(df[numeric_cols].describe()) # After handling

# Featuring Engineering : Categorical Encoding
categorical_cols = ['AIRLINE_CODE', 'ORIGIN', 'DEST'] # Unicos que fazem sentido dividir em categorias para o modelo
# Separa-se o OneHotEncoded e o Label Encode
print("Number of unique airlines:", df['AIRLINE_CODE'].nunique())
print(f"Number of unique origin airports:", df['ORIGIN'].nunique())
print(f"Number of unique destination airports:", df['DEST'].nunique())

# HotEncoder para o AIRLINE_CODE pois contém poucos valores únicos
onehot_encoder = OneHotEncoder(sparse_output=False)
airline_encoded = onehot_encoder.fit_transform(df[['AIRLINE_CODE']])

# Converte para df e adiciona ao df 'limpo'
encoded_airline_df = pd.DataFrame(
    airline_encoded,
    columns=onehot_encoder.get_feature_names_out(['AIRLINE_CODE'])
)
encoded_airline_df.columns = [col.replace('AIRLINE_CODE', 'encoded_airline') for col in encoded_airline_df.columns]
df = pd.concat([df, encoded_airline_df], axis=1)

# LabelEncoder para o ORIGIN e DEST pois contêm muitos valores unicos
label_encoder = LabelEncoder()
df['ORIGIN_label'] = label_encoder.fit_transform(df['ORIGIN'])
df['DEST_label'] = label_encoder.fit_transform(df['DEST'])









