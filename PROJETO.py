#%% 1- Phase 1: Problem Formulation
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

#%% 2- Phase 2: Data Analysis and Clensing / Pre-processing
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Loading CSV
df = pd.read_csv('flights_sample_3m.csv')
print(list(df.columns))
df_eda = df.copy()  # Pre pre-processing, guarda o original para a parte EDA
#print(df.head()) # Checking if it loaded, primeiras linhas
#print(df.info()) # Info sobre as colunas e tipos de valores (string, int, etc..)
#print(df.describe()) # Dados estatísticos sobre as colunas

# Remover linhas desnecessárias ou que contêm null em colunas importantes
#print(df.isnull().sum()) # Verificando o nº de nulls nas colunas por linha
#print("Dataset shape:", df.shape) # linhas x colunas nº
df.dropna(subset=['CRS_ELAPSED_TIME'], inplace=True) # Apaga as linhas onde "CRS_ELAPSED_TIME" é null
df = df[df['CANCELLED'] == 0] # Apenas removemos as linhas dos voos cancelados, podemos agora remover a coluna

# Dataset após apagar linhas
#print("\nDataset after dropping rows with missing CRS_ELAPSED_TIME:")
#print(df.isnull().sum())

# Remover colunas desnecessárias, estas colunas são dados do futuro e não ajudam a prever
cols_to_drop = [
    'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
    'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
    'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
    'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME','CANCELLED','CANCELLATION_CODE'
]
df.drop(columns=cols_to_drop, inplace=True)
#print("New dataset shape:", df.shape) # Verificando se foram apagadas
print(df.isnull().sum()) # Verificar se o df ficou limpo.
# print(list(df.columns))

# Remover outliers usando IQR
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
df = df.reset_index(drop=True) # Faz reset do index para resolver as linhas saltadas

# Categorical Encoding
categorical_cols = ['AIRLINE_CODE', 'ORIGIN', 'DEST'] # Unicos que fazem sentido dividir em categorias para o modelo
# Separa-se o OneHotEncoded e o Label Encode (mais valores e menos valores)
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
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df['ORIGIN_label'] = le_origin.fit_transform(df['ORIGIN'])
df['DEST_label'] = le_dest.fit_transform(df['DEST'])

# Apagando as colunas para o modelo e verificando se tudo ok
df.drop(columns=['AIRLINE_CODE', 'ORIGIN', 'DEST'], inplace=True)
#print("New dataset shape:", df.shape)
#print(list(df.columns)) # Verifica-se se as colunas encoded foram adicionadas

# Standarization/Normalization
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df[numeric_cols])
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df[numeric_cols])

# Convert to DataFrame and add suffix "_std"
df_standard_scaled = pd.DataFrame(
    standard_scaled,
    columns=[f"{col}_std" for col in numeric_cols]
)
# Convert to DataFrame and add suffix "_minmax"
df_minmax_scaled = pd.DataFrame(
    minmax_scaled,
    columns=[f"{col}_minmax" for col in numeric_cols]
)
# Concatenate scaled columns with original DataFrame
df = pd.concat([df, df_standard_scaled, df_minmax_scaled], axis=1)
df.to_csv('flights_cleaned_scaled.csv', index=False)
print(f"Processing done")

#%% 3- Phase 2: Data Analysis and Cleansing / Exploratory Data Analysis (EDA)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Filter FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Numeric columns for analysis
numeric_cols = ['CRS_ELAPSED_TIME', 'DISTANCE', 'ARR_DELAY']

# Summary statistics with pandas
print("\nBasic statistics:\n", df_eda[numeric_cols].describe())
print("\nMedian values:\n", df_eda[numeric_cols].median())
print("\nMean values:\n", df_eda[numeric_cols].mean())
print("\nStandard deviation:\n", df_eda[numeric_cols].std())

# Data distribution analysis
# Histograms
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df_eda[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplots por companhia
for col in ['ARR_DELAY']:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='AIRLINE', y=col, data=df_eda)
    plt.xticks(rotation=45)
    plt.title(f"{col} by Airline")
    plt.show()

# Correlation matrix
corr_matrix = df_eda[numeric_cols].corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# Heatmap of correlations
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# KDE Plots (density per airline)
for col in ['ARR_DELAY']:
    plt.figure(figsize=(12,6))
    sns.kdeplot(data=df_eda, x=col, hue='AIRLINE', fill=True, alpha=0.5)
    plt.title(f"KDE Plot of {col} by Airline")
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()

# Scatter Plots ARR_DELAY VS DISTANCE E ARR_DELAY VS SCHEDULE_DURATION, AQUI ENCONTRA-SE CORRELAÇÃO DISTANCE-SCHEDULE
plt.figure(figsize=(8,6))
sns.scatterplot(x='DISTANCE', y='ARR_DELAY', data=df_eda, alpha=0.3)
plt.title("Scatter Plot: Distance vs Arrival Delay")
plt.xlabel("Distance (miles)")
plt.ylabel("Arrival Delay (minutes)")
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(x='CRS_ELAPSED_TIME', y='ARR_DELAY', data=df_eda, alpha=0.3)
plt.title("Scatter Plot: Scheduled Duration vs Arrival Delay")
plt.xlabel("Scheduled Duration (minutes)")
plt.ylabel("Arrival Delay (minutes)")
plt.show()

print(f"EDA with original dataset done")

#%% 4a - Phase 2: Dimensionality Reduction / PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Pegando nas colunas númericas e encoded e ignora o resto
features_for_dr = df.drop(
    columns=[
        'ARR_DELAY', 'AIRLINE', 'ORIGIN_CITY', 'DEST_CITY',
        'FL_DATE', 'FL_NUMBER', 'DOT_CODE', 'AIRLINE_DOT',
        'CRS_ELAPSED_TIME_minmax', 'DISTANCE_minmax'
    ],
    errors='ignore'
)
print(features_for_dr.isnull().sum()) # confirmando que não existe valores nulos antes de PCA ou UMAP

# PCA (Principal Component Analysis)
pca = PCA(n_components=2)  # reduzindo as novas colunas  e conjunto de dados para 2 dimensões
pca_result = pca.fit_transform(features_for_dr)

df_pca = pd.DataFrame(
    pca_result,
    columns=['PC1', 'PC2']
)
# adicionando as companhias para comparação
df_pca['AIRLINE'] = df['AIRLINE'].values

plt.figure(figsize=(14,6))
sns.scatterplot(
    data=df_pca,
    x='PC1',
    y='PC2',
    hue='AIRLINE',
    palette='tab20',
    alpha=0.5
)

plt.title("PCA Projection of Flights")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

#%% 4b - Phase 2: Dimensionality Reduction / UMAP
import umap
import matplotlib.pyplot as plt
import seaborn as sns

umap_reducer = umap.UMAP(
    n_components=2,     # project down to 2D
    n_neighbors=50,     # controls local vs global structure
    min_dist=0.1,       # how tightly points are clustered
    random_state=42
)

# Fit and transform
print("Fitting")
umap_result = umap_reducer.fit_transform(features_for_dr) # pode demorar.

# Create DataFrame for plotting
df_umap = pd.DataFrame(
    umap_result,
    columns=['UMAP1', 'UMAP2']
)
df_umap['AIRLINE'] = df['AIRLINE'].values


# Plot UMAP projection
print("Plotting")
plt.figure(figsize=(14,6))
sns.scatterplot(
    data=df_umap.sample(200000, random_state=42),
    x='UMAP1',
    y='UMAP2',
    hue='AIRLINE',
    palette='tab20',
    alpha=0.5
)
plt.title("UMAP Projection of Flights")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.show()

#%% 5- Phase 2: Data Analysis and Cleansing /  Insights from EDA

# Tempo de voo (CRS_ELAPSED_TIME):
# - Média: ~142 min (~2h 22min)
# - Desvio padrão: 71 min → grande variabilidade; há voos muito curtos e muito longos
# - Mínimo/Máximo: 1 min → 705 min
#     * 1 min provavelmente é erro ou outlier
#     * 705 min pode ser válido, mas incomum se houver muitos voos desse tipo
# - Percentis 25/75: 90 / 172 → a maioria dos voos dura entre 1.5 e 3 horas
#
# Atraso na chegada (ARR_DELAY):
# - Mediana: -7 min < Média: ~4 min
#     * A maioria dos voos chega um pouco mais cedo
#     * Alguns atrasos grandes aumentam a média
#
# Correlações importantes:
# - CRS_ELAPSED_TIME ↔ DISTANCE: 0.98 → voos mais longos (maior distância) levam mais tempo
# - ARR_DELAY ↔ CRS_ELAPSED_TIME: -0.002
# - ARR_DELAY ↔ DISTANCE: 0.002
#     * Atrasos não dependem diretamente da distância ou duração do voo
#
# Histogramas:
# - Elapsed Time: confirma a distribuição indicada pelos quartis
# - Distance: a maioria dos voos cobre menos de 1.000 milhas, frequência diminui em rotas mais longas
# - ARR_DELAY: confirma a tendência de a maioria dos voos chegar no horário ou adiantados

#%% 6- Phase 2:  Hypothesis Testing
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, f_oneway

df_hyp = df.copy()
df_hyp = df_hyp.dropna(subset=['ARR_DELAY']) # remove as linhas que têm nans, no preprocessing não havia problem manter
print(df_hyp.isnull().sum()) # Verificando se foram removidas
airline_counts = df_hyp['AIRLINE'].value_counts()
top_airlines = airline_counts.index[:2]
print("Using these airlines for t-test:", top_airlines)

# 1️⃣ Hypothesis 1: Correlation between flight distance and arrival delay
print("Hypothesis 1: Distance vs Arrival Delay")
corr_coeff, p_value = pearsonr(df_hyp['DISTANCE'], df_hyp['ARR_DELAY'])
print(f"Pearson correlation coefficient: {corr_coeff:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("✅ Significant correlation: flight distance is associated with delays.\n")
else:
    print("❌ No significant correlation between distance and delay.\n")

# 2️⃣ Hypothesis 2: Airline A vs Airline B mean arrival delays (t-test)
airline_a = df_hyp['ARR_DELAY'][df_hyp['AIRLINE'] == 'Southwest Airlines Co.']
airline_b = df_hyp['ARR_DELAY'][df_hyp['AIRLINE'] == 'Delta Air Lines Inc.']

print("Hypothesis 2: Southwest Airlines Co. vs Delta Air Lines Inc. mean arrival delays") # muda airline para diferentes e verificar.
t_stat, p_value = ttest_ind(airline_a, airline_b)
print(f"T-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("✅ Significant difference in mean delays between AA and DL.\n")
else:
    print("❌ No significant difference in mean delays between AA and DL.\n")

# 3️⃣ Hypothesis 3: All airlines have the same mean arrival delay (ANOVA)
groups = [df_hyp['ARR_DELAY'][df_hyp['AIRLINE'] == airline] for airline in df_hyp['AIRLINE'].unique()]
f_stat, p_value = f_oneway(*groups)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("✅ Significant differences exist in delays between airlines.\n")
else:
    print("❌ No significant differences in delays between airlines.\n")
