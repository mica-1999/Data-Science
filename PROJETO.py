#%% 1- Phase 2: Data Analysis and Cleansing / Pre-processing
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Loading CSV
df = pd.read_csv('datasets/flights_sample_3m.csv')
df_eda = df.copy()  # Pre pre-processing, guarda o original para a parte EDA

# Prints
print("Starting pre-processing... ")
#print(df.head()) # Checking if it loaded
#print(df.info()) # Info sobre as colunas e tipos de valores (string, int, etc..)
#print(df.describe()) # Dados estatísticos sobre as colunas
#print(df.isnull().sum()) # Verificando o nº de nulls nas colunas por linha
#print("Dataset shape:", df.shape) # linhas x colunas nº
print(list(df.columns)) # Lista de colunas

# Remover linhas desnecessárias/contêm null em colunas importantes
df.dropna(subset=['CRS_ELAPSED_TIME'], inplace=True) # Apaga as linhas onde "CRS_ELAPSED_TIME" é null
df = df[df['CANCELLED'] == 0] # Apenas removemos as linhas dos voos cancelados, podemos agora remover a coluna
df = df[df['DIVERTED'] == 0] # Apenas removemos as linhas dos voos não rotados, podemos agora remover a coluna
df = df.dropna(subset=['ARR_DELAY']) # Removemos as 2 linhas que tinham null em ARR_DELAY

df_hyp = df.copy() # Guardando para hyp

# Remover colunas desnecessárias, estas colunas são dados do futuro e não ajudam a prever
cols_to_drop = [
    'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
    'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
    'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
    'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME','CANCELLED','CANCELLATION_CODE','DIVERTED','AIRLINE',
    'AIRLINE_DOT','DOT_CODE','FL_NUMBER','ORIGIN_CITY','DEST_CITY','CRS_ARR_TIME'
]
df.drop(columns=cols_to_drop, inplace=True)
#print("New dataset shape:", df.shape) # Verificando se foram apagadas
print(df.isnull().sum()) # Verificar se o df ficou limpo.

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
#print(df[numeric_cols].describe()) # Verificando se os max min ficaram resolvidos

#print(df[numeric_cols].describe()) # After handling
df = df.reset_index(drop=True) # Faz reset do index para resolver as linhas saltadas (quando foram removidas)

# Guardando dataset pre-scaled/encoded só em caso
df_cleaned = df.copy()
#df.to_csv('datasets/flights_cleaned.csv', index=False)

print(f"Processing done")

#%% 2- Phase 2: Data Analysis and Cleansing / Exploratory Data Analysis (EDA)
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
plt.figure(figsize=(12,6))
sns.countplot(y='AIRLINE', data=df_eda, order=df_eda['AIRLINE'].value_counts().index[:10])
plt.title("Top 10 Airlines by Number of Flights")
plt.xlabel("Count")
plt.ylabel("Airline")
plt.savefig("Outputs/top10_airlines_count.png", bbox_inches='tight')
plt.close()

# Histograms
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df_eda[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"Outputs/hist_{col}.png", bbox_inches='tight')
    plt.close()

# Box plots por companhia
for col in ['ARR_DELAY']:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='AIRLINE', y=col, data=df_eda)
    plt.xticks(rotation=45)
    plt.savefig(f"Outputs/boxplot_{col}_by_airline.png", bbox_inches='tight')
    plt.close()

# Correlation matrix & Heatmap
corr_matrix = df_eda[numeric_cols].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("Outputs/corr_heatmap.png", bbox_inches='tight')
plt.close()

# KDE Plots (density per airline)
for col in ['ARR_DELAY']:
    plt.figure(figsize=(12,6))
    sns.kdeplot(data=df_eda, x=col, hue='AIRLINE', fill=True, alpha=0.5)
    plt.title(f"KDE Plot of {col} by Airline")
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.savefig(f"Outputs/kde_{col}_by_airline.png", bbox_inches='tight')
    plt.close()

# Scatter Plots ARR_DELAY VS DISTANCE E ARR_DELAY VS SCHEDULE_DURATION, CORRELAÇÃO DISTANCE-SCHEDULE
plt.figure(figsize=(8,6))
sns.scatterplot(x='DISTANCE', y='ARR_DELAY', data=df_eda, alpha=0.3)
plt.title("Scatter Plot: Distance vs Arrival Delay")
plt.xlabel("Distance (miles)")
plt.ylabel("Arrival Delay (minutes)")
plt.savefig(f"Outputs/scatter_distance_arrdelay.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,6))
sns.scatterplot(x='CRS_ELAPSED_TIME', y='ARR_DELAY', data=df_eda, alpha=0.3)
plt.title("Scatter Plot: Scheduled Duration vs Arrival Delay")
plt.xlabel("Scheduled Duration (minutes)")
plt.ylabel("Arrival Delay (minutes)")
plt.savefig(f"Outputs/scatter_crs_arrdelay.png", bbox_inches='tight')
plt.close()

print(f"EDA with original dataset done")

#%% 3a - Phase 2: Dimensionality Reduction / PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Apenas colunas númericas/encoded
features_for_dr = df.drop(
    columns=[
        'ARR_DELAY', 'AIRLINE', 'ORIGIN_CITY', 'DEST_CITY',
        'FL_DATE', 'FL_NUMBER', 'DOT_CODE', 'AIRLINE_DOT',
        'CRS_ELAPSED_TIME_minmax', 'DISTANCE_minmax'
    ],
    errors='ignore'
)
#print(features_for_dr.isnull().sum())

# PCA (Principal Component Analysis)
pca = PCA(n_components=2)  # reduzindo as novas colunas e conjunto de dados para 2 dimensões
pca_result = pca.fit_transform(features_for_dr)
df_pca = pd.DataFrame(
    pca_result,
    columns=['PC1', 'PC2']
)
df_pca['AIRLINE'] = df['AIRLINE'].values # adicionando as companhias para comparação

# Scatterplot
plt.figure(figsize=(14,6))
sns.scatterplot(
    data=df_pca,
    x='PC1',
    y='PC2',
    hue='AIRLINE',
    palette='tab20',
    alpha=0.6
)

plt.title("PCA Projection of Flights")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.savefig("Outputs/pca_projection.png", bbox_inches='tight')
plt.close()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

#%% 3b - Phase 2: Dimensionality Reduction / UMAP
import umap
import matplotlib.pyplot as plt
import seaborn as sns

umap_reducer = umap.UMAP(
    n_components=2,     # project down to 2D
    n_neighbors=50,     # controls local vs global structure
    min_dist=0.1,       # how tightly points are clustered
    random_state=42
)

# Fit and transform (demora um pouco)
print("Fitting")
umap_result = umap_reducer.fit_transform(features_for_dr)

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
plt.savefig("Outputs/umap_projection.png", bbox_inches='tight')
plt.close()

print("UMAP done... ")

#%% 6- Phase 2:  Hypothesis Testing
from scipy.stats import pearsonr, ttest_ind, f_oneway

df_hyp = df_hyp.dropna(subset=['ARR_DELAY']) # remove as linhas que têm nans, no preprocessing não havia problem manter
#print(df_hyp.isnull().sum())
#print(list(df_hyp.columns)) # Lista de colunas

# 1️⃣ Hypothesis 1: Correlation between flight distance and arrival delay
print("Hypothesis 1: Distance vs Arrival Delay")
corr_coeff, p_value = pearsonr(df_hyp['DISTANCE'], df_hyp['ARR_DELAY'])
print(f"Pearson correlation coefficient: {corr_coeff:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("✅ Significant correlation: flight distance is associated with delays.\n")
else:
    print("❌ No significant correlation between distance and delay.\n")

# 2️⃣ Hypothesis 2: Airline Southwest Airlines Co. vs Airline Delta Air Lines Inc. mean arrival delays (t-test)
airline_a = df_hyp['ARR_DELAY'][df_hyp['AIRLINE'] == 'Southwest Airlines Co.']
airline_b = df_hyp['ARR_DELAY'][df_hyp['AIRLINE'] == 'Delta Air Lines Inc.']

print("Hypothesis 2: Southwest Airlines Co. vs Delta Air Lines Inc. mean arrival delays") # muda airline para diferentes e verificar.
t_stat, p_value = ttest_ind(airline_a, airline_b)
print(f"T-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("✅ Significant difference in mean delays between Southwest Airlines Co. and Delta Air Lines Inc. .\n")
else:
    print("❌ No significant difference in mean delays between Southwest Airlines Co. and Delta Air Lines Inc. .\n")

# 3️⃣ Hypothesis 3: All airlines have the same mean arrival delay (ANOVA)
groups = [df_hyp['ARR_DELAY'][df_hyp['AIRLINE'] == airline] for airline in df_hyp['AIRLINE'].unique()]
f_stat, p_value = f_oneway(*groups)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("✅ Significant differences exist in delays between airlines.\n")
else:
    print("❌ No significant differences in delays between airlines.\n")

# 4️⃣ Hypothesis 4: Weather Delays VS Arrival Delays
print("Hypothesis 4: Weather-related delays vs Arrival Delay")

df_weather = df_hyp.dropna(subset=['DELAY_DUE_WEATHER'])

corr_coeff, p_value = pearsonr(df_weather['DELAY_DUE_WEATHER'], df_weather['ARR_DELAY'])

print(f"Pearson correlation coefficient: {corr_coeff:.3f}, p-value: {p_value:.3f}")

if p_value < 0.05:
    print("✅ Weather-related delays significantly impact arrival delays.\n")
else:
    print("❌ Weather-related delays do not significantly impact arrival delays.\n")

# 5️⃣ Hypothesis 5: Departure time vs Arrival Delay
print("Hypothesis 5: Scheduled Departure Hour vs Arrival Delay")

# Converte inteiros 3 dígitos para 4 com 0 atrás e retira a hora
df_hyp['CRS_DEP_TIME'] = df_hyp['CRS_DEP_TIME'].astype(str).str.zfill(4)
df_hyp['DEP_HOUR'] = df_hyp['CRS_DEP_TIME'].str[:2].astype(int)

# Agrupa hora por hora de partida
hour_groups = [df_hyp['ARR_DELAY'][df_hyp['DEP_HOUR'] == h] for h in sorted(df_hyp['DEP_HOUR'].unique())]

f_stat, p_value = f_oneway(*hour_groups)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")

if p_value < 0.05:
    print("✅ Significant differences in mean arrival delays across departure hours.\n")
else:
    print("❌ No significant differences in mean arrival delays across departure hours.\n")

#%% 7- Phase 3: Model Selection / New Features

# Categorical Encoding
print("Encoding... ")
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

# LabelEncoder para o ORIGIN e DEST, pois contêm muitos valores unicos
le_origin = LabelEncoder()
le_dest = LabelEncoder()
df['ORIGIN_label'] = le_origin.fit_transform(df['ORIGIN'])
df['DEST_label'] = le_dest.fit_transform(df['DEST'])

# Apagando as colunas para o modelo
df.drop(columns=['AIRLINE_CODE', 'ORIGIN', 'DEST'], inplace=True)
#print("New dataset shape:", df.shape)
#print(list(df.columns)) # Verifica-se se as colunas encoded foram adicionadas
print("Done Encoding..")

# Standardization/Normalization
print("Standardizing/Normalizing...")
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
print("Done Standardizing/Normalizing..")

# Binning
# Divide CRS_DEP_TIME em 4 bins
df['CRS_DEP_TIME'] = df['CRS_DEP_TIME'].astype(str).str.zfill(4)
df['DEP_HOUR'] = df['CRS_DEP_TIME'].str[:2].astype(int)
hour_bins = [0, 6, 12, 18, 24]
labels = ['Early Morning', 'Morning', 'Afternoon', 'Evening/Night']

df['dep_time_bin'] = pd.cut(df['DEP_HOUR'], bins=hour_bins, labels=labels, right=False)
print(df['dep_time_bin'].value_counts())

# Divide CRS_ELAPSED_TIME em 4 bins
duration_bins = [0, 60, 120, 240, df['CRS_ELAPSED_TIME'].max()+1]
duration_labels = ['Short', 'Medium-Short', 'Medium-Long', 'Long']

df['flight_duration_bin'] = pd.cut(df['CRS_ELAPSED_TIME'],bins=duration_bins,labels=duration_labels,right=False)
print(df['flight_duration_bin'].value_counts())

# Divide DISTANCE em 4 bins
distance_bins = [0, df['DISTANCE'].quantile(0.25), df['DISTANCE'].quantile(0.5),
                 df['DISTANCE'].quantile(0.75), df['DISTANCE'].max()]
distance_labels = ['Short', 'Medium-Short', 'Medium-Long', 'Long']

df['distance_bin'] = pd.cut(df['DISTANCE'], bins=distance_bins, labels=distance_labels, include_lowest=True)
print(df['distance_bin'].value_counts())

# DURATION X DISTANCE
df['elapsed_x_distance'] = df['CRS_ELAPSED_TIME'] * df['DISTANCE']

# DEP_HOUR X CRS_ELAPSED_TIME
df['dep_hour_x_elapsed'] = df['DEP_HOUR'] * df['CRS_ELAPSED_TIME']

# TIME FEATURES
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
df['MONTH'] = df['FL_DATE'].dt.month
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
df['IS_RUSH_HOUR'] = df['DEP_HOUR'].between(16, 20).astype(int)
df.drop(columns=['FL_DATE'], inplace=True) # Remover pois já extraimos features boas

print(df[['elapsed_x_distance', 'dep_hour_x_elapsed']].head())
df.to_csv('datasets/flights_cleaned_and_scaled.csv', index=False)

#%% 8- Phase 3: Model Selection / Model Selection - Linear Regression Testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(list(df.columns))  # Check all columns

# Preparando
X = df.drop(columns=['ARR_DELAY', 'CRS_DEP_TIME', 'dep_time_bin', 'flight_duration_bin', 'distance_bin'])
y = df['ARR_DELAY']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit baseline Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2: {r2:.3f}")

#%% 8- Phase 3: Model Selection / Model Selection - Random Forest Regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Prepare features and target
X = df.drop(columns=['ARR_DELAY', 'CRS_DEP_TIME', 'dep_time_bin', 'flight_duration_bin', 'distance_bin'])
y = df['ARR_DELAY']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=200,  # number of trees
    max_depth=15,      # limit depth to prevent overfitting
    random_state=42,
    n_jobs=-1          # use all cores
)

# Train model
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Results:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2: {r2:.3f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nTop Features by Importance:")
print(feature_importance.head(15))

# Optional: Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feature_importance['feature'].head(15)[::-1], feature_importance['importance'].head(15)[::-1])
plt.xlabel("Importance")
plt.title("Top 15 Features - Random Forest")
plt.show()
