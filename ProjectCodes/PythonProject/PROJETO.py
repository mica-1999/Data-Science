#%% 1- Phase 2: Data Analysis and Cleansing / Pre-processing
import pandas as pd
import numpy as np

# Loading CSV
df = pd.read_csv('ProjectDatasets/flights_sample_3m.csv')
df_eda = df.copy()  # Pre pre-processing, guarda o original para a parte EDA

# Prints
print("Starting pre-processing... ")
#print(df.head()) # Checking if it loaded
#print(df.info()) # Info sobre as colunas e tipos de valores (string, int, etc..)
#print(df.describe()) # Dados estatísticos sobre as colunas
#print(df.isnull().sum()) # Verificando o nº de nulls nas colunas por linha
#print("Dataset shape:", df.shape) # linhas x colunas nº
#print(list(df.columns)) # Lista de colunas

# Remover linhas desnecessárias/contêm null em colunas importantes
df.dropna(subset=['CRS_ELAPSED_TIME'], inplace=True) # Apaga as linhas onde "CRS_ELAPSED_TIME" é null
df = df[df['CANCELLED'] == 0] # Apenas removemos as linhas dos voos cancelados, podemos agora remover a coluna
df = df[df['DIVERTED'] == 0] # Apenas removemos as linhas dos voos não rotados, podemos agora remover a coluna
df = df.dropna(subset=['ARR_DELAY']) # Removemos as 2 linhas que tinham null em ARR_DELAY
df = df[(df['DISTANCE'] >= 50) & (df['DISTANCE'] <= 5500)].reset_index(drop=True) # Removemos voos demasiado curtos ou demasiado longos para ser verdade

df_hyp = df.copy() # Guardando para hyp

# Remover colunas desnecessárias, estas colunas são dados do futuro e não ajudam a prever
cols_to_drop = [
    'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
    'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
    'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
    'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME','CANCELLED','CANCELLATION_CODE','DIVERTED',
    'AIRLINE_DOT','DOT_CODE','FL_NUMBER','ORIGIN_CITY','DEST_CITY','AIRLINE'
]
df.drop(columns=cols_to_drop, inplace=True)
#print("New dataset shape:", df.shape) # Verificando se foram apagadas
print(df.isnull().sum()) # Verificar se o df ficou limpo.
print(list(df.columns)) # Lista de colunas

# Remover outliers usando IQR
numeric_cols_preproc = ['CRS_ELAPSED_TIME'] # Colunas que fazem sentido
#print(df[numeric_cols_preproc].describe()) # Before handling
for column_name in numeric_cols_preproc:
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
for column_name in numeric_cols_preproc:
    df[column_name] = df[column_name].fillna(df[column_name].median())

df = df.reset_index(drop=True) # Faz reset do index para resolver as linhas saltadas (quando foram removidas)

# Guardando dataset pre-scaled/encoded só em caso
df_cleaned = df.copy()
#df.to_csv('ProjectDatasets/flights_cleaned.csv', index=False)

print(f"Processing done")

#%% 1.1- Phase 2: Standardization for PCA / EDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

numeric_cols_pca = ['CRS_ELAPSED_TIME', 'DISTANCE', 'CRS_DEP_TIME', 'CRS_ARR_TIME']

# Standardization
standard_scaler = StandardScaler()
df_scaled_std = pd.DataFrame(
    standard_scaler.fit_transform(df_cleaned[numeric_cols_pca]),
    columns=[f"{col}_std" for col in numeric_cols_pca]
)

# Normalization
minmax_scaler = MinMaxScaler()
df_scaled_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(df_cleaned[numeric_cols_pca]),
    columns=[f"{col}_minmax" for col in numeric_cols_pca]
)

# Concatenate scaled data with df_cleaned if needed
df = pd.concat([df_cleaned.reset_index(drop=True), df_scaled_std, df_scaled_minmax], axis=1)

print(df[[f"{col}_std" for col in numeric_cols_pca] +
         [f"{col}_minmax" for col in numeric_cols_pca]].head())
print("Standardization and normalization done.")

#%% 2- Phase 2: Data Analysis and Cleansing / Exploratory Data Analysis (EDA)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Filter FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Numeric columns for analysis
numeric_cols_eda = ['CRS_ELAPSED_TIME', 'DISTANCE', 'ARR_DELAY']

# Summary statistics with pandas
print("\nBasic statistics:\n", df_eda[numeric_cols_eda].describe())
print("\nMedian values:\n", df_eda[numeric_cols_eda].median())
print("\nMean values:\n", df_eda[numeric_cols_eda].mean())
print("\nStandard deviation:\n", df_eda[numeric_cols_eda].std())

# Data distribution analysis
plt.figure(figsize=(12,6))
sns.countplot(y='AIRLINE', data=df_eda, order=df_eda['AIRLINE'].value_counts().index[:10])
plt.title("Top 10 Airlines by Number of Flights")
plt.xlabel("Count")
plt.ylabel("Airline")
plt.savefig("OutputFiles/Python/preFeatureEngineering/top10_airlines_count.png", bbox_inches='tight')
plt.close()

numeric_cols_eda = [
    'CRS_ELAPSED_TIME',  # scheduled flight time
    'ELAPSED_TIME',      # actual flight time
    'AIR_TIME',          # time in the air
    'DISTANCE',          # distance between airports
    'ARR_DELAY',         # arrival delay (target)
    'DEP_DELAY',         # departure delay
    'TAXI_OUT',          # time from gate to takeoff
    'TAXI_IN',           # time from landing to gate
    'DELAY_DUE_CARRIER', # delay caused by airline
    'DELAY_DUE_WEATHER', # weather delays
    'DELAY_DUE_NAS',     # national airspace system
    'DELAY_DUE_LATE_AIRCRAFT'  # late aircraft cascading delays
]

# Histograms
for col in numeric_cols_eda:
    plt.figure(figsize=(8,4))
    sns.histplot(df_eda[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"OutputFiles/Python/preFeatureEngineering/hist_{col}.png", bbox_inches='tight')
    plt.close()

# Box plots por companhia
for col in ['ARR_DELAY']:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='AIRLINE', y=col, data=df_eda)
    plt.ylim(-100, 100)
    plt.xticks(rotation=45)
    plt.savefig(f"OutputFiles/Python/preFeatureEngineering/boxplot_{col}_by_airline.png", bbox_inches='tight')
    plt.close()

# Correlation matrix & Heatmap
corr_matrix = df_eda[numeric_cols_eda].corr()
plt.figure(figsize=(12,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("OutputFiles/Python/preFeatureEngineering/corr_heatmap.png", bbox_inches='tight')
plt.close()

# KDE Plots (density per airline)
for col in ['ARR_DELAY']:
    plt.figure(figsize=(12,6))
    sns.kdeplot(data=df_eda, x=col, hue='AIRLINE', fill=True, alpha=0.5)
    plt.title(f"KDE Plot of {col} by Airline")
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.savefig(f"OutputFiles/Python/preFeatureEngineering/kde_{col}_by_airline.png", bbox_inches='tight')
    plt.close()

# Scatter Plots
scatter_plots = [
    ('DISTANCE', 'ARR_DELAY', "Distance vs Arrival Delay", "scatter_distance_arrdelay.png"),
    ('CRS_ELAPSED_TIME', 'ARR_DELAY', "Scheduled Duration vs Arrival Delay", "scatter_crs_arrdelay.png"),
    ('DEP_DELAY', 'ARR_DELAY', "Departure Delay vs Arrival Delay", "scatter_depdelay_arrdelay.png"),
    ('TAXI_OUT', 'ARR_DELAY', "Taxi Out vs Arrival Delay", "scatter_taxiout_arrdelay.png"),
    ('TAXI_IN', 'ARR_DELAY', "Taxi In vs Arrival Delay", "scatter_taxiin_arrdelay.png"),
    ('AIR_TIME', 'ARR_DELAY', "Air Time vs Arrival Delay", "scatter_airtime_arrdelay.png"),
    ('DISTANCE', 'CRS_ELAPSED_TIME', "Distance vs Scheduled Duration", "scatter_distance_crselapsed.png"),
    ('DELAY_DUE_CARRIER', 'ARR_DELAY', "Carrier Delay vs Arrival Delay", "scatter_carrierdelay_arrdelay.png"),
    ('DELAY_DUE_WEATHER', 'ARR_DELAY', "Weather Delay vs Arrival Delay", "scatter_weatherdelay_arrdelay.png"),
    ('DELAY_DUE_NAS', 'ARR_DELAY', "NAS Delay vs Arrival Delay", "scatter_nasdelay_arrdelay.png"),
    ('DELAY_DUE_LATE_AIRCRAFT', 'ARR_DELAY', "Late Aircraft Delay vs Arrival Delay", "scatter_lateaircraft_arrdelay.png")
]

# Loop through and generate each scatter plot
for x_col, y_col, title, filename in scatter_plots:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x_col, y=y_col, data=df_eda, alpha=0.3)
    plt.title(f"Scatter Plot: {title}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig(f"OutputFiles/Python/preFeatureEngineering/{filename}", bbox_inches='tight')
    plt.close()

print(f"EDA with original dataset done")

#%% 3a - Phase 2: Dimensionality Reduction / PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

print(list(df.columns))
# Columns to include in PCA
features_for_dr = df[['CRS_ELAPSED_TIME_std', 'DISTANCE_std','CRS_DEP_TIME_std', 'CRS_ARR_TIME_std']]

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_for_dr)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca['AIRLINE'] = df_eda.loc[features_for_dr.index, 'AIRLINE'].values

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
plt.savefig("OutputFiles/Python/preFeatureEngineering/pca_projection.png", bbox_inches='tight')
plt.close()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

# %% 3b - Phase 2: Dimensionality Reduction / UMAP
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Sample features for faster computation
features_sample = features_for_dr.sample(200_000, random_state=42)
airline_sample = df_eda.loc[features_sample.index, 'AIRLINE'].values

# Initialize UMAP
umap_reducer = umap.UMAP(
    n_components=2,
    n_neighbors=50,
    min_dist=0.1,
    random_state=42
)

# Fit and transform
print("Fitting UMAP on sample...")
umap_result = umap_reducer.fit_transform(features_sample)

# Create DataFrame for plotting
df_umap = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
df_umap['AIRLINE'] = airline_sample

# Plot UMAP projection
print("Plotting...")
plt.figure(figsize=(14, 6))
sns.scatterplot(
    data=df_umap,
    x='UMAP1',
    y='UMAP2',
    hue='AIRLINE',
    palette='tab20',
    alpha=0.5
)
plt.title("UMAP Projection of Flights (Sample)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("OutputFiles/Python/preFeatureEngineering/umap_projection.png", bbox_inches='tight')
plt.close()

print("UMAP done...")

# Quick check: airline cluster centers
print(df_umap.groupby('AIRLINE')[['UMAP1', 'UMAP2']].mean())

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

# Extract the departure hour directly from CRS_DEP_TIME
df_hyp['DEP_HOUR'] = df_hyp['CRS_DEP_TIME'] // 100

# Agrupa hora por hora de partida
hour_groups = [df_hyp['ARR_DELAY'][df_hyp['DEP_HOUR'] == h] for h in sorted(df_hyp['DEP_HOUR'].unique())]

f_stat, p_value = f_oneway(*hour_groups)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")

if p_value < 0.05:
    print("✅ Significant differences in mean arrival delays across departure hours.\n")
else:
    print("❌ No significant differences in mean arrival delays across departure hours.\n")

# 7️⃣ Hypothesis 6: Interaction Between Flight Distance and Weather Delays

# Remove rows with missing weather delay data
df_weather = df_hyp.dropna(subset=['DELAY_DUE_WEATHER'])

# Overall correlation between weather delays and arrival delays
corr_coeff, p_value = pearsonr(df_weather['DELAY_DUE_WEATHER'], df_weather['ARR_DELAY'])

print("Bonus Hypothesis 6: Weather Delay vs Arrival Delay")
print(f"Pearson correlation coefficient: {corr_coeff:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    print("✅ Weather-related delays significantly impact arrival delays.\n")
else:
    print("❌ Weather-related delays do not significantly impact arrival delays.\n")

#%% 7- Phase 3: Model Selection / New Features
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Categorical Encoding
print("Encoding... ")
categorical_cols = ['AIRLINE_CODE', 'ORIGIN', 'DEST'] # Unicos que fazem sentido dividir em categorias para o modelo
numeric_cols_features = ['CRS_ELAPSED_TIME', 'DISTANCE']

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
print(list(df.columns)) # Lista de colunas
#df.to_csv('ProjectDatasets/flights_cleaned_and_scaled.csv', index=False)

#%% 8- Phase 3: Model Selection / Model Selection - Linear Regression Testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(list(df.columns))  # Check all columns

# Preparando
X = df.drop(columns=['ARR_DELAY', 'CRS_DEP_TIME', 'dep_time_bin', 'flight_duration_bin', 'distance_bin',
                     'CRS_ELAPSED_TIME_minmax','DISTANCE_minmax'])
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
X = df.drop(columns=['ARR_DELAY', 'CRS_DEP_TIME', 'dep_time_bin', 'flight_duration_bin', 'distance_bin',
                     'CRS_ELAPSED_TIME_std','DISTANCE_std','CRS_ELAPSED_TIME_minmax','DISTANCE_minmax'])
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
plt.savefig(f"../../OutputFiles/top15features_for_pred.png", bbox_inches='tight')
plt.show()
