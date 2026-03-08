#%% 1 - Phase 2: Data Analysis and Cleansing / Pre-processing
# ==============================

# Load libraries
library(dplyr)

# Load CSV
df <- read.csv("datasets/flights_sample_3m.csv")
df_eda <- df  # Guarda para a parte EDA
cat("Starting pre-processing...\n")

# Show columns (Com tudo)
print(colnames(df))

# Remove as linhas que contêm null em linhas importantes
df <- df %>% 
  filter(!is.na(CRS_ELAPSED_TIME)) %>%
  filter(CANCELLED == 0) %>%
  filter(DIVERTED == 0) %>%
  filter(!is.na(ARR_DELAY))

# Guarda para a parte hyp
df_hyp <- df

# Remover colunas desnecessárias, estas colunas são dados do futuro e não ajudam a prever 
cols_to_drop <- c(
  'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
  'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
  'ARR_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON',
  'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME','CANCELLED','CANCELLATION_CODE','DIVERTED','AIRLINE',
  'AIRLINE_DOT','DOT_CODE','FL_NUMBER','ORIGIN_CITY','DEST_CITY','CRS_ARR_TIME'
)
df <- df %>% select(-all_of(cols_to_drop))
print(colSums(is.na(df))) # Verificar se o df ficou limpo.
print(colnames(df))

# Remove outliers using IQR
numeric_cols <- c('CRS_ELAPSED_TIME', 'DISTANCE') # Colunas que fazem sentido

for (col_name in numeric_cols) {
  Q1 <- quantile(df[[col_name]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[col_name]], 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_val
  upper_bound <- Q3 + 1.5 * IQR_val
  
  # Replace outliers with NA
  df[[col_name]] <- ifelse(df[[col_name]] < lower_bound | df[[col_name]] > upper_bound,NA, df[[col_name]])
}

# Preenchendo os nans com valores medianos
for (col_name in numeric_cols) {
  median_val <- median(df[[col_name]], na.rm = TRUE)
  df[[col_name]][is.na(df[[col_name]])] <- median_val
}

# Guardando dataset pre-scaled/encoded só em caso
df_cleaned <- df
cat("Processing done.\n")

#%% 2 - Phase 2: Data Analysis and Cleansing / Exploratory Data Analysis (EDA)
# ==============================
library(ggplot2)
library(dplyr)
library(tidyr)

# Numeric columns for analysis
numeric_cols <- c('CRS_ELAPSED_TIME', 'DISTANCE', 'ARR_DELAY')

# Summary statistics
cat("\nBasic statistics:\n")
print(summary(df_eda[, numeric_cols]))
cat("\nMedian values:\n")
print(sapply(df_eda[, numeric_cols], median, na.rm = TRUE))
cat("\nMean values:\n")
print(sapply(df_eda[, numeric_cols], mean, na.rm = TRUE))
cat("\nStandard deviation:\n")
print(sapply(df_eda[, numeric_cols], sd, na.rm = TRUE))

# Ensure Outputs directory exists
if(!dir.exists("Outputs")) dir.create("Outputs")

# Top 10 airlines by number of flights
top_airlines <- df_eda %>%
  count(AIRLINE) %>%
  arrange(desc(n)) %>%
  slice(1:10)

ggplot(top_airlines, aes(x = n, y = reorder(AIRLINE, n))) +
  geom_col(fill = "steelblue") +
  labs(title = "Top 10 Airlines by Number of Flights", x = "Count", y = "Airline") +
  theme_minimal()
ggsave("Outputs/R_top10_airlines_count.png", width = 12, height = 6)

# Histograms + Density
for(col in numeric_cols){
  p <- ggplot(df_eda, aes(x = .data[[col]])) +
    geom_histogram(bins = 50, fill = "skyblue", color = "black", na.rm = TRUE) +
    geom_density(aes(y = after_stat(count)), color = "red", na.rm = TRUE) +
    labs(title = paste("Distribution of", col), x = col, y = "Count") +
    theme_minimal()
  
  ggsave(paste0("Outputs/R_hist_", col, ".png"), plot = p, width = 8, height = 4)
}

# Boxplot ARR_DELAY by airline
p <- ggplot(df_eda, aes(x = AIRLINE, y = ARR_DELAY)) +
  geom_boxplot(fill = "lightgreen", na.rm = TRUE) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplot of Arrival Delay by Airline", x = "Airline", y = "Arrival Delay")
ggsave("Outputs/R_boxplot_ARR_DELAY_by_airline.png", plot = p, width = 12, height = 6)

# Correlation heatmap using tidyr instead of reshape2
corr_matrix <- round(cor(df_eda[, numeric_cols], use = "complete.obs"), 2)
corr_melt <- as.data.frame(as.table(corr_matrix))  # melt replacement

p <- ggplot(corr_melt, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap") +
  theme_minimal()
ggsave("Outputs/R_corr_heatmap.png", plot = p, width = 6, height = 5)

# Scatter plots
p <- ggplot(df_eda, aes(x = DISTANCE, y = ARR_DELAY)) +
  geom_point(alpha = 0.3, na.rm = TRUE) +
  labs(title = "Scatter Plot: Distance vs Arrival Delay", x = "Distance (miles)", y = "Arrival Delay (minutes)") +
  theme_minimal()
ggsave("Outputs/R_scatter_distance_arrdelay.png", plot = p, width = 8, height = 6)

p <- ggplot(df_eda, aes(x = CRS_ELAPSED_TIME, y = ARR_DELAY)) +
  geom_point(alpha = 0.3, na.rm = TRUE) +
  labs(title = "Scatter Plot: Scheduled Duration vs Arrival Delay", x = "Scheduled Duration (minutes)", y = "Arrival Delay (minutes)") +
  theme_minimal()
ggsave("Outputs/R_scatter_crs_arrdelay.png", plot = p, width = 8, height = 6)

cat("\nEDA with modernized code done\n")