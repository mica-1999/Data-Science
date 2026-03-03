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
#%% 1- Pre-processing
