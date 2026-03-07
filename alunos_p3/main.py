#%% 1- Regression Modeling

# California Housing dataset
# --------------------------
#
# **Data Set Characteristics:**
#
# :Number of Instances: 20640
#
# :Number of Attributes: 8 numeric, predictive attributes and the target
#
# :Attribute Information:
#     - MedInc        median income in block group
#     - HouseAge      median house age in block group
#     - AveRooms      average number of rooms per household
#     - AveBedrms     average number of bedrooms per household
#     - Population    block group population
#     - AveOccup      average number of household members
#     - Latitude      block group latitude
#     - Longitude     block group longitude
#
# The target variable is the median house value for California districts,
# expressed in hundreds of thousands of dollars ($100,000).
#
# This dataset was derived from the 1990 U.S. census, using one row per census
# block group. A block group is the smallest geographical unit for which the U.S.
# Census Bureau publishes sample data (a block group typically has a population
# of 600 to 3,000 people).
#
# A household is a group of people residing within a home. Since the average
# number of rooms and bedrooms in this dataset are provided per household, these
# columns may take surprisingly large values for block groups with few households
# and many empty houses, such as vacation resorts.

# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the California Housing dataset
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['price'] = california_housing.target

# Use 'price' as target variable
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the predicted vs actual values with tendency line
plt.grid(True)
plt.scatter(y_test, y_pred, label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Tendency Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.legend()
plt.show()

#%% 2- Classification Modeling

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from pycm import *

# Load the California Housing dataset
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['price'] = california_housing.target

# Define a threshold for binary classification
threshold = 3.0  # You can adjust this threshold as needed

# Convert 'price' to binary (0 or 1)
df['target'] = (df['price'] > threshold).astype(int)

# Use 'target' as the binary target variable
X = df.drop(['price', 'target'], axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a logistic regression model
model = LogisticRegression(max_iter=10000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model showing all performance metrics
cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))
cm.stat(summary=True)

# Plot the confusion matrix
pp_matrix_from_data(np.array(y_test), np.array(y_pred))
cm.print_normalized_matrix()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#%% 3-Hypothesis Testing - Correlation test

import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

def perform_correlation_test(feature1, feature2, alpha=0.05):
    # Calculate Pearson correlation coefficient and p-value
    corr_coeff, p_value = pearsonr(feature1, feature2)

    # Interpret the results
    if p_value < alpha:
        print(f"There is a significant correlation between the two features, the correlation coefficient was ", corr_coeff, " and the p-value was ", p_value)
    else:
        print(f"There is no significant correlation between the two features, the correlation coefficient was ", corr_coeff, " and the p-value was ", p_value)

    return corr_coeff, p_value

# Load the California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# Normalize the features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('Price', axis=1)), columns=df.columns[:-1])
df_scaled['Price'] = df['Price']  # Add back the 'Price' column

# Test all pairs of features
for i, feature1 in enumerate(df_scaled.columns[:-1]):
    for j, feature2 in enumerate(df_scaled.columns[:-1]):
        if i < j:  # Avoid redundant pairs and self-correlation
            print(f"\nTesting correlation between '{feature1}' and '{feature2}':")
            perform_correlation_test(df_scaled[feature1], df_scaled[feature2])

#%% 4-Hypothesis Testing - Means test

import pandas as pd
from scipy.stats import ttest_ind
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def perform_t_test(group1, group2, alpha=0.05):
    # Perform t-test
    t_statistic, p_value = ttest_ind(group1, group2)

    # Print results
    print("\nT-Test Results:")
    print("------------------------------------------------------")
    print("Comparison: Mean Housing Prices for High vs. Low Income")
    print("------------------------------------------------------")
    print(f"High Income Mean Price: {group1.mean()}")
    print(f"Low Income Mean Price: {group2.mean()}")

    print("\nT-Test Results:")
    print(f"  - T-Statistic: {t_statistic}")
    print(f"  - P-Value: {p_value}")

    print("\nConclusion:")
    if p_value < alpha:
        print("  There is a significant difference in mean housing prices between high and low-income houses.")
    else:
        print("  There is no significant difference in mean housing prices between high and low-income houses.")

# Load the California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target

# Normalize the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('price', axis=1)), columns=df.columns[:-1])
df_scaled['price'] = df['price']  # Add back the 'price' column

# Determine the means of housing prices for houses with high and low median income to then perform t-test
# Using 1 as threshold to define the high and low income groups (in the MedInc column)
high_income_prices = df_scaled['price'][df_scaled['MedInc'] > 1]
low_income_prices = df_scaled['price'][df_scaled['MedInc'] <= 1]
perform_t_test(high_income_prices, low_income_prices)

#%% 5-Means test with groups

import warnings
# Suppress all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from scipy.stats import f_oneway
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = data.target

# Add a column for categorical average rooms
df['AveRoomsCategory'] = pd.cut(df['AveRooms'], bins=[0, 3, 6, 9, float('inf')], labels=['0-3', '3-6', '6-9', '9+'])

# Perform ANOVA
f_statistic, p_value = f_oneway(
    df['price'][df['AveRoomsCategory'] == '0-3'],
    df['price'][df['AveRoomsCategory'] == '3-6'],
    df['price'][df['AveRoomsCategory'] == '6-9'],
    df['price'][df['AveRoomsCategory'] == '9+']
)

# Print ANOVA results
print("\nANOVA Results:")
print("------------------------------------------------------")
print(f"F-Statistic: {f_statistic}")
print(f"P-Value: {p_value}")

print("\nConclusion:")
if p_value < 0.05:
    print("  There are significant differences in housing prices among different ranges of average rooms.")
else:
    print("  There are no significant differences in housing prices among different ranges of average rooms.")

#%% 6-Training Strategies

# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier with multinomial to use multiclass
clf = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='lbfgs', random_state=42)
clf.fit(X_train, y_train)

# Evaluate on the test set
y_pred = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)
print(f'Test Accuracy With Standard Train-Test Split: {accuracy_test:.2f}')

# Cross-validation
clf = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='lbfgs', random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f'Cross-Validation Mean Accuracy: {np.mean(cv_scores):.2f}')
print(f'Cross-Validation SD Accuracy: {np.std(cv_scores):.2f}')

# Bootstrapping
clf = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='lbfgs', random_state=42)
bootstrap = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
bootstrap_scores = cross_val_score(clf, X, y, cv=bootstrap)
print(f'Bootstrapping Mean Accuracy: {np.mean(bootstrap_scores):.2f}')
print(f'Bootstrapping SD Accuracy: {np.std(bootstrap_scores):.2f}')
