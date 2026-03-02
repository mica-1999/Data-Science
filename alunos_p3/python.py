#%% 1- Regression Modeling
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
print(df.head()) # Mostra a dataset sem o price column
df['price'] = california_housing.target  # Adiciona uma coluna target e combina

# Use 'price' as target variable
y = df['price']  # target variable / coisa que queremos para o modelo prever
X = df.drop('price', axis=1)  # all other columns are features / coisas que o modelo usa para prever o preço

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 20% dos dados vão ser testados e 80% para o treino
)

# Create a linear regression model
model = LinearRegression() # Cria uma instância

# Train the model
model.fit(X_train, y_train) # Treina o model usando 80% dos dados

# Make predictions on the test set
y_pred = model.predict(X_test) # O modelo vai prever os preços

# Evaluate the model calculating mse and r2
mse = mean_squared_error(y_test, y_pred) # Verificamos se o modelo funciona bem
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot the predicted vs actual values with tendency line
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # tendency line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices")
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pycm import *

# Load the California Housing dataset
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['price'] = california_housing.target

# Define a threshold for binary classification
threshold = 3.0

# Convert 'price' to binary (0 or 1)
df['target'] = (df['price'] > threshold).astype(int)

# Use 'target' as the binary target variable
y = df['target']  # target variable (0 = cheap, 1 = expensive)
X = df.drop(['price', 'target'], axis=1)  # all other columns are features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 20% para teste, 80% para treino
)

# Create a logistic regression model
model = LogisticRegression(max_iter=1000)  # max_iter increased to ensure convergence

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)  # predicted classes (0 or 1)
y_prob = model.predict_proba(X_test)[:, 1]  # predicted probabilities for class 1

# Evaluate the model showing all performance metrics
cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))
cm.stat(summary=True)

# Plot the confusion matrix
cm_matrix = confusion_matrix(y_test, y_pred)       # compute confusion matrix
disp = ConfusionMatrixDisplay(cm_matrix)           # create display object
disp.plot(cmap=plt.cm.Blues)                        # plot with color map
plt.title("Confusion Matrix")
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # compute false/true positive rates
roc_auc = auc(fpr, tpr)                            # compute AUC score

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')  # random baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%% 3-Hypothesis Testing - Correlation test

import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

def perform_correlation_test(feature1, feature2, alpha=0.05):
    # Calculate Pearson correlation coefficient and p-value
    # The Pearson correlation test evaluates whether the correlation coefficient calculated 
    # from the sample data is significantly different from 0, indicating whether there is a 
    # statistically significant linear relationship between the two variables in the population.
    # A value of +1 indicates a perfect positive linear relationship, meaning that as one variable 
    # increases, the other variable also increases proportionally.
    # A value of -1 indicates a perfect negative linear relationship, meaning that as one variable 
    # increases, the other variable decreases proportionally.
    # A value of 0 indicates no linear relationship between the variables.
    corr_coeff, p_value = pearsonr(feature1, feature2)

    # Interpret the results
    if p_value < alpha:
        print(f"There is a significant correlation between the two features, the correlation coefficient was ", corr_coeff, " and the p-value was ", p_value)
    else:
        print(f"There is no significant correlation between the two features, the correlation coefficient was ", corr_coeff, " and the p-value was ", p_value)

    return corr_coeff, p_value

# Load the California Housing dataset


# Normalize the features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('Price', axis=1)), columns=df.columns[:-1])
df_scaled['Price'] = df['Price']  # Add back the 'Price' column

# Test all pairs of features


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


# Normalize the features


# Determine the means of housing prices for houses with high and low median income to then perform t-test
# Using 1 as threshold to define the high and low income groups (in the MedInc column)



#%% 5-Means test with groups

import warnings
# Suppress all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from scipy.stats import f_oneway
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset


# Add a column for categorical average rooms


# Perform ANOVA
f_statistic, p_value = f_oneway(
    df['price'][df['AveRoomsCategory'] == '0-3'],
    df['price'][df['AveRoomsCategory'] == '3-6'],
    df['price'][df['AveRoomsCategory'] == '6-9'],
    df['price'][df['AveRoomsCategory'] == '9+']
)

# Print ANOVA results (F-Statistic and P-Value) and check if it is significant


#%% 6-Training Strategies

# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset


# Split the data into training and testing sets


# Train a Logistic Regression classifier with multinomial to use multiclass


# Evaluate on the test set (check accuracy)


# Perform cross-validation using and print the mean and standard deviation of accuracy


# Bootstrapping
clf = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='lbfgs', random_state=42)
bootstrap = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
bootstrap_scores = cross_val_score(clf, X, y, cv=bootstrap)
print(f'Bootstrapping Mean Accuracy: {np.mean(bootstrap_scores):.2f}')
print(f'Bootstrapping SD Accuracy: {np.std(bootstrap_scores):.2f}')
