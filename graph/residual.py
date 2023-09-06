import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your combined CSV data directly
file_path = '/home/wajid/IPC prediction/output.csv'
combined_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# Define the input parameters (features) you want to use
input_parameters = [
    'numLoadInsts',
    'numStoreInsts',
    'numInsts',
    'numBranches',
    'intAluAccesses',
    'numOps',
    'model'
]

# Define the target variable (y)
target_variable = 'ipc'

# Split the dataset into training and testing sets
X = combined_data[input_parameters]
y = combined_data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regression model with 100 estimators (you can adjust this value)
model = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=False)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate residuals (errors)
residuals = y_test - y_pred

# Create residual plots
plt.figure(figsize=(12, 6))

# Residuals vs. Predicted Values (to check for heteroscedasticity)
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.axhline(0, color='red', linestyle='--')

# Histogram of Residuals (to check for normality)
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.axvline(0, color='red', linestyle='--')

plt.tight_layout()
plt.show()
