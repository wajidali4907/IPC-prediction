import os
import re
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the aggregated data from the CSV file
input_csv = 'output.csv'
data = pd.read_csv(input_csv)

# Define your features (X) and target variable (y)
X = data[['cpi', 'numLoadInsts', 'numStoreInsts', 'numInsts', 'numBranches', 'intAluAccesses', 'numOps']]
y = data['ipc']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict IPC for the test set
y_pred = model.predict(X_test)

# Create a scatter plot of actual vs. predicted IPC
plt.scatter(y_test, y_pred)
plt.xlabel("Actual IPC")
plt.ylabel("Predicted IPC")
plt.title("Actual IPC vs. Predicted IPC")
plt.grid(True)

# Plot a line for perfect predictions
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)

plt.show()
