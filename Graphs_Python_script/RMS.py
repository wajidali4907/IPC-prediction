import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your combined CSV data directly
file_path_rf = '/home/wajid/IPC prediction/output1.csv'
file_path_lr = '/home/wajid/IPC prediction/output1.csv'
file_path_svm = '/home/wajid/IPC prediction/output1.csv'

combined_data_rf = pd.read_csv(file_path_rf, delimiter=',', encoding='utf-8')
combined_data_lr = pd.read_csv(file_path_lr, delimiter=',', encoding='utf-8')
combined_data_svm = pd.read_csv(file_path_svm, delimiter=',', encoding='utf-8')

# Define the input parameters (features) you want to use
input_parameters_rf =  [
    'numLoadInsts',
    'numStoreInsts',
    'numInsts',
    'numBranches',
    'intAluAccesses',
    'numOps',
    'model',
    'L1Icache',
    'L1Dcache',
    'L2cache',
    'pipelinewidth'
]

input_parameters_lr =  [
    'numLoadInsts',
    'numStoreInsts',
    'numInsts',
    'numBranches',
    'intAluAccesses',
    'numOps',
    'model'
]

input_parameters_svm =  [
    'numLoadInsts',
    'numStoreInsts',
    'numInsts',
    'numBranches',
    'intAluAccesses',
    'numOps',
]

# Define the target variable (y)
target_variable = 'ipc'

# Create dictionaries to store the trained models
models = {
    "Random Forest": None,
    "Linear Regression": None,
    "Support Vector Machine": None
}

# Calculate the Mean Squared Error (MSE) for each model
mse_scores = {}

# Random Forest
X_rf = combined_data_rf[input_parameters_rf]
y_rf = combined_data_rf[target_variable]
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train_rf, y_train_rf)
y_pred_rf = model_rf.predict(X_test_rf)
mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
mse_scores["Random Forest"] = mse_rf
models["Random Forest"] = model_rf

# Linear Regression
X_lr = combined_data_lr[input_parameters_lr]
y_lr = combined_data_lr[target_variable]
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train_lr)
y_pred_lr = model_lr.predict(X_test_lr)
mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
mse_scores["Linear Regression"] = mse_lr
models["Linear Regression"] = model_lr

# Support Vector Machine (SVM)
X_svm = combined_data_svm[input_parameters_svm]
y_svm = combined_data_svm[target_variable]
scaler_svm = StandardScaler()
X_scaled_svm = scaler_svm.fit_transform(X_svm)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled_svm, y_svm, test_size=0.2, random_state=42)
model_svm = SVR(kernel='linear')
model_svm.fit(X_train_svm, y_train_svm)
y_pred_svm = model_svm.predict(X_test_svm)
mse_svm = mean_squared_error(y_test_svm, y_pred_svm)
mse_scores["Support Vector Machine"] = mse_svm
models["Support Vector Machine"] = model_svm

# Define colors for the bars
colors = ['b', 'g', 'r']

# Create a bar chart to compare the models with colors and a legend
plt.figure(figsize=(10, 6))
for i, model_name in enumerate(mse_scores.keys()):
    plt.bar(model_name, mse_scores[model_name], color=colors[i], alpha=0.6, label=model_name)
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Model Comparison')
plt.legend()
plt.show()
