import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# Create a list to store model metrics
model_metrics = []

# Train and evaluate the Random Forest Regression model
X = combined_data[input_parameters]
y = combined_data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

model_metrics.append({'Model': 'Random Forest', 'RMSE': rmse, 'MAE': mae, 'R-squared': r2})

# Train and evaluate the Linear Regression model
X = combined_data[input_parameters]
y = combined_data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

model_metrics.append({'Model': 'Linear Regression', 'RMSE': rmse, 'MAE': mae, 'R-squared': r2})

# Train and evaluate the Support Vector Machine (SVM) model
X = combined_data[input_parameters]
y = combined_data[target_variable]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVR(kernel='linear')  # You can adjust the kernel as needed (e.g., 'rbf' for radial basis function)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

model_metrics.append({'Model': 'Support Vector Machine', 'RMSE': rmse, 'MAE': mae, 'R-squared': r2})

# Convert the list of dictionaries to a DataFrame
model_metrics_df = pd.DataFrame(model_metrics)

# Create a bar chart to compare the models and their performance metrics
plt.figure(figsize=(10, 6))
models = model_metrics_df['Model']
rmse = model_metrics_df['RMSE']
mae = model_metrics_df['MAE']
r2 = model_metrics_df['R-squared']

plt.bar(models, rmse, color='b', alpha=0.6, label='RMSE')
plt.bar(models, mae, color='g', alpha=0.6, label='MAE')
plt.bar(models, r2, color='r', alpha=0.6, label='R-squared')
plt.xlabel('Models')
plt.ylabel('Metrics')
plt.title('Model Comparison')
plt.legend()
plt.show()
