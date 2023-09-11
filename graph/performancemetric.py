import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load your combined CSV data directly
file_path = '/home/wajid/IPC prediction/output1.csv'
combined_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# Define the input parameters (features) you want to use
input_parameters =  [
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

# Define a function to calculate performance metrics
def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# Calculate performance metrics for the model
rmse, mae, r2 = calculate_metrics(model, X_test, y_test)

# Create a bar chart to display the metrics
metrics = ['RMSE', 'MAE', 'R-squared']
metric_values = [rmse, mae, r2]

plt.bar(metrics, metric_values, color=['blue', 'green', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Metric Values')
plt.title('Performance Metrics for the Model')
plt.show()
