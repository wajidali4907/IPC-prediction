import pandas as pd
import matplotlib.pyplot as plt
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

# Select the input parameters and target variable from the DataFrame
X = combined_data[input_parameters]
y = combined_data[target_variable]

# Create a Random Forest Regression model with 100 estimators (you can adjust this value)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the entire dataset
model.fit(X, y)

# Get feature importances from the trained model
feature_importances = model.feature_importances_

# Create a bar plot to visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(input_parameters, feature_importances)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Plot')
plt.xticks(rotation=45)
plt.show()
