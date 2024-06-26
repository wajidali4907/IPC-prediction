import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load your combined CSV data directly
file_path = '/home/wajid/IPC prediction/output1.csv'  # Update the absolute file path
combined_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# Define the input parameters (features) you want to use
input_parameters =  [
    'numLoadInsts',
    'numStoreInsts',
    'numBranches',
    'intAluAccesses',
    'numOps',
    'model',
    'L1Icache',
    'L1Dcache',
    'L2cache',
    'pipelinewidth'
]

# Normalize the input features using StandardScaler, scaling them by 'numInsts'
scaler = StandardScaler()
for parameter in input_parameters:
    if parameter != 'numInsts':  # Skip 'numOps' itself
        combined_data[parameter] /= combined_data['numInsts']

X_normalized = scaler.fit_transform(combined_data[input_parameters])

# Define the target variable (y)
target_variables = [
    'ipc'
]

# Select the input parameters and target variable from the DataFrame
X = pd.DataFrame(X_normalized, columns=input_parameters)  # Convert back to DataFrame
y = combined_data[target_variables[0]]  # Assuming only one target variable for simplicity

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target variable values on the test set
y_pred = model.predict(X_test)

# Create a scatter plot of actual vs. predicted values

# Save the actual and predicted IPC values to a CSV file
result_df = pd.DataFrame({'Actual IPC': y_test, 'Predicted IPC': y_pred})
result_df.to_csv('/home/wajid/IPC prediction/LinearRegression_normalize.csv', index=False)  # Update the absolute file path
