import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load your combined CSV data directly
file_path = '/home/wajid/IPC prediction/output.csv'
combined_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# Define the input parameters (features) you want to use
input_parameters =  [
    'numLoadInsts',
    'numStoreInsts',
    'numInsts',
    'numBranches',
    'intAluAccesses',
    'numOps',
]

# Define the target variable (y)
target_variables = [
    'ipc'
]

# Create dictionaries to store the trained models and predictions
models = {}
predictions = {}

# Train a separate Support Vector Machine (SVM) model for each target variable
for target_variable in target_variables:
    # Select the input parameters and current target variable from the DataFrame
    X = combined_data[input_parameters]
    y = combined_data[target_variable]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Create and train the Support Vector Machine (SVM) model
    model = SVR(kernel='linear')  # You can adjust the kernel as needed (e.g., 'rbf' for radial basis function)
    model.fit(X_train, y_train)
    
    # Store the trained model in the dictionary
    models[target_variable] = model

# Define a function to predict target variables based on input metrics
def predict_target(input_metrics):
    for target_variable, model in models.items():
        # Use the trained model to predict the current target variable for the input metrics
        input_metrics_scaled = scaler.transform(np.array(input_metrics).reshape(1, -1))
        predicted_value = model.predict(input_metrics_scaled)[0]
        predictions[target_variable] = predicted_value

# Prompt the user to input values for the input parameters
input_metrics = []
for parameter in input_parameters:
    value = float(input(f'Enter value for {parameter}: '))
    input_metrics.append(value)

# Get the predicted values for all target variables
predict_target(input_metrics)

# Print the predicted values for each target variable
for target_variable, predicted_value in predictions.items():
    print(f'Predicted {target_variable}: {predicted_value:.4f}')
