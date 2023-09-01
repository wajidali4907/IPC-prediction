import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your combined CSV data directly
file_path = '/home/wajid/IPC prediction/output.csv'
combined_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# Define the input parameters (features) you want to use
input_parameters =  [
    'system.o3Cpu0.cpi',
    'system.o3Cpu0.numLoadInsts',
    'system.o3Cpu0.numInsts',
    'system.o3Cpu0.numBranches',
    'system.o3Cpu0.intAluAccesses',
    'system.o3Cpu0.thread_0.numOps',


]
# Define the target variable (y)
target_variables = [
    'system.o3Cpu0.ipc',

    
]


# Create dictionaries to store the trained models and predictions
models = {}
predictions = {}

# Train a separate Linear Regression model for each target variable
for target_variable in target_variables:
    # Select the input parameters and current target variable from the DataFrame
    X = combined_data[input_parameters]
    y = combined_data[target_variable]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Store the trained model in the dictionary
    models[target_variable] = model

# Define a function to predict target variables based on input metrics
def predict_target(input_metrics):
    for target_variable, model in models.items():
        # Use the trained model to predict the current target variable for the input metrics
        predicted_value = model.predict([input_metrics])
        predictions[target_variable] = predicted_value[0]

# Prompt the user to input values for the input parameters
input_metrics = []
for parameter in input_parameters:
    value = float(input(f'Enter value for {parameter}: '))
    input_metrics.append(value)

# Get the predicted values for all target variables
predict_target(input_metrics)

# Print the predicted values for each target variable
for target_variable, predicted_value in predictions.items():
    print(f'Predicted {target_variable}: {predicted_value:.2f}')