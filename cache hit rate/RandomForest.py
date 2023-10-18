import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load your combined CSV data directly
file_path = '/home/wajid/IPC prediction/a.csv'

combined_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
#print(combined_data.columns)

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
target_variables = [
    'ipc',
    'hit'
]

# Create dictionaries to store the trained models and predictions
models = {}
predictions = {}

# Handle missing values in the target variable
for target_variable in target_variables:
    y = combined_data[target_variable]

    # Check for and handle missing values
    if y.isna().any():
        print(f"Warning: Missing values found in {target_variable}. Imputing with mean.")
        y.fillna(y.mean(), inplace=True)

    # Split the dataset into training and testing sets
    X = combined_data[input_parameters]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Regression model with 100 estimators (you can adjust this value)
    model = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=False)

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
    print(f'Predicted {target_variable}: {predicted_value:.4f}')
