import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

# Initialize a DataFrame to store model metrics
model_metrics = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R-squared'])

# Create a function to plot learning curves
def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Negative MSE")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training MSE")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation MSE")

    plt.legend(loc="best")
    plt.show()

# Train and evaluate the Random Forest Regression model
X_rf = combined_data[input_parameters]
y_rf = combined_data[target_variable]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=False)
model_rf.fit(X_train_rf, y_train_rf)

# Plot learning curve for Random Forest
plot_learning_curve(model_rf, X_rf, y_rf, "Learning Curve (Random Forest)")

# Train and evaluate the Linear Regression model
X_lr = combined_data[input_parameters]
y_lr = combined_data[target_variable]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train_lr)

# Plot learning curve for Linear Regression
plot_learning_curve(model_lr, X_lr, y_lr, "Learning Curve (Linear Regression)")

# Train and evaluate the Support Vector Machine (SVM) model
X_svm = combined_data[input_parameters]
y_svm = combined_data[target_variable]

scaler_svm = StandardScaler()
X_scaled_svm = scaler_svm.fit_transform(X_svm)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled_svm, y_svm, test_size=0.2, random_state=42)

model_svm = SVR(kernel='linear')  # You can adjust the kernel as needed (e.g., 'rbf' for radial basis function)
model_svm.fit(X_train_svm, y_train_svm)

# Plot learning curve for SVM
plot_learning_curve(model_svm, X_scaled_svm, y_svm, "Learning Curve (Support Vector Machine)")
