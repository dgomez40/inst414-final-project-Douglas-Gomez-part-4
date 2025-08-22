import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(X_test_scaled, y_test):
    """Creates metrics necessary for evaluating the model, including mean absolute error, root mean squared error, and the r^2 value.

    Args:
        X_test_scaled (ndarray): Scaled test features  
        y_test (Series): Test target values 
    """
    model = joblib.load('data/models/linear_regression_model.joblib')
    y_pred = model.predict(X_test_scaled)
    # print(y_pred)
    # print(y_test)

    mean_absolute_err = mean_absolute_error(y_test, y_pred)
    root_mean_squared_error = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f'mean absolute error {mean_absolute_err}')
    print(f'root mean squared error {root_mean_squared_error}')
    print(f'r2 {r2}')

    # Ensure y_test and y_pred are 1D
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7)