import pandas as pd
import numpy as np

def load_data(X_train_scaled, X_test_scaled, y_train, y_test, y_pred, lr_model):
    """Load the test data into the pipeline and creates the train and test data CSV's respectfully.

    Args:
        X_train_scaled (ndarray): Scaled training features
        X_test_scaled (ndarray): Scaled test features
        y_train (Series): Training target values
        y_test (Series): Test target values
        y_pred (ndarray): Predicted target values
        lr_model (LinearRegression): Trained regression model
    """
    train_data = X_train_scaled.copy()
    train_data['AQI'] = y_train.values.ravel()
    train_data.to_csv('data/train_data.csv')

    #test data pal
    test_data = X_test_scaled.copy()
    test_data['AQI'] = y_test.values.ravel()
    test_data.to_csv('data/test_data.csv')


    #let's save the predictions to one csv
    results = pd.DataFrame({
        'Actual_AQI': y_test.values.ravel(),
        'Predicted_AQI': np.rint(y_pred.ravel().astype(int))
        
    })
    results['Predicted_AQI'] = results['Predicted_AQI'].astype(int)
    results.to_csv('data/linear_regression_predictions.csv', index=False)

    #coefficients to csv
    coefficients = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Coefficient': lr_model.coef_.ravel()
    })
    coefficients.to_csv('data/linear_regression_coefficients.csv', index=False)











