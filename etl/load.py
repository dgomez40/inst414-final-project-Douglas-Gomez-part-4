import pandas as pd
import etl.transform as transform

def load_data(X_train_scaled, X_test_scaled, y_train, y_test):
    model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred = transform.train_lin_reg()

    #trained data buddy
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
        'Predicted_AQI': y_pred.ravel()
    })
    results.to_csv('data/linear_regression_predictions.csv', index=False)

    #coefficients to csv
    coefficients = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Coefficient': model.coef_.ravel()
    })
    coefficients.to_csv('data/linear_regression_coefficients.csv', index=False)











