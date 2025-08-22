import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib


def train_lin_reg(train_X_scaled, y_train, test_X_scaled, y_test ):
    """Creates linear regression model

    Args:
        train_X_scaled (ndarray): Scaled training features  
        y_train (Series): Training target values  
        test_X_scaled (ndarray): Scaled test features  
        y_test (Series): Test target values    
    """
    lr_model = LinearRegression()
    lr_model.fit(train_X_scaled, y_train)
    y_pred = lr_model.predict(test_X_scaled)
    results = pd.DataFrame({
    'Actual_AQI': y_test.values.ravel(), 
    'Predicted_AQI': y_pred.ravel()     
})
    joblib.dump(lr_model, 'data/models/linear_regression_model.joblib')
    return lr_model, train_X_scaled, test_X_scaled, y_train, y_test, y_pred


