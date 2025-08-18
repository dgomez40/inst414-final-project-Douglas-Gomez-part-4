import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def model_evaluation(pollutant_merge, data_path='pollutant_merge_polish.csv', model_path="lr_model.pkl"):
    data = pollutant_merge
    lr_model = joblib.load(model_path)

    x = data[['temperature']]
    y = data['pm2.5']


    y_prediction = lr_model.predict(x)

    return y_prediction